"""Evaluate a MAPPO checkpoint and optionally render a warehouse episode."""

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import sys
from time import sleep

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RWARE_SOURCE = PROJECT_ROOT / "robotic-warehouse"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(RWARE_SOURCE) not in sys.path:
    sys.path.insert(0, str(RWARE_SOURCE))

from rware.heterogeneous import HeterogeneousWarehouse
from rware_llm.mappo import MAPPOConfig, MAPPOExecutor
from rware_llm.planner import RuleBasedPriorPolicy, RulePlanner
from rware_llm.state import WarehouseStateAdapter


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mappo_tiny_1ag_curriculum.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="First fixed evaluation seed; defaults to training.seed + 10000.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file for the aggregate and per-episode diagnostics.",
    )
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument(
        "--prior-mix",
        type=float,
        default=0.0,
        help="0 evaluates the learned Actor only; larger values add stronger rule-prior guidance.",
    )
    parser.add_argument(
        "--rule-prior-only",
        action="store_true",
        help="Execute the deterministic RuleBasedPriorPolicy for environment visualization.",
    )
    return parser.parse_args()


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requires a CUDA-enabled PyTorch installation")
    return requested if requested != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_env(env_config, render_mode=None):
    return HeterogeneousWarehouse(
        size=env_config["size"],
        n_agvs=env_config["n_agvs"],
        n_pickers=env_config.get("n_pickers", 1),
        n_chargers=env_config.get("n_chargers"),
        request_queue_size=env_config["request_queue_size"],
        picking_duration=env_config.get("picking_duration", 2),
        max_steps=env_config["max_steps"],
        max_completed_tasks=env_config.get("max_completed_tasks"),
        terminate_on_death=env_config.get("terminate_on_death", False),
        allow_manual_unload=env_config.get("allow_manual_unload", False),
        randomize_initial_requests=env_config.get("randomize_initial_requests", False),
        initial_battery=env_config.get("initial_battery", 10.0),
        max_battery=env_config.get("max_battery", 10.0),
        safe_charge=env_config.get("safe_charge", 5.0),
        render_mode=render_mode,
    )


def evaluate(
    config,
    checkpoint_path,
    episodes,
    device,
    render=False,
    delay=0.15,
    prior_mix=0.0,
    rule_prior_only=False,
    seed_start=None,
):
    env_config = config["environment"]
    env = make_env(env_config, "human" if render else None)
    adapter = WarehouseStateAdapter(local_radius=config["model"]["local_radius"])
    planner = RulePlanner(**config["planner"])
    prior = RuleBasedPriorPolicy(config.get("prior", {}).get("confidence", 0.9))
    env.reset(seed=config["training"]["seed"])
    decision = planner.plan(adapter.snapshot(env))
    state = adapter.build(env, decision, prior)
    executor = MAPPOExecutor(
        vector_dim=state.actor_vectors.shape[-1],
        local_channels=state.local_grids.shape[1],
        global_channels=state.global_map.shape[0],
        action_dim=env.action_space[0].n,
        config=MAPPOConfig(**config["mappo"]),
        device=device,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    executor.load_state_dict(checkpoint["executor"])
    executor.set_prior_strength(prior_mix, 0.0)

    target_completed_tasks = env_config.get("max_completed_tasks")
    if target_completed_tasks is None:
        raise ValueError("evaluation requires environment.max_completed_tasks")
    if seed_start is None:
        seed_start = config["training"]["seed"] + 10_000
    successes = 0
    native_returns = []
    completion_steps = []
    total_deaths = 0
    completed_task_counts = []
    battery_means = []
    episode_diagnostics = []
    for episode in range(episodes):
        seed = seed_start + episode
        env.reset(seed=seed)
        decision = planner.plan(adapter.snapshot(env))
        state = adapter.build(env, decision, prior)
        events = Counter()
        native_return = 0.0
        battery_samples = []
        if render:
            env.render()
            sleep(delay)
        while True:
            if rule_prior_only:
                actions = np.argmax(state.prior_action_probs, axis=1)
            else:
                actions = executor.act(state, deterministic=True).actions
            _, rewards, terminated, truncated, info = env.step(actions.tolist())
            native_return += float(np.mean(rewards))
            events.update(event["type"] for event in info["events"])
            battery_samples.append(float(np.mean(info["battery"])))
            if render:
                env.render()
                print(
                    f"episode={episode + 1} step={info['steps']} "
                    f"actions={actions.tolist()} events={[event['type'] for event in info['events']]}"
                )
                sleep(delay)
            if terminated or truncated:
                break
            event_types = {event["type"] for event in info["events"]}
            if env._steps % config["training"]["planner_interval"] == 0 or event_types & {
                "AGV_DEAD",
                "PICKING_COMPLETED",
                "SHELF_LOADED",
                "SHELF_UNLOADED",
            }:
                decision = planner.plan(adapter.snapshot(env, [event["type"] for event in info["events"]]))
            state = adapter.build(env, decision, prior)
        completed_tasks = env.completed_tasks
        success = completed_tasks >= target_completed_tasks
        successes += int(success)
        total_deaths += events["AGV_DEAD"]
        native_returns.append(native_return)
        completed_task_counts.append(completed_tasks)
        battery_mean = float(np.mean(battery_samples))
        battery_means.append(battery_mean)
        if success:
            completion_steps.append(env._steps)
        episode_diagnostics.append(
            {
                "episode": episode + 1,
                "seed": seed,
                "full_success": bool(success),
                "completed_tasks": completed_tasks,
                "steps": env._steps,
                "native_return": native_return,
                "deaths": events["AGV_DEAD"],
                "battery_mean": battery_mean,
                "event_counts": dict(sorted(events.items())),
            }
        )
        print(
            f"episode={episode + 1} full_success={int(success)} "
            f"completed_tasks={completed_tasks}/{target_completed_tasks} "
            f"steps={env._steps} native_return={native_return:.3f} "
            f"deaths={events['AGV_DEAD']}"
        )
    summary = {
        "episodes": episodes,
        "seed_start": seed_start,
        "target_completed_tasks": target_completed_tasks,
        "full_success_rate": successes / episodes,
        "mean_completed_tasks": float(np.mean(completed_task_counts)),
        "task_completion_ratio": float(
            np.mean(completed_task_counts) / target_completed_tasks
        ),
        "mean_native_return": float(np.mean(native_returns)),
        "death_count": total_deaths,
        "mean_battery": float(np.mean(battery_means)),
        "mean_steps_to_completion": (
            float(np.mean(completion_steps)) if completion_steps else None
        ),
        "episodes_detail": episode_diagnostics,
    }
    print(
        "summary full_success_rate={full_success_rate:.2%} "
        "mean_completed_tasks={mean_completed_tasks:.2f}/{target_completed_tasks} "
        "mean_native_return={mean_native_return:.3f} death_count={death_count} "
        "mean_steps_to_completion={mean_steps_to_completion}".format(
            **summary
        )
    )
    env.close()
    return summary


if __name__ == "__main__":
    arguments = parse_args()
    configuration = load_config(arguments.config)
    summary = evaluate(
        configuration,
        arguments.checkpoint,
        arguments.episodes,
        choose_device(arguments.device),
        arguments.render,
        arguments.delay,
        arguments.prior_mix,
        arguments.rule_prior_only,
        arguments.seed_start,
    )
    if arguments.output:
        output_path = Path(arguments.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"saved evaluation diagnostics: {output_path}")
