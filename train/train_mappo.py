"""Train the parameter-shared CTDE MAPPO executor with a RulePlanner."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml

# The unified repository vendors RWARE as source rather than a Git submodule.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RWARE_SOURCE = PROJECT_ROOT / "robotic-warehouse"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(RWARE_SOURCE) not in sys.path:
    sys.path.insert(0, str(RWARE_SOURCE))

from rware.heterogeneous import HeterogeneousWarehouse
from rware_llm.mappo import MAPPOConfig, MAPPOExecutor
from rware_llm.mappo.buffer import RolloutBuffer
from rware_llm.planner import RuleBasedPriorPolicy, RulePlanner
from rware_llm.rewards import LegalPathRewardShaper
from rware_llm.state import WarehouseStateAdapter


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mappo_tiny_2ag.yaml")
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but this Python environment has no "
            "available CUDA device. Use --device cpu on the MateBook or run "
            "the CUDA-enabled PyTorch environment on the RTX 4080 Super server."
        )
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def should_replan(step, info, interval):
    event_types = {event["type"] for event in info.get("events", [])}
    disruptive = {"AGV_DEAD", "PICKING_COMPLETED", "SHELF_LOADED", "SHELF_UNLOADED"}
    return step % interval == 0 or bool(event_types & disruptive)


def linear_decay(initial, episode, decay_episodes):
    if decay_episodes <= 0:
        return 0.0
    return float(initial) * max(0.0, 1.0 - (episode - 1) / decay_episodes)


def build_environment(env_config, render_mode=None):
    return HeterogeneousWarehouse(
        size=env_config["size"],
        n_agvs=env_config["n_agvs"],
        request_queue_size=env_config["request_queue_size"],
        picking_duration=env_config.get("picking_duration", 2),
        max_steps=env_config["max_steps"],
        max_completed_tasks=env_config.get("max_completed_tasks"),
        terminate_on_death=env_config.get("terminate_on_death", False),
        render_mode=render_mode,
    )


def train(config, device):
    set_seed(config["training"]["seed"])
    env_config = config["environment"]
    env = build_environment(env_config)
    adapter = WarehouseStateAdapter(local_radius=config["model"]["local_radius"])
    planner = RulePlanner(**config["planner"])
    prior_policy = RuleBasedPriorPolicy(
        confidence=config.get("prior", {}).get("confidence", 0.9)
    )
    env.reset(seed=config["training"]["seed"])
    initial_plan = planner.plan(adapter.snapshot(env))
    initial_state = adapter.build(env, initial_plan, prior_policy)
    executor = MAPPOExecutor(
        vector_dim=initial_state.actor_vectors.shape[-1],
        local_channels=initial_state.local_grids.shape[1],
        global_channels=initial_state.global_map.shape[0],
        action_dim=env.action_space[0].n,
        config=MAPPOConfig(**config["mappo"]),
        device=device,
    )
    checkpoint_path = Path(config["training"]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for episode in range(1, config["training"]["episodes"] + 1):
        decay_episodes = config["training"].get("prior_decay_episodes")
        if decay_episodes is None:
            executor.set_prior_strength(
                executor.config.prior_mixing_coefficient,
                executor.config.prior_coefficient,
            )
        else:
            executor.set_prior_strength(
                linear_decay(
                    executor.config.prior_mixing_coefficient, episode, decay_episodes
                ),
                linear_decay(
                    executor.config.prior_coefficient, episode, decay_episodes),
            )
        env.reset(seed=config["training"]["seed"] + episode)
        decision = planner.plan(adapter.snapshot(env))
        state = adapter.build(env, decision, prior_policy)
        rollout = RolloutBuffer()
        shaper = LegalPathRewardShaper(
            progress_scale=config.get("reward", {}).get("path_progress_scale", 0.0)
        )
        shaper.reset(env, decision)
        native_return = 0.0
        progress_return = 0.0
        shaped_return = 0.0
        event_counts = Counter()
        last_done = False

        while True:
            action_output = executor.act(state)
            _, rewards, terminated, truncated, info = env.step(action_output.actions.tolist())
            last_done = terminated or truncated
            native_team_reward = float(np.mean(rewards))
            path_progress_reward = shaper.reward(env, decision)
            team_reward = native_team_reward + path_progress_reward
            native_return += native_team_reward
            progress_return += path_progress_reward
            shaped_return += team_reward
            event_counts.update(event["type"] for event in info["events"])
            rollout.add(
                state,
                action_output.actions,
                action_output.log_probs,
                action_output.value,
                team_reward,
                last_done,
            )
            if should_replan(env._steps, info, config["training"]["planner_interval"]):
                decision = planner.plan(adapter.snapshot(env, [event["type"] for event in info["events"]]))
                shaper.set_plan(env, decision)
            state = adapter.build(env, decision, prior_policy)
            if last_done:
                break

        metrics = executor.update(rollout, state.global_map, last_done)
        if episode % config["training"]["log_interval"] == 0:
            print(
                "episode={episode} shaped_return={shaped_return:.3f} "
                "native_return={native_return:.3f} progress_return={progress_return:.3f} "
                "steps={steps} success={success} events="
                "pick_start:{pick_start},pick_done:{pick_done},loaded:{loaded},"
                "charged:{charged},dead:{dead} battery_mean={battery_mean:.3f} "
                "prior_mix={prior_mix:.3f} prior_kl_coef={prior_kl_coef:.3f} "
                "actor_loss={actor_loss:.4f} critic_loss={critic_loss:.4f} "
                "entropy={entropy:.4f} prior_loss={prior_loss:.4f}".format(
                    episode=episode,
                    shaped_return=shaped_return,
                    native_return=native_return,
                    progress_return=progress_return,
                    steps=len(rollout),
                    success=int(event_counts["PICKING_COMPLETED"] > 0),
                    pick_start=event_counts["PICKING_STARTED"],
                    pick_done=event_counts["PICKING_COMPLETED"],
                    loaded=event_counts["SHELF_LOADED"],
                    charged=event_counts["CHARGED"],
                    dead=event_counts["AGV_DEAD"],
                    battery_mean=float(np.mean(info["battery"])),
                    prior_mix=executor.prior_mixing_coefficient,
                    prior_kl_coef=executor.prior_coefficient,
                    **metrics,
                )
            )
        if episode % config["training"]["checkpoint_interval"] == 0:
            torch.save(
                {"episode": episode, "config": config, "executor": executor.state_dict()},
                checkpoint_path,
            )

    torch.save(
        {
            "episode": config["training"]["episodes"],
            "config": config,
            "executor": executor.state_dict(),
        },
        checkpoint_path,
    )
    print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    arguments = parse_args()
    configuration = load_config(arguments.config)
    train(configuration, choose_device(arguments.device or configuration.get("device", "auto")))
