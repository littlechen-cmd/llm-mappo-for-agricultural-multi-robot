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
    parser.add_argument(
        "--init-actor-checkpoint",
        default=None,
        help="Optional checkpoint whose shared Actor initializes this new stage.",
    )
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


def linear_decay(initial, episode, decay_episodes, final=0.0):
    if decay_episodes <= 0:
        return float(final)
    progress = min(max((episode - 1) / decay_episodes, 0.0), 1.0)
    return float(initial) + (float(final) - float(initial)) * progress


def curriculum_strength(initial, episode, warmup_episodes, decay_episodes, final=0.0):
    """Hold a teacher strength during warmup, then linearly anneal it."""

    if episode <= warmup_episodes:
        return float(initial)
    return linear_decay(initial, episode - warmup_episodes + 1, decay_episodes, final)


def build_environment(env_config, render_mode=None):
    return HeterogeneousWarehouse(
        size=env_config["size"],
        n_agvs=env_config["n_agvs"],
        request_queue_size=env_config["request_queue_size"],
        picking_duration=env_config.get("picking_duration", 2),
        max_steps=env_config["max_steps"],
        max_completed_tasks=env_config.get("max_completed_tasks"),
        terminate_on_death=env_config.get("terminate_on_death", False),
        allow_manual_unload=env_config.get("allow_manual_unload", False),
        render_mode=render_mode,
    )


def evaluate_actor(executor, config, device, episodes):
    """Evaluate the learned Actor without behavior-prior assistance."""

    env_config = config["environment"]
    env = build_environment(env_config)
    adapter = WarehouseStateAdapter(local_radius=config["model"]["local_radius"])
    planner = RulePlanner(**config["planner"])
    prior_policy = RuleBasedPriorPolicy(
        confidence=config.get("prior", {}).get("confidence", 0.9)
    )
    saved_mix = executor.prior_mixing_coefficient
    saved_kl = executor.prior_coefficient
    executor.set_prior_strength(0.0, 0.0)
    successes = 0
    native_returns = []
    deaths = 0
    try:
        for index in range(episodes):
            env.reset(seed=config["training"]["seed"] + 100_000 + index)
            decision = planner.plan(adapter.snapshot(env))
            state = adapter.build(env, decision, prior_policy)
            episode_return = 0.0
            event_counts = Counter()
            while True:
                action_output = executor.act(state, deterministic=True)
                _, rewards, terminated, truncated, info = env.step(
                    action_output.actions.tolist()
                )
                episode_return += float(np.mean(rewards))
                event_counts.update(event["type"] for event in info["events"])
                if terminated or truncated:
                    break
                if should_replan(
                    env._steps, info, config["training"]["planner_interval"]
                ):
                    decision = planner.plan(
                        adapter.snapshot(env, [event["type"] for event in info["events"]])
                    )
                state = adapter.build(env, decision, prior_policy)
            successes += int(event_counts["PICKING_COMPLETED"] > 0)
            deaths += event_counts["AGV_DEAD"]
            native_returns.append(episode_return)
    finally:
        executor.set_prior_strength(saved_mix, saved_kl)
        env.close()
    return {
        "success_rate": successes / max(episodes, 1),
        "native_return": float(np.mean(native_returns)),
        "deaths": deaths,
    }


def save_checkpoint(path, episode, config, executor):
    torch.save(
        {"episode": episode, "config": config, "executor": executor.state_dict()},
        path,
    )


def initialize_actor_from_checkpoint(executor, checkpoint_path, device):
    """Transfer only the shared Actor; reset critic and optimizer for a new stage."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"initial Actor checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    try:
        actor_state = checkpoint["executor"]["actor"]
    except KeyError as error:
        raise ValueError(f"checkpoint does not contain an executor Actor: {path}") from error
    executor.actor.load_state_dict(actor_state, strict=True)
    print(
        "initialized shared Actor from checkpoint={path} source_episode={episode}; "
        "critic and optimizer were reset for this curriculum stage".format(
            path=path, episode=checkpoint.get("episode", "unknown")
        )
    )


def train(config, device, init_actor_checkpoint=None):
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
    source_actor_checkpoint = init_actor_checkpoint or config["training"].get(
        "initial_actor_checkpoint"
    )
    if source_actor_checkpoint:
        initialize_actor_from_checkpoint(executor, source_actor_checkpoint, device)
    checkpoint_path = Path(config["training"]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = Path(
        config["training"].get(
            "best_checkpoint_path", checkpoint_path
        )
    )
    latest_checkpoint_path = Path(
        config["training"].get(
            "latest_checkpoint_path", checkpoint_path.with_stem(f"{checkpoint_path.stem}_latest")
        )
    )
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_actor_score = (-1.0, float("-inf"))

    for episode in range(1, config["training"]["episodes"] + 1):
        decay_episodes = config["training"].get("prior_decay_episodes")
        warmup_episodes = config["training"].get("prior_warmup_episodes", 0)
        if decay_episodes is None:
            executor.set_prior_strength(
                executor.config.prior_mixing_coefficient,
                executor.config.prior_coefficient,
            )
        else:
            executor.set_prior_strength(
                curriculum_strength(
                    executor.config.prior_mixing_coefficient,
                    episode,
                    warmup_episodes,
                    decay_episodes,
                    config["training"].get("prior_mixing_final", 0.0),
                ),
                curriculum_strength(
                    executor.config.prior_coefficient,
                    episode,
                    warmup_episodes,
                    decay_episodes,
                    config["training"].get("prior_kl_final_coefficient", 0.0),
                ),
            )
        env.reset(seed=config["training"]["seed"] + episode)
        decision = planner.plan(adapter.snapshot(env))
        state = adapter.build(env, decision, prior_policy)
        rollout = RolloutBuffer()
        shaper = LegalPathRewardShaper(
            progress_scale=config.get("reward", {}).get("path_progress_scale", 0.0),
            time_penalty=config.get("reward", {}).get("time_penalty", 0.0),
            load_bonus=config.get("reward", {}).get("load_bonus", 0.0),
            picking_start_bonus=config.get("reward", {}).get("picking_start_bonus", 0.0),
            unload_penalty=config.get("reward", {}).get("unload_penalty", 0.0),
        )
        shaper.reset(env, decision)
        native_return = 0.0
        shaping_return = 0.0
        shaped_return = 0.0
        event_counts = Counter()
        last_done = False

        while True:
            action_output = executor.act(state)
            _, rewards, terminated, truncated, info = env.step(action_output.actions.tolist())
            last_done = terminated or truncated
            native_team_reward = float(np.mean(rewards))
            task_shaping_reward = shaper.reward(
                env, decision, action_output.actions, info["events"]
            )
            team_reward = native_team_reward + task_shaping_reward
            native_return += native_team_reward
            shaping_return += task_shaping_reward
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
                decision = planner.plan(
                    adapter.snapshot(env, [event["type"] for event in info["events"]])
                )
                shaper.set_plan(env, decision)
            state = adapter.build(env, decision, prior_policy)
            if last_done:
                break

        metrics = executor.update(rollout, state.global_map, last_done)
        if episode % config["training"]["log_interval"] == 0:
            print(
                "episode={episode} shaped_return={shaped_return:.3f} "
                "native_return={native_return:.3f} task_shape_return={shaping_return:.3f} "
                "steps={steps} success={success} events="
                "pick_start:{pick_start},pick_done:{pick_done},loaded:{loaded},"
                "unloaded:{unloaded},charged:{charged},dead:{dead} battery_mean={battery_mean:.3f} "
                "prior_mix={prior_mix:.3f} prior_kl_coef={prior_kl_coef:.3f} "
                "actor_loss={actor_loss:.4f} critic_loss={critic_loss:.4f} "
                "entropy={entropy:.4f} prior_loss={prior_loss:.4f}".format(
                    episode=episode,
                    shaped_return=shaped_return,
                    native_return=native_return,
                    shaping_return=shaping_return,
                    steps=len(rollout),
                    success=int(event_counts["PICKING_COMPLETED"] > 0),
                    pick_start=event_counts["PICKING_STARTED"],
                    pick_done=event_counts["PICKING_COMPLETED"],
                    loaded=event_counts["SHELF_LOADED"],
                    unloaded=event_counts["SHELF_UNLOADED"],
                    charged=event_counts["CHARGED"],
                    dead=event_counts["AGV_DEAD"],
                    battery_mean=float(np.mean(info["battery"])),
                    prior_mix=executor.prior_mixing_coefficient,
                    prior_kl_coef=executor.prior_coefficient,
                    **metrics,
                )
            )
        actor_eval_interval = config["training"].get("actor_eval_interval", 0)
        actor_eval_episodes = config["training"].get("actor_eval_episodes", 0)
        if actor_eval_interval and actor_eval_episodes and episode % actor_eval_interval == 0:
            actor_evaluation = evaluate_actor(
                executor, config, device, actor_eval_episodes
            )
            print(
                "actor_eval episode={episode} success_rate={success_rate:.2%} "
                "native_return={native_return:.3f} deaths={deaths}".format(
                    episode=episode, **actor_evaluation
                )
            )
            actor_score = (
                actor_evaluation["success_rate"], actor_evaluation["native_return"]
            )
            if actor_score > best_actor_score:
                best_actor_score = actor_score
                save_checkpoint(best_checkpoint_path, episode, config, executor)
                print(f"saved best actor checkpoint: {best_checkpoint_path}")
        if episode % config["training"]["checkpoint_interval"] == 0:
            save_checkpoint(latest_checkpoint_path, episode, config, executor)

    save_checkpoint(latest_checkpoint_path, config["training"]["episodes"], config, executor)
    if best_actor_score[0] < 0.0:
        save_checkpoint(best_checkpoint_path, config["training"]["episodes"], config, executor)
    print(f"saved best actor checkpoint: {best_checkpoint_path}")
    print(f"saved latest training checkpoint: {latest_checkpoint_path}")


if __name__ == "__main__":
    arguments = parse_args()
    configuration = load_config(arguments.config)
    train(
        configuration,
        choose_device(arguments.device or configuration.get("device", "auto")),
        arguments.init_actor_checkpoint,
    )
