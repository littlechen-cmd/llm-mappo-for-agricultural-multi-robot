"""Train the parameter-shared CTDE MAPPO executor with a RulePlanner."""

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
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


TRACKED_EVENT_TYPES = (
    "PICKING_STARTED",
    "PICKING_COMPLETED",
    "SHELF_LOADED",
    "SHELF_UNLOADED",
    "REQUEST_GENERATED",
    "CHARGED",
    "BATTERY_SAFETY_CROSSED",
    "AGV_DEAD",
)


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
    disruptive = {
        "AGV_DEAD",
        "PICKING_COMPLETED",
        "SHELF_LOADED",
        "SHELF_UNLOADED",
        "CHARGED",
        "BATTERY_SAFETY_CROSSED",
    }
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


def build_tensorboard_writers(config):
    """Create separate writers for rollout training and Actor-only evaluation."""

    tensorboard_config = config.get("tensorboard", {})
    if not tensorboard_config.get("enabled", True):
        return None, None, 0
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "TensorBoard logging is enabled, but the 'tensorboard' package is "
            "not installed. Install it with: python -m pip install tensorboard"
        ) from error

    log_dir = Path(tensorboard_config.get("log_dir", "runs/mappo"))
    run_name = tensorboard_config.get("run_name") or datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    run_dir = log_dir / run_name
    flush_secs = int(tensorboard_config.get("flush_secs", 15))
    flush_interval = int(tensorboard_config.get("flush_interval", 20))
    if flush_interval < 1:
        raise ValueError("tensorboard.flush_interval must be at least one")
    print(f"TensorBoard logs: {run_dir}")
    return (
        SummaryWriter(log_dir=str(run_dir / "train"), flush_secs=flush_secs),
        SummaryWriter(log_dir=str(run_dir / "eval"), flush_secs=flush_secs),
        flush_interval,
    )


def log_event_counts(writer, event_counts, step, prefix="events_count", divisor=1):
    """Record all tracked event types, including zero counts for quiet episodes."""

    for event_type in TRACKED_EVENT_TYPES:
        writer.add_scalar(
            f"{prefix}/{event_type}",
            event_counts[event_type] / divisor,
            step,
        )


def log_training_scalars(
    writer,
    episode,
    native_return,
    completed_tasks,
    target_completed_tasks,
    event_counts,
    battery_mean,
    prior_mix,
    metrics,
):
    """Write per-episode learning and environment measurements."""

    success = int(completed_tasks >= target_completed_tasks)
    writer.add_scalar("native_return", native_return, episode)
    writer.add_scalar("success_rate", success, episode)
    writer.add_scalar("deaths", event_counts["AGV_DEAD"], episode)
    writer.add_scalar("prior_mix", prior_mix, episode)
    writer.add_scalar("entropy", metrics["entropy"], episode)
    writer.add_scalar("critic_loss", metrics["critic_loss"], episode)
    writer.add_scalar("actor_loss", metrics["actor_loss"], episode)
    writer.add_scalar("battery_mean", battery_mean, episode)
    writer.add_scalar("completed_tasks", completed_tasks, episode)
    writer.add_scalar(
        "task_completion_ratio", completed_tasks / target_completed_tasks, episode
    )
    writer.add_scalar("shaped_return", metrics["shaped_return"], episode)
    writer.add_scalar("task_shaping_return", metrics["task_shaping_return"], episode)
    writer.add_scalar("prior_loss", metrics["prior_loss"], episode)
    log_event_counts(writer, event_counts, episode)


def log_evaluation_scalars(writer, episode, evaluation):
    """Write deterministic Actor-only evaluation measurements.

    Actor and Critic losses are intentionally absent: evaluation takes no PPO
    update, so those optimization losses have no valid evaluation meaning.
    """

    evaluated_episodes = evaluation["episodes"]
    writer.add_scalar("native_return", evaluation["native_return"], episode)
    writer.add_scalar("success_rate", evaluation["full_success_rate"], episode)
    writer.add_scalar("deaths", evaluation["deaths"] / evaluated_episodes, episode)
    writer.add_scalar("deaths_total", evaluation["deaths"], episode)
    writer.add_scalar("prior_mix", 0.0, episode)
    writer.add_scalar("entropy", evaluation["entropy"], episode)
    writer.add_scalar("battery_mean", evaluation["battery_mean"], episode)
    writer.add_scalar(
        "completed_tasks", evaluation["mean_completed_tasks"], episode
    )
    writer.add_scalar(
        "task_completion_ratio", evaluation["task_completion_ratio"], episode
    )
    writer.add_scalar(
        "mean_steps_to_full_completion",
        evaluation["mean_steps_to_full_completion"],
        episode,
    )
    log_event_counts(
        writer,
        evaluation["event_counts"],
        episode,
        divisor=evaluated_episodes,
    )
    log_event_counts(
        writer,
        evaluation["event_counts"],
        episode,
        prefix="events_total",
    )


def build_environment(env_config, render_mode=None):
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
    target_completed_tasks = env_config.get("max_completed_tasks")
    if target_completed_tasks is None:
        raise ValueError("Actor evaluation requires environment.max_completed_tasks")
    full_successes = 0
    native_returns = []
    completed_task_counts = []
    completion_steps = []
    battery_samples = []
    entropies = []
    total_event_counts = Counter()
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
                entropies.append(action_output.entropy)
                battery_samples.append(float(np.mean(info["battery"])))
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
            completed_tasks = env.completed_tasks
            full_successes += int(completed_tasks >= target_completed_tasks)
            deaths += event_counts["AGV_DEAD"]
            native_returns.append(episode_return)
            completed_task_counts.append(completed_tasks)
            total_event_counts.update(event_counts)
            if completed_tasks >= target_completed_tasks:
                completion_steps.append(env._steps)
    finally:
        executor.set_prior_strength(saved_mix, saved_kl)
        env.close()
    return {
        "target_completed_tasks": target_completed_tasks,
        "full_success_rate": full_successes / max(episodes, 1),
        "mean_completed_tasks": float(np.mean(completed_task_counts)),
        "task_completion_ratio": float(
            np.mean(completed_task_counts) / target_completed_tasks
        ),
        "native_return": float(np.mean(native_returns)),
        "deaths": deaths,
        "battery_mean": float(np.mean(battery_samples)),
        "entropy": float(np.mean(entropies)),
        "event_counts": total_event_counts,
        "episodes": episodes,
        "mean_steps_to_full_completion": (
            float(np.mean(completion_steps)) if completion_steps else 0.0
        ),
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
    target_completed_tasks = env_config.get("max_completed_tasks")
    if target_completed_tasks is None:
        raise ValueError("training requires environment.max_completed_tasks")
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
        config["training"].get("best_checkpoint_path", checkpoint_path)
    )
    latest_checkpoint_path = Path(
        config["training"].get(
            "latest_checkpoint_path",
            checkpoint_path.with_stem(f"{checkpoint_path.stem}_latest"),
        )
    )
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_actor_score = (-1.0, float("-inf"))
    train_writer, evaluation_writer, tensorboard_flush_interval = (
        build_tensorboard_writers(config)
    )

    try:
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
                picking_start_bonus=config.get("reward", {}).get(
                    "picking_start_bonus", 0.0
                ),
                unload_penalty=config.get("reward", {}).get("unload_penalty", 0.0),
                safe_charge=config.get("reward", {}).get("safe_charge", env.safe_charge),
                safe_charge_streak_steps=config.get("reward", {}).get(
                    "safe_charge_streak_steps", 8
                ),
                safe_charge_reward=config.get("reward", {}).get(
                    "safe_charge_reward", 0.0
                ),
            )
            shaper.reset(env, decision)
            native_return = 0.0
            shaping_return = 0.0
            shaped_return = 0.0
            event_counts = Counter()
            battery_samples = []
            last_done = False

            while True:
                action_output = executor.act(state)
                _, rewards, terminated, truncated, info = env.step(
                    action_output.actions.tolist()
                )
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
                battery_samples.append(float(np.mean(info["battery"])))
                rollout.add(
                    state,
                    action_output.actions,
                    action_output.log_probs,
                    action_output.value,
                    team_reward,
                    last_done,
                )
                if should_replan(
                    env._steps, info, config["training"]["planner_interval"]
                ):
                    decision = planner.plan(
                        adapter.snapshot(
                            env, [event["type"] for event in info["events"]]
                        )
                    )
                    shaper.set_plan(env, decision)
                state = adapter.build(env, decision, prior_policy)
                if last_done:
                    break

            metrics = executor.update(rollout, state.global_map, last_done)
            battery_mean = float(np.mean(battery_samples))
            completed_tasks = env.completed_tasks
            tensorboard_metrics = {
                **metrics,
                "shaped_return": shaped_return,
                "task_shaping_return": shaping_return,
            }
            if train_writer is not None:
                log_training_scalars(
                    train_writer,
                    episode,
                    native_return,
                    completed_tasks,
                    target_completed_tasks,
                    event_counts,
                    battery_mean,
                    executor.prior_mixing_coefficient,
                    tensorboard_metrics,
                )
                if episode % tensorboard_flush_interval == 0:
                    train_writer.flush()

            if episode % config["training"]["log_interval"] == 0:
                print(
                    "episode={episode} shaped_return={shaped_return:.3f} "
                    "native_return={native_return:.3f} task_shape_return={shaping_return:.3f} "
                    "steps={steps} completed_tasks={completed_tasks}/{target_completed_tasks} "
                    "full_success={full_success} events="
                    "pick_start:{pick_start},pick_done:{pick_done},loaded:{loaded},"
                    "unloaded:{unloaded},requests:{requests},charged:{charged},"
                    "safe_cross:{safe_cross},dead:{dead} "
                    "battery_mean={battery_mean:.3f} "
                    "prior_mix={prior_mix:.3f} prior_kl_coef={prior_kl_coef:.3f} "
                    "actor_loss={actor_loss:.4f} critic_loss={critic_loss:.4f} "
                    "entropy={entropy:.4f} prior_loss={prior_loss:.4f}".format(
                        episode=episode,
                        shaped_return=shaped_return,
                        native_return=native_return,
                        shaping_return=shaping_return,
                        steps=len(rollout),
                        completed_tasks=completed_tasks,
                        target_completed_tasks=target_completed_tasks,
                        full_success=int(completed_tasks >= target_completed_tasks),
                        pick_start=event_counts["PICKING_STARTED"],
                        pick_done=event_counts["PICKING_COMPLETED"],
                        loaded=event_counts["SHELF_LOADED"],
                        unloaded=event_counts["SHELF_UNLOADED"],
                        requests=event_counts["REQUEST_GENERATED"],
                        charged=event_counts["CHARGED"],
                        safe_cross=event_counts["BATTERY_SAFETY_CROSSED"],
                        dead=event_counts["AGV_DEAD"],
                        battery_mean=battery_mean,
                        prior_mix=executor.prior_mixing_coefficient,
                        prior_kl_coef=executor.prior_coefficient,
                        **metrics,
                    )
                )

            # Persist parameters before evaluation so an evaluation exception
            # cannot discard a completed checkpoint interval.
            if episode % config["training"]["checkpoint_interval"] == 0:
                save_checkpoint(latest_checkpoint_path, episode, config, executor)

            actor_eval_interval = config["training"].get("actor_eval_interval", 0)
            actor_eval_episodes = config["training"].get("actor_eval_episodes", 0)
            if (
                actor_eval_interval
                and actor_eval_episodes
                and episode % actor_eval_interval == 0
            ):
                actor_evaluation = evaluate_actor(
                    executor, config, device, actor_eval_episodes
                )
                if evaluation_writer is not None:
                    log_evaluation_scalars(evaluation_writer, episode, actor_evaluation)
                    evaluation_writer.flush()
                print(
                    "actor_eval episode={episode} full_success_rate="
                    "{full_success_rate:.2%} mean_completed_tasks="
                    "{mean_completed_tasks:.2f}/{target_completed_tasks} "
                    "completion_ratio={task_completion_ratio:.2%} "
                    "native_return={native_return:.3f} deaths={deaths}".format(
                        episode=episode, **actor_evaluation
                    )
                )
                actor_score = (
                    actor_evaluation["task_completion_ratio"],
                    actor_evaluation["full_success_rate"],
                    actor_evaluation["native_return"],
                )
                if actor_score > best_actor_score:
                    best_actor_score = actor_score
                    save_checkpoint(best_checkpoint_path, episode, config, executor)
                    print(f"saved best actor checkpoint: {best_checkpoint_path}")

        save_checkpoint(
            latest_checkpoint_path, config["training"]["episodes"], config, executor
        )
        if best_actor_score[0] < 0.0:
            save_checkpoint(
                best_checkpoint_path, config["training"]["episodes"], config, executor
            )
        print(f"saved best actor checkpoint: {best_checkpoint_path}")
        print(f"saved latest training checkpoint: {latest_checkpoint_path}")
    finally:
        if train_writer is not None:
            train_writer.close()
        if evaluation_writer is not None:
            evaluation_writer.close()
        env.close()


if __name__ == "__main__":
    arguments = parse_args()
    configuration = load_config(arguments.config)
    train(
        configuration,
        choose_device(arguments.device or configuration.get("device", "auto")),
        arguments.init_actor_checkpoint,
    )
