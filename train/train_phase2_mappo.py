"""Train Phase 2 MAPPO on the medium FIFO/static-A* local-execution oracle."""

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RWARE_SOURCE = PROJECT_ROOT / "robotic-warehouse"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(RWARE_SOURCE) not in sys.path:
    sys.path.insert(0, str(RWARE_SOURCE))

from rware_llm.mappo import MAPPOConfig, MAPPOExecutor
from rware_llm.mappo.buffer import RolloutBuffer
from rware_llm.phase2_runtime import Phase2EpisodeRuntime
from rware_llm.rewards import LegalPathRewardShaper


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase2_medium_1ag_oracle.yaml")
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    return parser.parse_args()


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requires a CUDA-enabled PyTorch installation")
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def curriculum_strength(initial, episode, warmup_episodes, decay_episodes, final):
    if episode <= warmup_episodes:
        return float(initial)
    if decay_episodes <= 0:
        return float(final)
    progress = min((episode - warmup_episodes) / decay_episodes, 1.0)
    return float(initial) + (float(final) - float(initial)) * progress


def training_seed_schedule(config, episodes):
    dynamic_config = config["dynamic_tasks"]
    generator = np.random.default_rng(config["training"]["seed"])
    return generator.integers(
        dynamic_config["train_seed_start"],
        dynamic_config["train_seed_start"] + dynamic_config["train_seed_count"],
        size=episodes,
        endpoint=False,
    ).tolist()


def build_shaper(config, env):
    reward = config["reward"]
    return LegalPathRewardShaper(
        progress_scale=reward["path_progress_scale"],
        time_penalty=reward["time_penalty"],
        load_bonus=reward["load_bonus"],
        picking_start_bonus=reward["picking_start_bonus"],
        unload_penalty=reward["unload_penalty"],
        safe_charge=reward["safe_charge"],
        safe_charge_streak_steps=reward["safe_charge_streak_steps"],
        safe_charge_reward=reward["safe_charge_reward"],
        low_battery_threshold=reward["low_battery_threshold"],
        low_battery_penalty=reward["low_battery_penalty"],
        collision_penalty=reward["collision_penalty"],
        wait_streak_steps=reward["wait_streak_steps"],
        wait_streak_penalty=reward["wait_streak_penalty"],
    )


def episode_result(runtime, event_counts, native_return, shaped_return, trace=None):
    target_tasks = runtime.config["environment"]["max_completed_tasks"]
    completed_tasks = runtime.env.completed_tasks
    diagnostics = runtime.diagnostics()
    execution_counts = diagnostics["execution_event_counts"]
    deadlock = bool(
        execution_counts.get("STALLED", 0) or execution_counts.get("BLOCKED", 0)
    )
    return {
        "completed_tasks": completed_tasks,
        "target_completed_tasks": target_tasks,
        "full_success": completed_tasks >= target_tasks,
        "fixed_target_completion_ratio": min(completed_tasks / target_tasks, 1.0),
        "released_task_count": diagnostics["released_task_count"],
        "released_task_completion_ratio": min(
            completed_tasks / max(diagnostics["released_task_count"], 1), 1.0
        ),
        "steps": runtime.env._steps,
        "native_return": native_return,
        "shaped_return": shaped_return,
        "deaths": event_counts["AGV_DEAD"],
        "collision_blocks": diagnostics["collision_blocks"],
        "deadlock": deadlock,
        "event_counts": dict(sorted(event_counts.items())),
        "execution_event_counts": execution_counts,
        "oracle_scheduling_failures": diagnostics["oracle_scheduling_failures"],
        "oracle_path_failures": diagnostics["oracle_path_failures"],
        "oracle_replans": diagnostics["oracle_replans"],
        "trace": trace,
    }


def evaluate_actor(executor, config, device, episodes=None, seed_start=None, traces=False):
    """Evaluate only the learned Actor over the held-out dynamic seed pool."""

    dynamic_config = config["dynamic_tasks"]
    episodes = episodes or dynamic_config["heldout_seed_count"]
    seed_start = (
        dynamic_config["heldout_seed_start"] if seed_start is None else seed_start
    )
    runtime = Phase2EpisodeRuntime(config, use_rule_prior=False)
    saved_strength = (
        executor.prior_mixing_coefficient,
        executor.prior_coefficient,
    )
    executor.set_prior_strength(0.0, 0.0)
    details = []
    try:
        for index in range(episodes):
            seed = seed_start + index
            state = runtime.reset(seed, capture_trace=traces)
            event_counts = Counter()
            native_return = 0.0
            while True:
                action_output = executor.act(state, deterministic=True)
                transition = runtime.step(action_output.actions, capture_trace=traces)
                native_return += float(np.mean(transition.rewards))
                event_counts.update(
                    event["type"] for event in transition.info["events"]
                )
                state = transition.state
                if transition.terminated or transition.truncated:
                    break
            result = episode_result(
                runtime,
                event_counts,
                native_return,
                native_return,
                trace=runtime.trace if traces and runtime.env.completed_tasks < config["environment"]["max_completed_tasks"] else None,
            )
            result.update({"episode": index + 1, "seed": seed})
            details.append(result)
    finally:
        executor.set_prior_strength(*saved_strength)
        runtime.close()
    return aggregate_evaluation(details, seed_start)


def aggregate_evaluation(details, seed_start):
    """Produce the Phase 2 Go/No-Go metrics from per-seed diagnostics."""

    full_successes = [item["full_success"] for item in details]
    completion_ratios = [item["fixed_target_completion_ratio"] for item in details]
    return {
        "mode": "actor-only",
        "prior_mixing_coefficient": 0.0,
        "episodes": len(details),
        "seed_start": seed_start,
        "target_completed_tasks": details[0]["target_completed_tasks"],
        "full_success_rate": float(np.mean(full_successes)),
        "fixed_target_completion_rate": float(np.mean(completion_ratios)),
        "seed_success_stddev": float(np.std(full_successes)),
        "mean_released_task_completion_rate": float(
            np.mean([item["released_task_completion_ratio"] for item in details])
        ),
        "mean_completed_tasks": float(np.mean([item["completed_tasks"] for item in details])),
        "mean_steps": float(np.mean([item["steps"] for item in details])),
        "death_count": sum(item["deaths"] for item in details),
        "mean_collision_blocks": float(
            np.mean([item["collision_blocks"] for item in details])
        ),
        "deadlock_episode_rate": float(np.mean([item["deadlock"] for item in details])),
        "oracle_scheduling_failures": sum(
            item["oracle_scheduling_failures"] for item in details
        ),
        "oracle_path_failures": sum(item["oracle_path_failures"] for item in details),
        "episodes_detail": details,
    }


def save_checkpoint(path, episode, config, executor, seed_schedule):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "phase": "phase2-medium-oracle-v1",
            "episode": episode,
            "config": config,
            "seed_schedule": seed_schedule,
            "executor": executor.state_dict(),
        },
        path,
    )


def load_checkpoint(path, executor, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    executor.load_state_dict(checkpoint["executor"])
    return checkpoint


def build_tensorboard_writers(config):
    tensorboard_config = config.get("tensorboard", {})
    if not tensorboard_config.get("enabled", True):
        return None, None
    from torch.utils.tensorboard import SummaryWriter

    root = Path(tensorboard_config["log_dir"])
    return (
        SummaryWriter(str(root / "train"), flush_secs=tensorboard_config["flush_secs"]),
        SummaryWriter(str(root / "eval"), flush_secs=tensorboard_config["flush_secs"]),
    )


def train(config, device, resume_checkpoint=None):
    training = config["training"]
    total_episodes = training["episodes"]
    set_seed(training["seed"])
    runtime = Phase2EpisodeRuntime(config, use_rule_prior=True)
    initial_state = runtime.reset(training_seed_schedule(config, 1)[0])
    executor = MAPPOExecutor(
        vector_dim=initial_state.actor_vectors.shape[-1],
        local_channels=initial_state.local_grids.shape[1],
        global_channels=initial_state.global_map.shape[0],
        action_dim=runtime.env.action_space[0].n,
        config=MAPPOConfig(**config["mappo"]),
        device=device,
    )
    seed_schedule = training_seed_schedule(config, total_episodes)
    start_episode = 1
    if resume_checkpoint:
        checkpoint = load_checkpoint(resume_checkpoint, executor, device)
        start_episode = int(checkpoint["episode"]) + 1
        saved_schedule = checkpoint.get("seed_schedule")
        if saved_schedule is not None and seed_schedule[: len(saved_schedule)] != saved_schedule:
            raise ValueError(
                "resume checkpoint seed schedule is not a prefix of the config schedule"
            )
        print(f"resumed full MAPPO checkpoint at episode={start_episode - 1}")

    train_writer, eval_writer = build_tensorboard_writers(config)
    checkpoint_path = Path(training["checkpoint_path"])
    latest_checkpoint_path = Path(training["latest_checkpoint_path"])
    best_score = (-1.0, -1.0)
    history = []
    try:
        for episode in range(start_episode, total_episodes + 1):
            executor.set_prior_strength(
                curriculum_strength(
                    config["mappo"]["prior_mixing_coefficient"],
                    episode,
                    training["prior_warmup_episodes"],
                    training["prior_decay_episodes"],
                    training["prior_mixing_final"],
                ),
                curriculum_strength(
                    config["mappo"]["prior_coefficient"],
                    episode,
                    training["prior_warmup_episodes"],
                    training["prior_decay_episodes"],
                    training["prior_kl_final_coefficient"],
                ),
            )
            seed = seed_schedule[episode - 1]
            state = runtime.reset(seed)
            decision = runtime.decision
            shaper = build_shaper(config, runtime.env)
            shaper.reset(runtime.env, decision)
            rollout = RolloutBuffer()
            event_counts = Counter()
            native_return = 0.0
            shaping_return = 0.0
            while True:
                action_output = executor.act(state)
                transition = runtime.step(action_output.actions)
                native_team_reward = float(np.mean(transition.rewards))
                task_shaping_reward = shaper.reward(
                    runtime.env,
                    decision,
                    action_output.actions,
                    transition.info["events"],
                    collision_blocks=transition.collision_blocks,
                )
                team_reward = native_team_reward + task_shaping_reward
                is_done = transition.terminated or transition.truncated
                rollout.add(
                    state,
                    action_output.actions,
                    action_output.log_probs,
                    action_output.value,
                    team_reward,
                    is_done,
                )
                event_counts.update(
                    event["type"] for event in transition.info["events"]
                )
                native_return += native_team_reward
                shaping_return += task_shaping_reward
                state = transition.state
                decision = transition.decision
                shaper.set_plan(runtime.env, decision)
                if is_done:
                    break
            metrics = executor.update(rollout, state.global_map, True)
            result = episode_result(
                runtime, event_counts, native_return, native_return + shaping_return
            )
            result.update(
                {
                    "episode": episode,
                    "seed": int(seed),
                    "prior_mixing_coefficient": executor.prior_mixing_coefficient,
                    "prior_kl_coefficient": executor.prior_coefficient,
                    "training_steps": len(rollout),
                    "metrics": metrics,
                }
            )
            history.append(result)
            if train_writer is not None:
                for key, value in {**metrics, **result}.items():
                    if isinstance(value, (float, int, bool)):
                        train_writer.add_scalar(key, value, episode)
            if episode % training["log_interval"] == 0:
                print(
                    "episode={episode} seed={seed} completion={completion:.2%} "
                    "full_success={full_success} steps={steps} native={native:.3f} "
                    "shape={shape:.3f} prior_mix={prior_mix:.3f} ratio={ratio:.3f} "
                    "kl={kl:.5f} entropy={entropy:.4f} grad_norm={grad_norm:.4f}".format(
                        episode=episode,
                        seed=seed,
                        completion=result["fixed_target_completion_ratio"],
                        full_success=int(result["full_success"]),
                        steps=result["steps"],
                        native=result["native_return"],
                        shape=result["shaped_return"] - result["native_return"],
                        prior_mix=executor.prior_mixing_coefficient,
                        ratio=metrics["policy_ratio"],
                        kl=metrics["approx_kl"],
                        entropy=metrics["entropy"],
                        grad_norm=metrics["grad_norm"],
                    )
                )
            if episode % training["checkpoint_interval"] == 0:
                save_checkpoint(latest_checkpoint_path, episode, config, executor, seed_schedule)
            if episode % training["actor_eval_interval"] == 0:
                evaluation = evaluate_actor(
                    executor,
                    config,
                    device,
                    episodes=training["actor_eval_episodes"],
                )
                if eval_writer is not None:
                    for key, value in evaluation.items():
                        if isinstance(value, (float, int, bool)):
                            eval_writer.add_scalar(key, value, episode)
                print(
                    "actor_eval episode={episode} fixed_completion={rate:.2%} "
                    "full_success={success:.2%} deadlock={deadlock:.2%} deaths={deaths}".format(
                        episode=episode,
                        rate=evaluation["fixed_target_completion_rate"],
                        success=evaluation["full_success_rate"],
                        deadlock=evaluation["deadlock_episode_rate"],
                        deaths=evaluation["death_count"],
                    )
                )
                score = (
                    evaluation["fixed_target_completion_rate"],
                    evaluation["full_success_rate"],
                )
                if score > best_score:
                    best_score = score
                    save_checkpoint(checkpoint_path, episode, config, executor, seed_schedule)
                    print(f"saved best actor checkpoint: {checkpoint_path}")
        save_checkpoint(latest_checkpoint_path, total_episodes, config, executor, seed_schedule)
        if best_score[0] < 0.0:
            save_checkpoint(checkpoint_path, total_episodes, config, executor, seed_schedule)
        report = {
            "phase": "phase2-medium-oracle-v1",
            "device": device,
            "episodes": total_episodes,
            "best_actor_score": list(best_score),
            "training_history": history,
        }
        diagnostics_path = Path(training["diagnostics_path"])
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        with diagnostics_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"saved best actor checkpoint: {checkpoint_path}")
        print(f"saved latest training checkpoint: {latest_checkpoint_path}")
        print(f"saved training diagnostics: {diagnostics_path}")
        return report
    finally:
        if train_writer is not None:
            train_writer.close()
        if eval_writer is not None:
            eval_writer.close()
        runtime.close()


def main():
    arguments = parse_args()
    config = load_config(arguments.config)
    if arguments.episodes is not None:
        config["training"]["episodes"] = arguments.episodes
    train(config, choose_device(arguments.device or config.get("device")), arguments.resume_checkpoint)


if __name__ == "__main__":
    main()
