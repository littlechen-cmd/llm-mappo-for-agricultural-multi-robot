"""Validate the Phase 2 FIFO/static-A* oracle before MAPPO training."""

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RWARE_SOURCE = PROJECT_ROOT / "robotic-warehouse"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(RWARE_SOURCE) not in sys.path:
    sys.path.insert(0, str(RWARE_SOURCE))

from rware_llm.phase2_runtime import Phase2EpisodeRuntime


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase2_medium_1ag_oracle.yaml")
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--output", default="artifacts/phase2_medium_1ag_oracle_baseline.json"
    )
    return parser.parse_args()


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_oracle(config, seed_start, episodes):
    """Run the deterministic rule teacher through the same Phase 2 runtime."""

    target_tasks = config["environment"]["max_completed_tasks"]
    runtime = Phase2EpisodeRuntime(config, use_rule_prior=True)
    details = []
    try:
        for episode_index in range(episodes):
            seed = seed_start + episode_index
            state = runtime.reset(seed, capture_trace=True)
            environment_events = Counter()
            execution_events = Counter()
            while True:
                actions = np.argmax(state.prior_action_probs, axis=1)
                transition = runtime.step(actions, capture_trace=True)
                environment_events.update(
                    event["type"] for event in transition.info["events"]
                )
                execution_events.update(
                    event.event_type.value for event in transition.execution_events
                )
                state = transition.state
                if transition.terminated or transition.truncated:
                    break
            diagnostics = runtime.diagnostics()
            completed_tasks = runtime.env.completed_tasks
            full_success = completed_tasks >= target_tasks
            details.append(
                {
                    "episode": episode_index + 1,
                    "seed": seed,
                    "full_success": full_success,
                    "completed_tasks": completed_tasks,
                    "fixed_target_completion_ratio": min(
                        completed_tasks / target_tasks, 1.0
                    ),
                    "released_task_count": diagnostics["released_task_count"],
                    "released_task_completion_ratio": completed_tasks
                    / max(diagnostics["released_task_count"], 1),
                    "steps": runtime.env._steps,
                    "deaths": environment_events["AGV_DEAD"],
                    "collision_blocks": diagnostics["collision_blocks"],
                    "event_counts": dict(sorted(environment_events.items())),
                    "execution_event_counts": dict(sorted(execution_events.items())),
                    "oracle_scheduling_failures": diagnostics[
                        "oracle_scheduling_failures"
                    ],
                    "oracle_path_failures": diagnostics["oracle_path_failures"],
                    "oracle_replans": diagnostics["oracle_replans"],
                    "failure_trace": runtime.trace if not full_success else None,
                }
            )
    finally:
        runtime.close()

    return {
        "mode": "rule-prior-only oracle contract validation; not MAPPO acceptance",
        "episodes": episodes,
        "seed_start": seed_start,
        "target_completed_tasks": target_tasks,
        "full_success_rate": float(np.mean([item["full_success"] for item in details])),
        "mean_fixed_target_completion_ratio": float(
            np.mean([item["fixed_target_completion_ratio"] for item in details])
        ),
        "mean_released_task_completion_ratio": float(
            np.mean([item["released_task_completion_ratio"] for item in details])
        ),
        "mean_completed_tasks": float(np.mean([item["completed_tasks"] for item in details])),
        "death_count": sum(item["deaths"] for item in details),
        "collision_blocks": sum(item["collision_blocks"] for item in details),
        "oracle_scheduling_failures": sum(
            item["oracle_scheduling_failures"] for item in details
        ),
        "oracle_path_failures": sum(item["oracle_path_failures"] for item in details),
        "episodes_detail": details,
    }


def main():
    arguments = parse_args()
    config = load_config(arguments.config)
    dynamic_config = config["dynamic_tasks"]
    seed_start = arguments.seed_start
    if seed_start is None:
        seed_start = dynamic_config["heldout_seed_start"]
    episodes = arguments.episodes or dynamic_config["heldout_seed_count"]
    result = validate_oracle(config, seed_start, episodes)
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(
        "oracle full_success_rate={full_success_rate:.2%} fixed_target_ratio="
        "{mean_fixed_target_completion_ratio:.2%} released_task_ratio="
        "{mean_released_task_completion_ratio:.2%} deaths={death_count} "
        "path_failures={oracle_path_failures} output={output}".format(
            output=output_path, **result
        )
    )


if __name__ == "__main__":
    main()
