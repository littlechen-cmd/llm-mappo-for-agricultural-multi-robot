"""Run reproducible medium/small dynamic-environment stability validation."""

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

from rware.heterogeneous import HeterogeneousWarehouse
from rware_llm.dynamic_tasks import PoissonArrivalConfig, PoissonTaskController
from rware_llm.planner import RuleBasedPriorPolicy, RulePlanner
from rware_llm.state import WarehouseStateAdapter


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_medium_dynamic.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument(
        "--output", default="artifacts/phase1_medium_stability.json"
    )
    return parser.parse_args()


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_environment(values):
    return HeterogeneousWarehouse(
        size=values["size"],
        n_agvs=values["n_agvs"],
        n_pickers=values.get("n_pickers", 1),
        n_chargers=values.get("n_chargers"),
        request_queue_size=values["request_queue_size"],
        picking_duration=values.get("picking_duration", 2),
        max_steps=values["max_steps"],
        max_completed_tasks=values.get("max_completed_tasks"),
        continuous_task_generation=values.get("continuous_task_generation", False),
        randomize_initial_requests=values.get("randomize_initial_requests", True),
        initial_battery=values.get("initial_battery", 10.0),
        max_battery=values.get("max_battery", 10.0),
        safe_charge=values.get("safe_charge", 5.0),
    )


def run_validation(config, episodes, seed_start):
    if episodes < 1:
        raise ValueError("episodes must be at least one")
    env_values = config["environment"]
    dynamic_values = config["dynamic_tasks"]
    if seed_start is None:
        seed_start = dynamic_values["heldout_seed_start"]
    environment = build_environment(env_values)
    adapter = WarehouseStateAdapter()
    planner = RulePlanner(plan_horizon=20)
    prior = RuleBasedPriorPolicy(confidence=1.0)
    arrival_config = PoissonArrivalConfig(
        rate_per_step=dynamic_values["rate_per_step"],
        max_pending_requests=dynamic_values["max_pending_requests"],
    )
    results = []
    try:
        for episode_index in range(episodes):
            seed = seed_start + episode_index
            environment.reset(seed=seed)
            arrivals = PoissonTaskController(arrival_config)
            arrivals.reset(environment, seed=seed)
            decision = planner.plan(adapter.snapshot(environment))
            event_counts = Counter()
            while True:
                state = adapter.build(environment, decision, prior)
                actions = np.argmax(state.prior_action_probs, axis=1)
                _, _, terminated, truncated, info = environment.step(actions.tolist())
                event_counts.update(event["type"] for event in info["events"])
                arrivals.reconcile_request_queue(environment)
                released = arrivals.release_due(environment, environment._steps)
                if terminated or truncated:
                    break
                if (
                    environment._steps % 20 == 0
                    or info["events"]
                    or released
                ):
                    decision = planner.plan(
                        adapter.snapshot(
                            environment,
                            [event["type"] for event in info["events"]],
                        )
                    )
            results.append(
                {
                    "episode": episode_index + 1,
                    "seed": seed,
                    "steps": environment._steps,
                    "completed_tasks": environment.completed_tasks,
                    "deaths": event_counts["AGV_DEAD"],
                    "released_batch_count": len(arrivals.batches),
                    "released_task_ids": [
                        task_id for batch in arrivals.batches for task_id in batch.task_ids
                    ],
                    "event_counts": dict(sorted(event_counts.items())),
                }
            )
    finally:
        environment.close()
    return {
        "schema_version": "phase1-medium-stability-v1",
        "environment": env_values,
        "dynamic_tasks": dynamic_values,
        "seed_start": seed_start,
        "episodes": results,
        "summary": {
            "episode_count": episodes,
            "crash_count": 0,
            "death_count": sum(result["deaths"] for result in results),
            "mean_completed_tasks": float(
                np.mean([result["completed_tasks"] for result in results])
            ),
            "distinct_release_sequences": len(
                {tuple(result["released_task_ids"]) for result in results}
            ),
        },
    }


def main():
    arguments = parse_args()
    result = run_validation(
        load_config(arguments.config), arguments.episodes, arguments.seed_start
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(
        "saved={path} episodes={episodes} deaths={deaths} "
        "mean_completed_tasks={completed:.2f} distinct_release_sequences={sequences}".format(
            path=output,
            episodes=result["summary"]["episode_count"],
            deaths=result["summary"]["death_count"],
            completed=result["summary"]["mean_completed_tasks"],
            sequences=result["summary"]["distinct_release_sequences"],
        )
    )


if __name__ == "__main__":
    main()
