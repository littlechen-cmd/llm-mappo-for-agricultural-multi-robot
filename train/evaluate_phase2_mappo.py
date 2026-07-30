"""Held-out Actor-only evaluation for a Phase 2 MAPPO checkpoint."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RWARE_SOURCE = PROJECT_ROOT / "robotic-warehouse"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(RWARE_SOURCE) not in sys.path:
    sys.path.insert(0, str(RWARE_SOURCE))

from rware_llm.mappo import MAPPOConfig, MAPPOExecutor
from rware_llm.phase2_runtime import Phase2EpisodeRuntime
from train.train_phase2_mappo import (
    choose_device,
    evaluate_actor,
    load_checkpoint,
    load_config,
)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase2_medium_1ag_oracle.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def build_executor(config, device):
    runtime = Phase2EpisodeRuntime(config, use_rule_prior=False)
    try:
        state = runtime.reset(config["dynamic_tasks"]["heldout_seed_start"])
        return MAPPOExecutor(
            vector_dim=state.actor_vectors.shape[-1],
            local_channels=state.local_grids.shape[1],
            global_channels=state.global_map.shape[0],
            action_dim=runtime.env.action_space[0].n,
            config=MAPPOConfig(**config["mappo"]),
            device=device,
        )
    finally:
        runtime.close()


def main():
    arguments = parse_args()
    config = load_config(arguments.config)
    device = choose_device(arguments.device)
    executor = build_executor(config, device)
    load_checkpoint(arguments.checkpoint, executor, device)
    result = evaluate_actor(
        executor,
        config,
        device,
        episodes=arguments.episodes,
        seed_start=arguments.seed_start,
        traces=True,
    )
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(
        "actor-only fixed_completion={fixed_target_completion_rate:.2%} "
        "full_success={full_success_rate:.2%} deaths={death_count} "
        "deadlock={deadlock_episode_rate:.2%} output={output}".format(
            output=output_path, **result
        )
    )


if __name__ == "__main__":
    main()
