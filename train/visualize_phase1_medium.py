"""Export a dynamic medium RWARE rule-driven validation episode as a GIF."""

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RWARE_SOURCE = PROJECT_ROOT / "robotic-warehouse"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(RWARE_SOURCE) not in sys.path:
    sys.path.insert(0, str(RWARE_SOURCE))

from rware_llm.dynamic_tasks import PoissonArrivalConfig, PoissonTaskController
from rware_llm.planner import RuleBasedPriorPolicy, RulePlanner
from rware_llm.state import WarehouseStateAdapter
from train.validate_phase1_medium import build_environment, load_config
from train.visualize_tiny_2ag import draw_rollout_frame


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_medium_dynamic.yaml")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--output", default="artifacts/phase1_medium_seed2000.gif")
    return parser.parse_args()


def export_episode(config, seed, frame_stride, fps, output_path):
    if frame_stride < 1 or fps < 1:
        raise ValueError("frame_stride and fps must be at least one")
    env = build_environment(config["environment"])
    dynamic = config["dynamic_tasks"]
    controller = PoissonTaskController(
        PoissonArrivalConfig(
            rate_per_step=dynamic["rate_per_step"],
            max_pending_requests=dynamic["max_pending_requests"],
        )
    )
    adapter = WarehouseStateAdapter()
    planner = RulePlanner(plan_horizon=20)
    prior = RuleBasedPriorPolicy(confidence=1.0)
    try:
        env.reset(seed=seed)
        controller.reset(env, seed=seed)
        decision = planner.plan(adapter.snapshot(env))
        frames = [
            draw_rollout_frame(
                env,
                decision,
                seed,
                ["TASK_BATCH_RELEASED"],
                title="medium dynamic warehouse | rule validation",
            )
        ]
        while True:
            state = adapter.build(env, decision, prior)
            actions = np.argmax(state.prior_action_probs, axis=1)
            _, _, terminated, truncated, info = env.step(actions.tolist())
            controller.reconcile_request_queue(env)
            released = controller.release_due(env, env._steps)
            event_names = [event["type"] for event in info["events"]]
            if released:
                event_names.append("TASK_BATCH_RELEASED")
            if env._steps % 20 == 0 or event_names:
                decision = planner.plan(adapter.snapshot(env, event_names))
            if env._steps % frame_stride == 0 or event_names or terminated or truncated:
                frames.append(
                    draw_rollout_frame(
                        env,
                        decision,
                        seed,
                        event_names,
                        title="medium dynamic warehouse | rule validation",
                    )
                )
            if terminated or truncated:
                break
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=round(1000 / fps),
            loop=0,
            optimize=False,
        )
        print(
            "saved={path} frames={frames} steps={steps} completed={completed} batches={batches}".format(
                path=output,
                frames=len(frames),
                steps=env._steps,
                completed=env.completed_tasks,
                batches=len(controller.batches),
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    arguments = parse_args()
    export_episode(
        load_config(arguments.config),
        arguments.seed,
        arguments.frame_stride,
        arguments.fps,
        arguments.output,
    )
