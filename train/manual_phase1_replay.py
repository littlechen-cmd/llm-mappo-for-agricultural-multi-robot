"""Replay direct one-AGV actions for the Phase 1 task lifecycle demonstration."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RWARE_SOURCE = PROJECT_ROOT / "robotic-warehouse"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(RWARE_SOURCE) not in sys.path:
    sys.path.insert(0, str(RWARE_SOURCE))

from rware.heterogeneous import Direction, HeterogeneousAction, HeterogeneousWarehouse
from rware_llm.interfaces import PlannerDecision, TaskAssignment, TaskType
from rware_llm.pathfinding import AStarRoutePlanner
from train.visualize_tiny_2ag import draw_rollout_frame


ACTION_CODES = {
    "N": HeterogeneousAction.NOOP,
    "F": HeterogeneousAction.FORWARD,
    "L": HeterogeneousAction.LEFT,
    "R": HeterogeneousAction.RIGHT,
    "T": HeterogeneousAction.TOGGLE_LOAD,
    "C": HeterogeneousAction.CHARGE,
}


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--actions",
        default=None,
        help="Optional comma-separated direct actions: N,F,L,R,T,C.",
    )
    parser.add_argument("--output", default="artifacts/phase1_manual_1ag.gif")
    return parser.parse_args()


def _decision(env, target, task_type):
    agv = env.agents[0]
    return PlannerDecision(
        plan_id=f"manual-{env._steps}",
        created_step=env._steps,
        valid_until_step=env._steps + 1,
        assignments={
            agv.id: TaskAssignment(agv.id, task_type, target, priority=1.0)
        },
        source="manual",
    )


def _turn_toward(direction, desired):
    directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    clockwise = (directions.index(desired) - directions.index(direction)) % 4
    return HeterogeneousAction.RIGHT if clockwise == 1 else HeterogeneousAction.LEFT


def _route_actions(env, target):
    route = AStarRoutePlanner().route(env, 1, target)
    if not route.reachable:
        raise RuntimeError(f"manual route is unreachable: {route.unreachable_reason}")
    position = env.agents[0].x, env.agents[0].y
    direction = env.agents[0].dir
    actions = []
    for waypoint in route.route.waypoints:
        delta = waypoint[0] - position[0], waypoint[1] - position[1]
        desired = {
            (0, -1): Direction.UP,
            (1, 0): Direction.RIGHT,
            (0, 1): Direction.DOWN,
            (-1, 0): Direction.LEFT,
        }[delta]
        while direction != desired:
            turn = _turn_toward(direction, desired)
            actions.append(turn)
            directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
            offset = 1 if turn == HeterogeneousAction.RIGHT else -1
            direction = directions[(directions.index(direction) + offset) % 4]
        actions.append(HeterogeneousAction.FORWARD)
        position = waypoint
    return actions


def run_replay(seed, output_path, action_codes=None):
    env = HeterogeneousWarehouse(
        size="medium",
        n_agvs=1,
        n_pickers=2,
        n_chargers=1,
        request_queue_size=1,
        continuous_task_generation=False,
        randomize_initial_requests=True,
        max_steps=400,
    )
    env.reset(seed=seed)
    shelf = env.request_queue[0]
    frames = []
    trace = []

    def apply(action, target, task_type):
        decision = _decision(env, target, task_type)
        _, _, terminated, truncated, info = env.step([action.value])
        events = [event["type"] for event in info["events"]]
        frames.append(
            draw_rollout_frame(
                env,
                decision,
                seed,
                events,
                title="medium manual action replay | 1 AGV",
            )
        )
        trace.append({"step": env._steps, "action": action.name, "events": events})
        if terminated or truncated:
            raise RuntimeError("manual replay ended before its lifecycle completed")

    try:
        start = _decision(env, (shelf.x, shelf.y), TaskType.COLLECT_SHELF)
        frames.append(
            draw_rollout_frame(
                env,
                start,
                seed,
                [],
                title="medium manual action replay | 1 AGV",
            )
        )
        if action_codes is None:
            for action in _route_actions(env, (shelf.x, shelf.y)):
                apply(action, (shelf.x, shelf.y), TaskType.COLLECT_SHELF)
            apply(HeterogeneousAction.TOGGLE_LOAD, (shelf.x, shelf.y), TaskType.COLLECT_SHELF)
            dock = env.picking_docks[0]
            for action in _route_actions(env, dock):
                apply(action, dock, TaskType.DELIVER_TO_PICKER)
            for _ in range(env.picking_duration):
                apply(HeterogeneousAction.NOOP, dock, TaskType.DELIVER_TO_PICKER)
            charger = env.charging_stations[0]
            for action in _route_actions(env, charger):
                apply(action, charger, TaskType.CHARGE)
            apply(HeterogeneousAction.CHARGE, charger, TaskType.CHARGE)
        else:
            for action in action_codes:
                apply(action, None, TaskType.WAIT)
        event_types = {event for item in trace for event in item["events"]}
        if action_codes is None and not {
            "SHELF_LOADED",
            "PICKING_STARTED",
            "PICKING_COMPLETED",
            "CHARGED",
        } <= event_types:
            raise RuntimeError("manual replay did not complete the required lifecycle")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=160,
            loop=0,
            optimize=False,
        )
        trace_path = output.with_suffix(".json")
        trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
        print(f"saved={output} trace={trace_path} steps={env._steps}")
        return trace
    finally:
        env.close()


if __name__ == "__main__":
    arguments = parse_args()
    manual_actions = (
        tuple(ACTION_CODES[code.strip().upper()] for code in arguments.actions.split(","))
        if arguments.actions
        else None
    )
    run_replay(arguments.seed, arguments.output, manual_actions)
