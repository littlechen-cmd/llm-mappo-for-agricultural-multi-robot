"""Render a deterministic, Actor-only MAPPO demonstration for tiny-2ag."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import sys
from time import sleep

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RWARE_SOURCE = PROJECT_ROOT / "robotic-warehouse"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(RWARE_SOURCE) not in sys.path:
    sys.path.insert(0, str(RWARE_SOURCE))

from rware.heterogeneous import HeterogeneousAction
from rware_llm.mappo import MAPPOConfig, MAPPOExecutor
from rware_llm.planner import RuleBasedPriorPolicy, RulePlanner
from rware_llm.state import WarehouseStateAdapter
from train.evaluate_mappo import choose_device, load_config, make_env


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mappo_tiny_2ag_stage1.yaml")
    parser.add_argument(
        "--checkpoint", default="artifacts/mappo_tiny_2ag_stage1.pt"
    )
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep the final rendered state visible until the window is closed.",
    )
    return parser.parse_args()


def should_replan(step, info, interval):
    event_types = {event["type"] for event in info.get("events", [])}
    return step % interval == 0 or bool(
        event_types & {"AGV_DEAD", "PICKING_COMPLETED", "SHELF_LOADED", "SHELF_UNLOADED"}
    )


def format_plan(env, decision):
    return " | ".join(
        "A{agv_id}:{task}@{target}".format(
            agv_id=agv.id,
            task=decision.assignment_for(agv.id).task_type.value,
            target=decision.assignment_for(agv.id).target,
        )
        for agv in env.agents
    )


def format_agents(env):
    return " | ".join(
        "A{id}=({x},{y}) dir={direction} battery={battery:.2f} load={loaded} lock={locked}".format(
            id=agv.id,
            x=agv.x,
            y=agv.y,
            direction=agv.dir.name,
            battery=agv.battery,
            loaded=int(agv.carrying_shelf is not None),
            locked=int(agv.locked),
        )
        for agv in env.agents
    )


def build_executor(config, checkpoint_path, device):
    env = make_env(config["environment"])
    adapter = WarehouseStateAdapter(local_radius=config["model"]["local_radius"])
    planner = RulePlanner(**config["planner"])
    prior = RuleBasedPriorPolicy(config.get("prior", {}).get("confidence", 1.0))
    env.reset(seed=config["training"]["seed"])
    state = adapter.build(env, planner.plan(adapter.snapshot(env)), prior)
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
    # The demonstration evaluates the learned policy, not its rule teacher.
    executor.set_prior_strength(0.0, 0.0)
    env.close()
    return executor, adapter, planner, prior


def run_demo(config, checkpoint_path, device, seed, delay, keep_open):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    executor, adapter, planner, prior = build_executor(config, checkpoint, device)
    env = make_env(config["environment"], render_mode="human")
    episode_seed = seed if seed is not None else config["training"]["seed"] + 10_000
    env.reset(seed=episode_seed)
    decision = planner.plan(adapter.snapshot(env))
    state = adapter.build(env, decision, prior)
    event_counts = Counter()
    native_return = 0.0

    print(
        "demo checkpoint={checkpoint} seed={seed} mode=actor-only".format(
            checkpoint=checkpoint, seed=episode_seed
        )
    )
    print(f"initial plan: {format_plan(env, decision)}")
    if not env.render():
        return
    sleep(delay)

    while True:
        output = executor.act(state, deterministic=True)
        action_names = [HeterogeneousAction(action).name for action in output.actions]
        _, rewards, terminated, truncated, info = env.step(output.actions.tolist())
        native_return += float(np.mean(rewards))
        event_names = [event["type"] for event in info["events"]]
        event_counts.update(event_names)
        print(
            "step={step:03d} plan=[{plan}] actions={actions} events={events}\n  {agents}".format(
                step=info["steps"],
                plan=format_plan(env, decision),
                actions=action_names,
                events=event_names or ["-"],
                agents=format_agents(env),
            )
        )
        if not env.render():
            return
        sleep(delay)
        if terminated or truncated:
            break
        if should_replan(env._steps, info, config["training"]["planner_interval"]):
            decision = planner.plan(
                adapter.snapshot(env, [event["type"] for event in info["events"]])
            )
            print(f"  replanned: {format_plan(env, decision)}")
        state = adapter.build(env, decision, prior)

    success = event_counts["PICKING_COMPLETED"] > 0
    print(
        "finished success={success} steps={steps} native_return={native_return:.3f} "
        "deaths={deaths}".format(
            success=int(success),
            steps=env._steps,
            native_return=native_return,
            deaths=event_counts["AGV_DEAD"],
        )
    )
    if keep_open:
        while env.render():
            sleep(0.05)
    env.close()


if __name__ == "__main__":
    arguments = parse_args()
    configuration = load_config(arguments.config)
    run_demo(
        configuration,
        arguments.checkpoint,
        choose_device(arguments.device),
        arguments.seed,
        arguments.delay,
        arguments.keep_open,
    )
