"""Export a headless Actor-only tiny-2AG MAPPO rollout as an animated GIF."""

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RWARE_SOURCE = PROJECT_ROOT / "robotic-warehouse"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(RWARE_SOURCE) not in sys.path:
    sys.path.insert(0, str(RWARE_SOURCE))

from rware_llm.mappo import MAPPOConfig, MAPPOExecutor
from rware_llm.planner import RuleBasedPriorPolicy, RulePlanner
from rware_llm.state import WarehouseStateAdapter
from train.evaluate_mappo import choose_device, load_config, make_env


COLORS = {
    "background": "#ffffff",
    "grid": "#202124",
    "shelf": "#483d8b",
    "requested": "#008080",
    "charger": "#1f77b4",
    "picker": "#9467bd",
    "dock": "#2ca02c",
    "agent": "#ff8c00",
    "loaded_agent": "#d62728",
    "dead_agent": "#505050",
    "target": "#e377c2",
    "text": "#202124",
}


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mappo_tiny_2ag.yaml")
    parser.add_argument(
        "--checkpoint", default="artifacts/mappo_tiny_2ag_recovery.pt"
    )
    parser.add_argument("--seed", type=int, default=200042)
    parser.add_argument("--output", default="artifacts/mappo_tiny_2ag_recovery.gif")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    return parser.parse_args()


def should_replan(step, info, interval):
    event_types = {event["type"] for event in info.get("events", [])}
    return step % interval == 0 or bool(
        event_types
        & {"AGV_DEAD", "PICKING_COMPLETED", "SHELF_LOADED", "SHELF_UNLOADED"}
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
    executor.set_prior_strength(0.0, 0.0)
    env.close()
    return executor, adapter, planner, prior


def draw_rollout_frame(env, decision, seed, events, title="tiny-2AG MAPPO | actor-only"):
    cell_size = 32
    header_height = 104
    rows, cols = env.grid_size
    image = Image.new(
        "RGB",
        (cols * cell_size + 1, header_height + rows * cell_size + 1),
        COLORS["background"],
    )
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), title, fill=COLORS["text"])
    draw.text(
        (8, 26),
        "seed={seed} step={step} completed={completed}/{target}".format(
            seed=seed,
            step=env._steps,
            completed=env.completed_tasks,
            target=env.max_completed_tasks,
        ),
        fill=COLORS["text"],
    )
    plan_text = " | ".join(
        "A{agent}:{task}@{target}".format(
            agent=agv.id,
            task=decision.assignment_for(agv.id).task_type.value,
            target=decision.assignment_for(agv.id).target,
        )
        for agv in env.agents
    )
    draw.text((8, 44), plan_text, fill=COLORS["text"])
    draw.text(
        (8, 62),
        "events=" + (", ".join(events) if events else "-"),
        fill=COLORS["text"],
    )
    draw.text(
        (8, 80),
        "purple=shelf teal=request blue=charger green=dock orange/red=AGV",
        fill=COLORS["text"],
    )

    requested_ids = {shelf.id for shelf in env.request_queue}
    for y in range(rows):
        for x in range(cols):
            left = x * cell_size
            top = header_height + y * cell_size
            draw.rectangle(
                (left, top, left + cell_size, top + cell_size),
                outline=COLORS["grid"],
            )

    def fill_cell(position, color, label):
        x, y = position
        left = x * cell_size + 3
        top = header_height + y * cell_size + 3
        draw.rectangle(
            (left, top, left + cell_size - 6, top + cell_size - 6), fill=color
        )
        if label:
            draw.text((left + 8, top + 6), label, fill="#ffffff")

    for shelf in env.shelfs:
        if shelf.active:
            fill_cell(
                (shelf.x, shelf.y),
                COLORS["requested"] if shelf.id in requested_ids else COLORS["shelf"],
                "S",
            )
    for position in env.charging_stations:
        fill_cell(position, COLORS["charger"], "C")
    for picker in env.picking_robots:
        fill_cell((picker.x, picker.y), COLORS["picker"], "P")
    for position in env.picking_docks:
        fill_cell(position, COLORS["dock"], "D")

    for agv in env.agents:
        assignment = decision.assignment_for(agv.id)
        if assignment.target is not None:
            x, y = assignment.target
            left = x * cell_size + 1
            top = header_height + y * cell_size + 1
            draw.rectangle(
                (left, top, left + cell_size - 2, top + cell_size - 2),
                outline=COLORS["target"],
                width=3,
            )

    directions = {"UP": (0, -8), "RIGHT": (8, 0), "DOWN": (0, 8), "LEFT": (-8, 0)}
    for agv in env.agents:
        center_x = agv.x * cell_size + cell_size // 2
        center_y = header_height + agv.y * cell_size + cell_size // 2
        color = (
            COLORS["dead_agent"]
            if agv.dead
            else COLORS["loaded_agent"]
            if agv.carrying_shelf is not None
            else COLORS["agent"]
        )
        draw.ellipse(
            (center_x - 10, center_y - 10, center_x + 10, center_y + 10),
            fill=color,
            outline=COLORS["grid"],
        )
        dx, dy = directions[agv.dir.name]
        draw.line((center_x, center_y, center_x + dx, center_y + dy), fill="#000000", width=3)
        draw.text((center_x - 5, center_y - 7), str(agv.id), fill="#ffffff")
    return image


def export_rollout(config, checkpoint_path, seed, output_path, fps, device):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if fps < 1:
        raise ValueError("fps must be at least one")
    executor, adapter, planner, prior = build_executor(config, checkpoint, device)
    env = make_env(config["environment"])
    try:
        env.reset(seed=seed)
        decision = planner.plan(adapter.snapshot(env))
        state = adapter.build(env, decision, prior)
        frames = [draw_rollout_frame(env, decision, seed, [])]
        while True:
            actions = executor.act(state, deterministic=True).actions
            _, _, terminated, truncated, info = env.step(actions.tolist())
            event_names = [event["type"] for event in info["events"]]
            if should_replan(env._steps, info, config["training"]["planner_interval"]):
                decision = planner.plan(adapter.snapshot(env, event_names))
            frames.append(draw_rollout_frame(env, decision, seed, event_names))
            if terminated or truncated:
                break
            state = adapter.build(env, decision, prior)
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
            "saved gif={path} frames={frames} completed={completed}/{target} "
            "steps={steps}".format(
                path=output,
                frames=len(frames),
                completed=env.completed_tasks,
                target=env.max_completed_tasks,
                steps=env._steps,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    arguments = parse_args()
    configuration = load_config(arguments.config)
    export_rollout(
        configuration,
        arguments.checkpoint,
        arguments.seed,
        arguments.output,
        arguments.fps,
        choose_device(arguments.device),
    )
