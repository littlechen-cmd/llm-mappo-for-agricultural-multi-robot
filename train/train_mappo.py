"""Train the parameter-shared CTDE MAPPO executor with a RulePlanner."""

from argparse import ArgumentParser
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml

# The unified repository vendors RWARE as source rather than a Git submodule.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RWARE_SOURCE = PROJECT_ROOT / "robotic-warehouse"
if str(RWARE_SOURCE) not in sys.path:
    sys.path.insert(0, str(RWARE_SOURCE))

from rware.heterogeneous import HeterogeneousWarehouse
from rware_llm.mappo import MAPPOConfig, MAPPOExecutor
from rware_llm.mappo.buffer import RolloutBuffer
from rware_llm.planner import RulePlanner
from rware_llm.state import WarehouseStateAdapter


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mappo_tiny_2ag.yaml")
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto")
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
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def should_replan(step, info, interval):
    event_types = {event["type"] for event in info.get("events", [])}
    disruptive = {"AGV_DEAD", "PICKING_COMPLETED"}
    return step % interval == 0 or bool(event_types & disruptive)


def train(config, device):
    set_seed(config["training"]["seed"])
    env_config = config["environment"]
    env = HeterogeneousWarehouse(
        size=env_config["size"],
        n_agvs=env_config["n_agvs"],
        request_queue_size=env_config["request_queue_size"],
        max_steps=env_config["max_steps"],
    )
    adapter = WarehouseStateAdapter(local_radius=config["model"]["local_radius"])
    planner = RulePlanner(**config["planner"])
    env.reset(seed=config["training"]["seed"])
    initial_plan = planner.plan(adapter.snapshot(env))
    initial_state = adapter.build(env, initial_plan)
    executor = MAPPOExecutor(
        vector_dim=initial_state.actor_vectors.shape[-1],
        local_channels=initial_state.local_grids.shape[1],
        global_channels=initial_state.global_map.shape[0],
        action_dim=env.action_space[0].n,
        config=MAPPOConfig(**config["mappo"]),
        device=device,
    )
    checkpoint_path = Path(config["training"]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for episode in range(1, config["training"]["episodes"] + 1):
        env.reset(seed=config["training"]["seed"] + episode)
        decision = planner.plan(adapter.snapshot(env))
        state = adapter.build(env, decision)
        rollout = RolloutBuffer()
        episode_reward = 0.0
        last_done = False

        while True:
            action_output = executor.act(state)
            _, rewards, terminated, truncated, info = env.step(action_output.actions.tolist())
            last_done = terminated or truncated
            team_reward = float(np.mean(rewards))
            episode_reward += team_reward
            rollout.add(
                state,
                action_output.actions,
                action_output.log_probs,
                action_output.value,
                team_reward,
                last_done,
            )
            if should_replan(env._steps, info, config["training"]["planner_interval"]):
                decision = planner.plan(adapter.snapshot(env, [event["type"] for event in info["events"]]))
            state = adapter.build(env, decision)
            if last_done:
                break

        metrics = executor.update(rollout, state.global_map, last_done)
        if episode % config["training"]["log_interval"] == 0:
            print(
                "episode={episode} return={return_value:.3f} steps={steps} "
                "actor_loss={actor_loss:.4f} critic_loss={critic_loss:.4f} entropy={entropy:.4f}".format(
                    episode=episode,
                    return_value=episode_reward,
                    steps=len(rollout),
                    **metrics,
                )
            )
        if episode % config["training"]["checkpoint_interval"] == 0:
            torch.save({"episode": episode, "executor": executor.state_dict()}, checkpoint_path)

    torch.save({"episode": config["training"]["episodes"], "executor": executor.state_dict()}, checkpoint_path)
    print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    arguments = parse_args()
    configuration = load_config(arguments.config)
    train(configuration, choose_device(arguments.device or configuration.get("device", "auto")))
