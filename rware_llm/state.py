"""State encoding from heterogeneous RWARE into MAPPO network inputs."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from rware_llm.interfaces import (
    AgentSnapshot,
    PlannerDecision,
    PlannerSnapshot,
    ShelfSnapshot,
    TaskAssignment,
    TaskType,
)


TASK_ORDER = tuple(TaskType)


@dataclass(frozen=True)
class MAPPOState:
    actor_vectors: np.ndarray
    local_grids: np.ndarray
    global_map: np.ndarray
    action_masks: np.ndarray
    prior_action_probs: np.ndarray


class WarehouseStateAdapter:
    """Build local CNN inputs, shared vectors, and a global CNN critic map."""

    local_channels = 10
    global_channels = 10

    def __init__(self, local_radius: int = 3):
        self.local_radius = local_radius
        self.local_size = 2 * local_radius + 1

    @property
    def condition_size(self) -> int:
        return len(TASK_ORDER) + 4

    def snapshot(self, env, events=()) -> PlannerSnapshot:
        requested_ids = {shelf.id for shelf in env.request_queue}
        agents = tuple(
            AgentSnapshot(
                id=agent.id,
                position=(agent.x, agent.y),
                battery=agent.battery,
                carrying_shelf_id=(
                    agent.carrying_shelf.id if agent.carrying_shelf is not None else None
                ),
                dead=agent.dead,
                locked=agent.locked,
            )
            for agent in env.agents
        )
        shelves = tuple(
            ShelfSnapshot(
                id=shelf.id,
                position=(shelf.x, shelf.y),
                requested=shelf.id in requested_ids,
            )
            for shelf in env.shelfs
            if shelf.active
        )
        return PlannerSnapshot(
            step=env._steps,
            grid_size=env.grid_size,
            agents=agents,
            shelves=shelves,
            picking_docks=tuple(env.picking_docks),
            charging_stations=tuple(env.charging_stations),
            events=tuple(events),
        )

    def build(self, env, decision: PlannerDecision, prior_policy=None) -> MAPPOState:
        observations = np.asarray(env.get_observations(), dtype=np.float32)
        assignments = [decision.assignment_for(agent.id) for agent in env.agents]
        conditions = np.asarray(
            [
                self._condition(agent, assignment, env.grid_size, decision, env._steps)
                for agent, assignment in zip(env.agents, assignments)
            ],
            dtype=np.float32,
        )
        actor_vectors = np.concatenate((observations, conditions), axis=1)
        base_map = self._base_global_map(env)
        global_map = self._global_map_with_plan(base_map, assignments, env.grid_size)
        local_grids = np.asarray(
            [self._local_grid(base_map, agent, assignment, env.grid_size) for agent, assignment in zip(env.agents, assignments)],
            dtype=np.float32,
        )
        action_masks = env.get_action_mask()
        prior_action_probs = (
            prior_policy.action_distribution(env, decision, action_masks)
            if prior_policy is not None
            else self._uniform_legal_action_probs(action_masks)
        )
        return MAPPOState(
            actor_vectors=actor_vectors,
            local_grids=local_grids,
            global_map=global_map,
            action_masks=action_masks,
            prior_action_probs=prior_action_probs,
        )

    @staticmethod
    def _uniform_legal_action_probs(action_masks: np.ndarray) -> np.ndarray:
        masks = np.asarray(action_masks, dtype=bool)
        legal_counts = masks.sum(axis=1, keepdims=True)
        if np.any(legal_counts == 0):
            raise ValueError("every AGV must have at least one legal action")
        return masks.astype(np.float32) / legal_counts.astype(np.float32)

    def _condition(
        self, agent, assignment: TaskAssignment, grid_size, decision, current_step
    ) -> np.ndarray:
        rows, cols = grid_size
        task_one_hot = np.zeros(len(TASK_ORDER), dtype=np.float32)
        task_one_hot[TASK_ORDER.index(assignment.task_type)] = 1.0
        if assignment.target is None:
            target_dx = target_dy = 0.0
        else:
            target_dx = (assignment.target[0] - agent.x) / max(cols - 1, 1)
            target_dy = (assignment.target[1] - agent.y) / max(rows - 1, 1)
        remaining = max(decision.valid_until_step - current_step, 0)
        return np.concatenate(
            (
                task_one_hot,
                np.asarray(
                    [
                        target_dx,
                        target_dy,
                        np.clip(assignment.priority, 0.0, 1.0),
                        min(remaining / 100.0, 1.0),
                    ],
                    dtype=np.float32,
                ),
            )
        )

    def _base_global_map(self, env) -> np.ndarray:
        rows, cols = env.grid_size
        feature_map = np.zeros((self.global_channels, rows, cols), dtype=np.float32)
        requested_ids = {shelf.id for shelf in env.request_queue}
        for shelf in env.shelfs:
            if not shelf.active:
                continue
            feature_map[0, shelf.y, shelf.x] = 1.0
            if shelf.id in requested_ids:
                feature_map[1, shelf.y, shelf.x] = 1.0
        for agent in env.agents:
            feature_map[2, agent.y, agent.x] = 1.0
            if agent.carrying_shelf is not None:
                feature_map[3, agent.y, agent.x] = 1.0
            if agent.dead:
                feature_map[4, agent.y, agent.x] = 1.0
        for picker in env.picking_robots:
            feature_map[5, picker.y, picker.x] = 1.0
        for x, y in env.picking_docks:
            feature_map[6, y, x] = 1.0
        for x, y in env.charging_stations:
            feature_map[7, y, x] = 1.0
        return feature_map

    def _global_map_with_plan(self, base_map, assignments, grid_size) -> np.ndarray:
        feature_map = base_map.copy()
        rows, cols = grid_size
        for assignment in assignments:
            if assignment.target is None:
                continue
            x, y = assignment.target
            if 0 <= x < cols and 0 <= y < rows:
                feature_map[8, y, x] = 1.0
                feature_map[9, y, x] = max(feature_map[9, y, x], assignment.priority)
        return feature_map

    def _local_grid(self, base_map, agent, assignment, grid_size) -> np.ndarray:
        rows, cols = grid_size
        grid = np.zeros((self.local_channels, self.local_size, self.local_size), dtype=np.float32)
        for local_y in range(self.local_size):
            for local_x in range(self.local_size):
                world_x = agent.x + local_x - self.local_radius
                world_y = agent.y + local_y - self.local_radius
                if not (0 <= world_x < cols and 0 <= world_y < rows):
                    grid[9, local_y, local_x] = 1.0
                    continue
                grid[:8, local_y, local_x] = base_map[:8, world_y, world_x]
                if assignment.target == (world_x, world_y):
                    grid[8, local_y, local_x] = 1.0
        grid[2, self.local_radius, self.local_radius] = 0.0
        return grid
