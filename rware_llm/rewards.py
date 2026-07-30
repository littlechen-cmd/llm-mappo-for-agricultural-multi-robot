"""Reward shaping utilities for warehouse MAPPO training."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Set, Tuple

from rware_llm.interfaces import PlannerDecision, TaskType


@dataclass
class LegalPathRewardShaper:
    """Reward safe progress through a single high-level task lifecycle.

    Native picking and death rewards remain unchanged.  Dense progress is
    credited only for a forward action while an AGV remains in the same task
    phase, and it is capped to one grid edge.  One-time load and dock bonuses
    bridge the otherwise sparse reward between navigation and picking.
    """

    progress_scale: float = 0.05
    time_penalty: float = 0.01
    load_bonus: float = 0.25
    picking_start_bonus: float = 0.25
    unload_penalty: float = 0.25
    safe_charge: float = 5.0
    safe_charge_streak_steps: int = 0
    safe_charge_reward: float = 0.0
    low_battery_threshold: float = 0.0
    low_battery_penalty: float = 0.0
    collision_penalty: float = 0.0
    wait_streak_steps: int = 0
    wait_streak_penalty: float = 0.0
    _plan_id: Optional[str] = field(default=None, init=False)
    _distances: Dict[int, Optional[int]] = field(default_factory=dict, init=False)
    _targets: Dict[int, Optional[Tuple[int, int]]] = field(default_factory=dict, init=False)
    _task_types: Dict[int, TaskType] = field(default_factory=dict, init=False)
    _loaded_shelves: Set[int] = field(default_factory=set, init=False)
    _started_shelves: Set[int] = field(default_factory=set, init=False)
    _wait_streaks: Dict[int, int] = field(default_factory=dict, init=False)

    def reset(self, env, decision: PlannerDecision) -> None:
        self._plan_id = decision.plan_id
        self._distances = {}
        self._targets = {}
        self._task_types = {}
        self._loaded_shelves = set()
        self._started_shelves = set()
        self._wait_streaks = {}
        self._prime(env, decision)

    def set_plan(self, env, decision: PlannerDecision) -> None:
        """Start a new baseline after any high-level-plan replacement."""

        self._plan_id = decision.plan_id
        self._distances = {}
        self._targets = {}
        self._task_types = {}
        self._prime(env, decision)

    def reward(
        self,
        env,
        decision: PlannerDecision,
        actions: Optional[Iterable[int]] = None,
        events=(),
        collision_blocks: int = 0,
    ) -> float:
        """Return team shaping reward after one environment transition."""

        if decision.plan_id != self._plan_id:
            self.reset(env, decision)
            return 0.0

        reward = -self.time_penalty if any(not agv.dead for agv in env.agents) else 0.0
        taken_actions = tuple(actions) if actions is not None else ()
        for index, agv in enumerate(env.agents):
            assignment = decision.assignment_for(agv.id)
            target = assignment.target
            previous_target = self._targets.get(agv.id)
            previous_distance = self._distances.get(agv.id)
            current_distance = self._distance(env, agv.id, target)
            previous_task = self._task_types.get(agv.id)
            forward = index < len(taken_actions) and int(taken_actions[index]) == 1
            if (
                forward
                and target is not None
                and target == previous_target
                and assignment.task_type == previous_task
                and assignment.task_type
                in {TaskType.COLLECT_SHELF, TaskType.DELIVER_TO_PICKER, TaskType.CHARGE}
                and previous_distance is not None
                and current_distance is not None
            ):
                distance_change = max(-1, min(1, previous_distance - current_distance))
                reward += self.progress_scale * distance_change
            self._targets[agv.id] = target
            self._distances[agv.id] = current_distance
            self._task_types[agv.id] = assignment.task_type
            if (
                self.low_battery_threshold > 0.0
                and 0.0 < agv.battery < self.low_battery_threshold
                and assignment.task_type != TaskType.CHARGE
            ):
                reward -= self.low_battery_penalty
            is_waiting = (
                index < len(taken_actions)
                and int(taken_actions[index]) == 0
                and assignment.target is not None
                and assignment.task_type not in {TaskType.IDLE, TaskType.WAIT}
                and not agv.locked
            )
            if is_waiting:
                self._wait_streaks[agv.id] = self._wait_streaks.get(agv.id, 0) + 1
                if (
                    self.wait_streak_steps > 0
                    and self._wait_streaks[agv.id] % self.wait_streak_steps == 0
                ):
                    reward -= self.wait_streak_penalty
            else:
                self._wait_streaks[agv.id] = 0

        for event in events:
            event_type = event.get("type")
            agv_id = event.get("agv_id")
            assignment = decision.assignment_for(agv_id) if agv_id is not None else None
            shelf_id = event.get("shelf_id")
            if (
                event_type == "SHELF_LOADED"
                and shelf_id is not None
                and shelf_id not in self._loaded_shelves
                and assignment is not None
                and assignment.task_type == TaskType.COLLECT_SHELF
                and assignment.shelf_id == shelf_id
            ):
                self._loaded_shelves.add(shelf_id)
                reward += self.load_bonus
            elif (
                event_type == "PICKING_STARTED"
                and assignment is not None
                and assignment.task_type == TaskType.DELIVER_TO_PICKER
            ):
                carried_shelf = next(
                    (
                        agv.carrying_shelf.id
                        for agv in env.agents
                        if agv.id == agv_id and agv.carrying_shelf is not None
                    ),
                    None,
                )
                if carried_shelf is not None and carried_shelf not in self._started_shelves:
                    self._started_shelves.add(carried_shelf)
                    reward += self.picking_start_bonus
            elif event_type == "SHELF_UNLOADED":
                reward -= self.unload_penalty

        # Match the trainer's native team reward, which averages AGV rewards.
        reward -= self.collision_penalty * collision_blocks
        return float(reward / max(env.n_agents, 1))

    def _prime(self, env, decision: PlannerDecision) -> None:
        for agv in env.agents:
            assignment = decision.assignment_for(agv.id)
            target = assignment.target
            self._targets[agv.id] = target
            self._distances[agv.id] = self._distance(env, agv.id, target)
            self._task_types[agv.id] = assignment.task_type

    @staticmethod
    def _distance(env, agv_id: int, target: Optional[Tuple[int, int]]) -> Optional[int]:
        if target is None:
            return None
        return env.shortest_path_distance(agv_id, target)
