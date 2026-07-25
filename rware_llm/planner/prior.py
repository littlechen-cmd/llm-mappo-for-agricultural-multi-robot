"""Rule-derived action prior used to bootstrap MAPPO exploration."""

from typing import Optional, Tuple

import numpy as np

from rware.heterogeneous import HeterogeneousAction
from rware.warehouse import Direction
from rware_llm.interfaces import PlannerDecision, TaskType


class RuleBasedPriorPolicy:
    """Produce a masked, soft low-level action distribution from a plan."""

    def __init__(self, confidence: float = 0.9):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        self.confidence = confidence

    def action_distribution(self, env, decision: PlannerDecision, action_masks=None) -> np.ndarray:
        masks = env.get_action_mask() if action_masks is None else np.asarray(action_masks)
        action_dim = len(HeterogeneousAction)
        distributions = np.zeros((env.n_agents, action_dim), dtype=np.float32)
        for index, agv in enumerate(env.agents):
            legal = np.flatnonzero(masks[index])
            if not len(legal):
                raise RuntimeError("environment action mask contains no legal action")
            recommended = self._recommended_action(env, agv, decision)
            if recommended not in legal:
                recommended = int(HeterogeneousAction.NOOP.value)
            if recommended not in legal:
                recommended = int(legal[0])
            if len(legal) == 1:
                distributions[index, recommended] = 1.0
                continue
            remainder = (1.0 - self.confidence) / (len(legal) - 1)
            distributions[index, legal] = remainder
            distributions[index, recommended] = self.confidence
        return distributions

    def _recommended_action(self, env, agv, decision: PlannerDecision) -> int:
        assignment = decision.assignment_for(agv.id)
        if agv.dead or agv.locked or assignment.task_type in {TaskType.IDLE, TaskType.WAIT}:
            return HeterogeneousAction.NOOP.value
        if assignment.task_type == TaskType.COLLECT_SHELF:
            if assignment.target == (agv.x, agv.y):
                return HeterogeneousAction.TOGGLE_LOAD.value
            return self._navigate(env, agv, assignment.target)
        if assignment.task_type == TaskType.DELIVER_TO_PICKER:
            if assignment.target == (agv.x, agv.y):
                # The picker starts the timed task automatically at a dock.
                return HeterogeneousAction.NOOP.value
            return self._navigate(env, agv, assignment.target)
        if assignment.task_type == TaskType.CHARGE:
            if assignment.target == (agv.x, agv.y):
                return HeterogeneousAction.CHARGE.value
            return self._navigate(env, agv, assignment.target)
        return HeterogeneousAction.NOOP.value

    def _navigate(self, env, agv, target: Optional[Tuple[int, int]]) -> int:
        if target is None:
            return HeterogeneousAction.NOOP.value
        # Unlike reward shaping, action selection must avoid live AGVs that
        # would make its immediate FORWARD action illegal.
        next_position = env.shortest_path_next_position(
            agv.id, target, include_live_agvs=True
        )
        if next_position is None:
            return HeterogeneousAction.NOOP.value
        desired = self._direction_to((agv.x, agv.y), next_position)
        if desired == agv.dir:
            return HeterogeneousAction.FORWARD.value
        directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        current_index = directions.index(agv.dir)
        desired_index = directions.index(desired)
        clockwise = (desired_index - current_index) % len(directions)
        return (
            HeterogeneousAction.RIGHT.value
            if clockwise == 1
            else HeterogeneousAction.LEFT.value
        )

    @staticmethod
    def _direction_to(start: Tuple[int, int], target: Tuple[int, int]) -> Direction:
        delta = (target[0] - start[0], target[1] - start[1])
        return {
            (0, -1): Direction.UP,
            (1, 0): Direction.RIGHT,
            (0, 1): Direction.DOWN,
            (-1, 0): Direction.LEFT,
        }[delta]
