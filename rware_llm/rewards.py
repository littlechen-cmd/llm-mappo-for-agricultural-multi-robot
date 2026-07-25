"""Reward shaping utilities for warehouse MAPPO training."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from rware_llm.interfaces import PlannerDecision


@dataclass
class LegalPathRewardShaper:
    """Reward reductions in legal shortest-path distance within one plan.

    The environment's native picking and death rewards remain unchanged.  This
    helper adds only the difference between consecutive legal path distances,
    preventing a stationary AGV near a target from collecting dense reward.
    """

    progress_scale: float = 0.05
    _plan_id: Optional[str] = field(default=None, init=False)
    _distances: Dict[int, Optional[int]] = field(default_factory=dict, init=False)
    _targets: Dict[int, Optional[Tuple[int, int]]] = field(default_factory=dict, init=False)

    def reset(self, env, decision: PlannerDecision) -> None:
        self._plan_id = decision.plan_id
        self._distances = {}
        self._targets = {}
        self._prime(env, decision)

    def set_plan(self, env, decision: PlannerDecision) -> None:
        """Start a new baseline after any high-level-plan replacement."""

        self.reset(env, decision)

    def reward(self, env, decision: PlannerDecision) -> float:
        """Return team shaping reward after one environment transition."""

        if decision.plan_id != self._plan_id:
            self.reset(env, decision)
            return 0.0

        reward = 0.0
        for agv in env.agents:
            assignment = decision.assignment_for(agv.id)
            target = assignment.target
            previous_target = self._targets.get(agv.id)
            previous_distance = self._distances.get(agv.id)
            current_distance = self._distance(env, agv.id, target)
            if (
                target is not None
                and target == previous_target
                and previous_distance is not None
                and current_distance is not None
            ):
                reward += self.progress_scale * (previous_distance - current_distance)
            self._targets[agv.id] = target
            self._distances[agv.id] = current_distance
        # Match the trainer's native team reward, which averages AGV rewards.
        return float(reward / max(env.n_agents, 1))

    def _prime(self, env, decision: PlannerDecision) -> None:
        for agv in env.agents:
            target = decision.assignment_for(agv.id).target
            self._targets[agv.id] = target
            self._distances[agv.id] = self._distance(env, agv.id, target)

    @staticmethod
    def _distance(env, agv_id: int, target: Optional[Tuple[int, int]]) -> Optional[int]:
        if target is None:
            return None
        return env.shortest_path_distance(agv_id, target)
