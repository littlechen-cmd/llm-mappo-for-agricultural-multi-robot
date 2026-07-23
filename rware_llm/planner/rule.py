"""Deterministic planner used to validate the MAPPO planning interface."""

from typing import Dict, Set, Tuple

from rware_llm.interfaces import (
    HighLevelPlanner,
    PlannerDecision,
    PlannerSnapshot,
    TaskAssignment,
    TaskType,
)


def _distance(first: Tuple[int, int], second: Tuple[int, int]) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


class RulePlanner(HighLevelPlanner):
    """Battery-aware nearest-request planner with the future LLM interface."""

    def __init__(self, charge_threshold: float = 0.2, plan_horizon: int = 20):
        self.charge_threshold = charge_threshold
        self.plan_horizon = plan_horizon
        self._next_plan_index = 1

    def plan(self, snapshot: PlannerSnapshot) -> PlannerDecision:
        requested = [shelf for shelf in snapshot.shelves if shelf.requested]
        reserved_shelves: Set[int] = set()
        assignments: Dict[int, TaskAssignment] = {}

        for agent in sorted(snapshot.agents, key=lambda item: item.id):
            assignment = self._assignment_for_agent(
                agent,
                requested,
                reserved_shelves,
                snapshot.picking_docks,
                snapshot.charging_stations,
            )
            assignments[agent.id] = assignment
            if assignment.shelf_id is not None:
                reserved_shelves.add(assignment.shelf_id)

        decision = PlannerDecision(
            plan_id=f"rule-{self._next_plan_index:06d}",
            created_step=snapshot.step,
            valid_until_step=snapshot.step + self.plan_horizon,
            assignments=assignments,
            source="rule",
        )
        self._next_plan_index += 1
        return decision

    def _assignment_for_agent(
        self,
        agent,
        requested,
        reserved_shelves: Set[int],
        docks,
        chargers,
    ) -> TaskAssignment:
        if agent.dead:
            return TaskAssignment(agent.id, TaskType.IDLE, None, 0.0)
        if agent.locked:
            return TaskAssignment(agent.id, TaskType.WAIT, agent.position, 1.0)
        if agent.carrying_shelf_id is not None:
            target = min(docks, key=lambda dock: _distance(agent.position, dock))
            return TaskAssignment(agent.id, TaskType.DELIVER_TO_PICKER, target, 1.0)
        if agent.battery <= self.charge_threshold:
            target = min(chargers, key=lambda charger: _distance(agent.position, charger))
            return TaskAssignment(agent.id, TaskType.CHARGE, target, 0.9)

        available = [shelf for shelf in requested if shelf.id not in reserved_shelves]
        if available:
            shelf = min(available, key=lambda item: _distance(agent.position, item.position))
            return TaskAssignment(
                agent.id,
                TaskType.COLLECT_SHELF,
                shelf.position,
                priority=0.8,
                shelf_id=shelf.id,
            )
        return TaskAssignment(agent.id, TaskType.WAIT, agent.position, 0.1)
