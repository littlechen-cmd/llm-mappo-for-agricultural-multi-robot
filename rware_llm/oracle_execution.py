"""Deterministic FIFO dispatch and one-waypoint oracle control for Phase 2."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from rware_llm.interfaces import (
    ExecutionEvent,
    ExecutionEventType,
    PlannerDecision,
    PlannerSnapshot,
    TaskAssignment,
    TaskType,
)
from rware_llm.pathfinding import AStarRoutePlanner


@dataclass(frozen=True)
class OraclePlanDiagnostics:
    """Failure accounting separated from learned local execution behavior."""

    scheduling_failures: int = 0
    path_failures: int = 0
    replans: int = 0
    path_failure_reasons: Tuple[str, ...] = ()


class FifoOraclePlanner:
    """Own FIFO task dispatch and reveal only the next static-A* waypoint.

    The planner deliberately ignores live AGVs during A* construction. This is
    the Phase 2 oracle contract: task ownership and global route feasibility
    are known, while local collision avoidance remains the Actor's problem.
    """

    def __init__(self, env, charge_threshold: float = 0.1, plan_horizon: int = 30):
        self.env = env
        self.charge_threshold = charge_threshold
        self.plan_horizon = plan_horizon
        self._route_planner = AStarRoutePlanner()
        self._next_plan_index = 1
        self._assigned_shelves: Dict[int, int] = {}
        self._diagnostics = Counter()
        self._path_failure_reasons: List[str] = []

    @property
    def diagnostics(self) -> OraclePlanDiagnostics:
        return OraclePlanDiagnostics(
            scheduling_failures=self._diagnostics["scheduling_failures"],
            path_failures=self._diagnostics["path_failures"],
            replans=self._diagnostics["replans"],
            path_failure_reasons=tuple(self._path_failure_reasons),
        )

    def reset(self) -> None:
        self._assigned_shelves.clear()
        self._diagnostics.clear()
        self._path_failure_reasons.clear()
        self._next_plan_index = 1

    def plan(self, snapshot: PlannerSnapshot) -> PlannerDecision:
        self._synchronize_assignments(snapshot)
        requested = [shelf for shelf in snapshot.shelves if shelf.requested]
        requested_by_id = {shelf.id: shelf for shelf in requested}
        reserved = {
            shelf_id
            for shelf_id in self._assigned_shelves.values()
            if shelf_id in requested_by_id
        }
        reserved.update(
            agent.carrying_shelf_id
            for agent in snapshot.agents
            if agent.carrying_shelf_id is not None
        )
        assignments = {}
        for agent in sorted(snapshot.agents, key=lambda item: item.id):
            assignment = self._assignment_for_agent(agent, requested, requested_by_id, reserved)
            assignments[agent.id] = assignment
            if assignment.shelf_id is not None:
                reserved.add(assignment.shelf_id)
        self._diagnostics["replans"] += 1
        decision = PlannerDecision(
            plan_id=f"phase2-oracle-{self._next_plan_index:06d}",
            created_step=snapshot.step,
            valid_until_step=snapshot.step + self.plan_horizon,
            assignments=assignments,
            source="phase2-fifo-static-astar",
        )
        self._next_plan_index += 1
        return decision

    def _synchronize_assignments(self, snapshot: PlannerSnapshot) -> None:
        active_requested = {shelf.id for shelf in snapshot.shelves if shelf.requested}
        for agent_id, shelf_id in tuple(self._assigned_shelves.items()):
            agent = next(agent for agent in snapshot.agents if agent.id == agent_id)
            if (
                agent.dead
                or agent.carrying_shelf_id == shelf_id
                or shelf_id not in active_requested
            ):
                del self._assigned_shelves[agent_id]

    def _assignment_for_agent(self, agent, requested, requested_by_id, reserved):
        if agent.dead:
            return TaskAssignment(agent.id, TaskType.IDLE, None, 0.0)
        if agent.locked:
            return TaskAssignment(agent.id, TaskType.WAIT, agent.position, 1.0)
        if agent.carrying_shelf_id is not None:
            target = min(
                self.env.picking_docks,
                key=lambda dock: self._manhattan(agent.position, dock),
            )
            return self._route_assignment(
                agent.id,
                TaskType.DELIVER_TO_PICKER,
                target,
                priority=1.0,
                shelf_id=agent.carrying_shelf_id,
            )
        if agent.battery <= self.charge_threshold:
            target = min(
                self.env.charging_stations,
                key=lambda charger: self._manhattan(agent.position, charger),
            )
            return self._route_assignment(
                agent.id, TaskType.CHARGE, target, priority=0.95
            )
        assigned_id = self._assigned_shelves.get(agent.id)
        shelf = requested_by_id.get(assigned_id)
        if shelf is None:
            candidates = [item for item in requested if item.id not in reserved]
            if not candidates:
                return TaskAssignment(agent.id, TaskType.WAIT, agent.position, 0.0)
            # ``request_queue`` is the task ingress order. Snapshot shelf order
            # is physical order, so select the first requested shelf in env order.
            candidate_ids = {candidate.id for candidate in candidates}
            shelf = next(
                item for item in self.env.request_queue if item.id in candidate_ids
            )
            self._assigned_shelves[agent.id] = shelf.id
        return self._route_assignment(
            agent.id,
            TaskType.COLLECT_SHELF,
            self._shelf_position(shelf),
            priority=self._task_urgency(),
            shelf_id=shelf.id,
        )

    def _route_assignment(self, agv_id, task_type, target, priority, shelf_id=None):
        result = self._route_planner.route(
            self.env,
            agv_id,
            target,
            include_live_agvs=False,
            reason="phase2-oracle-static-astar",
        )
        if not result.reachable:
            self._diagnostics["path_failures"] += 1
            self._path_failure_reasons.append(result.unreachable_reason or "unknown")
            return TaskAssignment(agv_id, TaskType.WAIT, None, 0.0, shelf_id=shelf_id)
        waypoint = result.route.waypoints[0] if result.route.waypoints else target
        return TaskAssignment(agv_id, task_type, waypoint, priority, shelf_id=shelf_id)

    def record_scheduling_failure(self) -> None:
        self._diagnostics["scheduling_failures"] += 1

    def _task_urgency(self) -> float:
        return min(len(self.env.request_queue) / max(self.env.request_queue_size, 1), 1.0)

    @staticmethod
    def _manhattan(first: Tuple[int, int], second: Tuple[int, int]) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    @staticmethod
    def _shelf_position(shelf) -> Tuple[int, int]:
        """Accept either the immutable protocol snapshot or RWARE shelf entity."""

        if hasattr(shelf, "position"):
            return shelf.position
        return shelf.x, shelf.y


@dataclass
class ExecutionMonitor:
    """Emit stable Phase 2 execution exceptions without controlling actions."""

    stalled_steps: int = 30
    events: List[ExecutionEvent] = field(default_factory=list)
    _targets: Dict[int, Optional[Tuple[int, int]]] = field(default_factory=dict)
    _best_distances: Dict[int, Optional[int]] = field(default_factory=dict)
    _unchanged_steps: Dict[int, int] = field(default_factory=dict)
    _reported_stalls: set = field(default_factory=set)

    def reset(self) -> None:
        self.events.clear()
        self._targets.clear()
        self._best_distances.clear()
        self._unchanged_steps.clear()
        self._reported_stalls.clear()

    def observe(self, env, decision: PlannerDecision) -> Tuple[ExecutionEvent, ...]:
        emitted = []
        for agv in env.agents:
            assignment = decision.assignment_for(agv.id)
            target = assignment.target
            if agv.dead or target is None or assignment.task_type in {TaskType.IDLE, TaskType.WAIT}:
                self._reset_agent(agv.id)
                continue
            distance = env.shortest_path_distance(agv.id, target)
            if distance is None:
                emitted.append(
                    ExecutionEvent(
                        ExecutionEventType.REPLAN_REQUIRED,
                        env._steps,
                        agv.id,
                        "waypoint_unreachable",
                    )
                )
                self._reset_agent(agv.id)
                continue
            if self._targets.get(agv.id) != target:
                self._targets[agv.id] = target
                self._best_distances[agv.id] = distance
                self._unchanged_steps[agv.id] = 0
                self._reported_stalls.discard(agv.id)
                continue
            best_distance = self._best_distances.get(agv.id, distance)
            if distance < best_distance:
                self._best_distances[agv.id] = distance
                self._unchanged_steps[agv.id] = 0
                self._reported_stalls.discard(agv.id)
            elif distance > 0:
                self._unchanged_steps[agv.id] = self._unchanged_steps.get(agv.id, 0) + 1
            if (
                self._unchanged_steps.get(agv.id, 0) >= self.stalled_steps
                and agv.id not in self._reported_stalls
            ):
                emitted.append(
                    ExecutionEvent(
                        ExecutionEventType.STALLED,
                        env._steps,
                        agv.id,
                        "no_waypoint_progress",
                    )
                )
                self._reported_stalls.add(agv.id)
            if self._blocked_by_live_agv(env, agv.id, target):
                emitted.append(
                    ExecutionEvent(
                        ExecutionEventType.BLOCKED,
                        env._steps,
                        agv.id,
                        "no_live_agv_legal_route",
                    )
                )
        self.events.extend(emitted)
        return tuple(emitted)

    def _reset_agent(self, agv_id: int) -> None:
        self._targets.pop(agv_id, None)
        self._best_distances.pop(agv_id, None)
        self._unchanged_steps.pop(agv_id, None)
        self._reported_stalls.discard(agv_id)

    @staticmethod
    def _blocked_by_live_agv(env, agv_id: int, target: Tuple[int, int]) -> bool:
        if env.shortest_path_distance(agv_id, target) == 0:
            return False
        return env.shortest_path_distance(agv_id, target, include_live_agvs=True) is None


def count_execution_events(events: Iterable[ExecutionEvent]) -> Counter:
    """Return a string-keyed counter suitable for JSON and TensorBoard output."""

    return Counter(event.event_type.value for event in events)
