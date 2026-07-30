"""Stable boundary between high-level planners and the MAPPO executor."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Protocol, Tuple


Coordinate = Tuple[int, int]


class TaskType(str, Enum):
    IDLE = "IDLE"
    COLLECT_SHELF = "COLLECT_SHELF"
    DELIVER_TO_PICKER = "DELIVER_TO_PICKER"
    CHARGE = "CHARGE"
    WAIT = "WAIT"


@dataclass(frozen=True)
class AgentSnapshot:
    id: int
    position: Tuple[int, int]
    battery: float
    carrying_shelf_id: Optional[int]
    dead: bool
    locked: bool
    charger_energy_cost: Optional[float] = None
    collection_energy_costs: Tuple[Tuple[int, float], ...] = ()


@dataclass(frozen=True)
class ShelfSnapshot:
    id: int
    position: Tuple[int, int]
    requested: bool


@dataclass(frozen=True)
class PlannerSnapshot:
    step: int
    grid_size: Tuple[int, int]
    agents: Tuple[AgentSnapshot, ...]
    shelves: Tuple[ShelfSnapshot, ...]
    picking_docks: Tuple[Tuple[int, int], ...]
    charging_stations: Tuple[Tuple[int, int], ...]
    events: Tuple[str, ...] = ()


# Phase 1 uses the same immutable state payload for warehouse ingress and
# high-level planning. Keeping this alias preserves the established MAPPO API.
WarehouseSnapshot = PlannerSnapshot


@dataclass(frozen=True)
class TaskBatch:
    """A versioned, seed-reproducible set of task arrivals."""

    batch_id: str
    arrival_step: int
    task_ids: Tuple[int, ...]
    user_priority_override: Optional[float] = None
    policy_version: str = "task-batch-v1"

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise ValueError("batch_id must not be empty")
        if self.arrival_step < 0:
            raise ValueError("arrival_step must be non-negative")
        if not self.task_ids or any(task_id < 1 for task_id in self.task_ids):
            raise ValueError("task_ids must contain positive IDs")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be unique within a batch")

    def to_dict(self) -> dict:
        return {
            "schema_version": "v1",
            "batch_id": self.batch_id,
            "arrival_step": self.arrival_step,
            "task_ids": list(self.task_ids),
            "user_priority_override": self.user_priority_override,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "TaskBatch":
        _validate_keys(
            value,
            {
                "schema_version",
                "batch_id",
                "arrival_step",
                "task_ids",
                "user_priority_override",
                "policy_version",
            },
            "TaskBatch",
        )
        if value.get("schema_version", "v1") != "v1":
            raise ValueError("unsupported TaskBatch schema_version")
        return cls(
            batch_id=str(value["batch_id"]),
            arrival_step=int(value["arrival_step"]),
            task_ids=tuple(int(task_id) for task_id in value["task_ids"]),
            user_priority_override=value.get("user_priority_override"),
            policy_version=str(value.get("policy_version", "task-batch-v1")),
        )


@dataclass(frozen=True)
class DispatchAssignment:
    agv_id: int
    task_id: int
    picker_id: Optional[int] = None
    locked: bool = False


@dataclass(frozen=True)
class SpaceTimeReservation:
    agv_id: int
    location: Coordinate
    start_step: int
    end_step: int

    def __post_init__(self) -> None:
        if self.agv_id < 1 or self.start_step < 0 or self.end_step <= self.start_step:
            raise ValueError("invalid space-time reservation")


@dataclass(frozen=True)
class EdgeTimeReservation:
    agv_id: int
    start: Coordinate
    end: Coordinate
    start_step: int
    end_step: int

    def __post_init__(self) -> None:
        if self.agv_id < 1 or self.start_step < 0 or self.end_step <= self.start_step:
            raise ValueError("invalid edge-time reservation")


@dataclass(frozen=True)
class DispatchPlan:
    """Validated high-level task ownership and optional route reservations."""

    plan_id: str
    created_step: int
    assignments: Tuple[DispatchAssignment, ...] = ()
    deferred_task_ids: Tuple[int, ...] = ()
    path_reservations: Tuple[SpaceTimeReservation, ...] = ()
    edge_reservations: Tuple[EdgeTimeReservation, ...] = ()
    policy_version: str = "dispatch-v1"

    def to_dict(self) -> dict:
        return {
            "schema_version": "v1",
            "plan_id": self.plan_id,
            "created_step": self.created_step,
            "assignments": [assignment.__dict__ for assignment in self.assignments],
            "deferred_task_ids": list(self.deferred_task_ids),
            "path_reservations": [
                {
                    "agv_id": reservation.agv_id,
                    "location": list(reservation.location),
                    "start_step": reservation.start_step,
                    "end_step": reservation.end_step,
                }
                for reservation in self.path_reservations
            ],
            "edge_reservations": [
                {
                    "agv_id": reservation.agv_id,
                    "start": list(reservation.start),
                    "end": list(reservation.end),
                    "start_step": reservation.start_step,
                    "end_step": reservation.end_step,
                }
                for reservation in self.edge_reservations
            ],
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "DispatchPlan":
        _validate_keys(
            value,
            {
                "schema_version",
                "plan_id",
                "created_step",
                "assignments",
                "deferred_task_ids",
                "path_reservations",
                "edge_reservations",
                "policy_version",
            },
            "DispatchPlan",
        )
        if value.get("schema_version", "v1") != "v1":
            raise ValueError("unsupported DispatchPlan schema_version")
        return cls(
            plan_id=str(value["plan_id"]),
            created_step=int(value["created_step"]),
            assignments=tuple(
                DispatchAssignment(**assignment)
                for assignment in value.get("assignments", ())
            ),
            deferred_task_ids=tuple(
                int(task_id) for task_id in value.get("deferred_task_ids", ())
            ),
            path_reservations=tuple(
                SpaceTimeReservation(
                    agv_id=int(reservation["agv_id"]),
                    location=tuple(reservation["location"]),
                    start_step=int(reservation["start_step"]),
                    end_step=int(reservation["end_step"]),
                )
                for reservation in value.get("path_reservations", ())
            ),
            edge_reservations=tuple(
                EdgeTimeReservation(
                    agv_id=int(reservation["agv_id"]),
                    start=tuple(reservation["start"]),
                    end=tuple(reservation["end"]),
                    start_step=int(reservation["start_step"]),
                    end_step=int(reservation["end_step"]),
                )
                for reservation in value.get("edge_reservations", ())
            ),
            policy_version=str(value.get("policy_version", "dispatch-v1")),
        )


@dataclass(frozen=True)
class RoutePlan:
    agv_id: int
    target: Coordinate
    waypoints: Tuple[Coordinate, ...]
    eta_step: int
    reason: str = "dispatch"
    schema_version: str = "v1"

    def __post_init__(self) -> None:
        if self.agv_id < 1 or self.eta_step < 0:
            raise ValueError("RoutePlan requires a positive agv_id and non-negative ETA")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "agv_id": self.agv_id,
            "target": list(self.target),
            "waypoints": [list(point) for point in self.waypoints],
            "eta_step": self.eta_step,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "RoutePlan":
        _validate_keys(
            value,
            {"schema_version", "agv_id", "target", "waypoints", "eta_step", "reason"},
            "RoutePlan",
        )
        if value.get("schema_version", "v1") != "v1":
            raise ValueError("unsupported RoutePlan schema_version")
        return cls(
            agv_id=int(value["agv_id"]),
            target=tuple(value["target"]),
            waypoints=tuple(tuple(point) for point in value["waypoints"]),
            eta_step=int(value["eta_step"]),
            reason=str(value.get("reason", "dispatch")),
        )


class ExecutionEventType(str, Enum):
    STALLED = "STALLED"
    BLOCKED = "BLOCKED"
    REPLAN_REQUIRED = "REPLAN_REQUIRED"
    RESERVATION_CONFLICT = "RESERVATION_CONFLICT"
    AGV_DEAD = "AGV_DEAD"


@dataclass(frozen=True)
class ExecutionEvent:
    event_type: ExecutionEventType
    step: int
    agv_id: int
    reason: str = ""

    def __post_init__(self) -> None:
        if self.step < 0 or self.agv_id < 1:
            raise ValueError("ExecutionEvent step and agv_id must be non-negative/positive")


def _validate_keys(value: dict, allowed: Iterable[str], type_name: str) -> None:
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"{type_name} contains unknown fields: {sorted(unknown)}")


def _intervals_overlap(first_start: int, first_end: int, second_start: int, second_end: int) -> bool:
    return first_start < second_end and second_start < first_end


def validate_dispatch_plan(
    plan: DispatchPlan,
    snapshot: PlannerSnapshot,
    known_task_ids: Optional[Iterable[int]] = None,
) -> Tuple[str, ...]:
    """Return deterministic validation errors without partially applying a plan."""

    errors = []
    agent_ids = {agent.id for agent in snapshot.agents}
    task_ids = set(known_task_ids) if known_task_ids is not None else {
        shelf.id for shelf in snapshot.shelves
    }
    assigned_agents = set()
    assigned_tasks = set()
    for assignment in plan.assignments:
        if assignment.agv_id not in agent_ids:
            errors.append(f"unknown agv_id={assignment.agv_id}")
        if assignment.task_id not in task_ids:
            errors.append(f"unknown task_id={assignment.task_id}")
        if assignment.picker_id is not None and not 1 <= assignment.picker_id <= len(
            snapshot.picking_docks
        ):
            errors.append(f"unknown picker_id={assignment.picker_id}")
        if assignment.agv_id in assigned_agents:
            errors.append(f"duplicate agv_id={assignment.agv_id}")
        if assignment.task_id in assigned_tasks:
            errors.append(f"duplicate task_id={assignment.task_id}")
        assigned_agents.add(assignment.agv_id)
        assigned_tasks.add(assignment.task_id)

    for reservation in (*plan.path_reservations, *plan.edge_reservations):
        if reservation.agv_id not in agent_ids:
            errors.append(f"reservation has unknown agv_id={reservation.agv_id}")
    for index, reservation in enumerate(plan.path_reservations):
        for other in plan.path_reservations[index + 1 :]:
            if (
                reservation.agv_id != other.agv_id
                and reservation.location == other.location
                and _intervals_overlap(
                    reservation.start_step,
                    reservation.end_step,
                    other.start_step,
                    other.end_step,
                )
            ):
                errors.append(f"overlapping path reservation at {reservation.location}")
    for index, reservation in enumerate(plan.edge_reservations):
        for other in plan.edge_reservations[index + 1 :]:
            if (
                reservation.agv_id != other.agv_id
                and reservation.start == other.end
                and reservation.end == other.start
                and _intervals_overlap(
                    reservation.start_step,
                    reservation.end_step,
                    other.start_step,
                    other.end_step,
                )
            ):
                errors.append(
                    f"opposing edge reservation from {reservation.start} to {reservation.end}"
                )
    return tuple(errors)


@dataclass(frozen=True)
class TaskAssignment:
    agv_id: int
    task_type: TaskType
    target: Optional[Tuple[int, int]]
    priority: float
    shelf_id: Optional[int] = None


@dataclass(frozen=True)
class PlannerDecision:
    """Versioned, structured high-level plan consumed by MAPPO conditioning."""

    plan_id: str
    created_step: int
    valid_until_step: int
    assignments: Dict[int, TaskAssignment]
    source: str = "rule"

    def assignment_for(self, agv_id: int) -> TaskAssignment:
        return self.assignments.get(
            agv_id,
            TaskAssignment(agv_id, TaskType.IDLE, None, priority=0.0),
        )


class HighLevelPlanner(Protocol):
    """Protocol implemented by RulePlanner now and LLMPlanner later."""

    def plan(self, snapshot: PlannerSnapshot) -> PlannerDecision:
        """Return a validated task-level decision, never low-level actions."""
