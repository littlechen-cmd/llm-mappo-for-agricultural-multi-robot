"""Stable boundary between high-level planners and the MAPPO executor."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Protocol, Tuple


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
