"""Validated dynamic task lifecycle independent of any scheduler choice."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional

from rware_llm.interfaces import DispatchPlan, TaskBatch


class TaskLifecycleState(str, Enum):
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    LOCKED = "LOCKED"
    PICKED = "PICKED"
    DELIVERING = "DELIVERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DEFERRED = "DEFERRED"


@dataclass(frozen=True)
class TaskLifecycleRecord:
    task_id: int
    state: TaskLifecycleState
    owner_agv_id: Optional[int] = None
    updated_step: int = 0


class TaskLifecycleRegistry:
    """Own legal transitions from arrival to completion or failure."""

    def __init__(self):
        self._records: Dict[int, TaskLifecycleRecord] = {}

    def record(self, task_id: int) -> TaskLifecycleRecord:
        return self._records[task_id]

    def records(self) -> tuple[TaskLifecycleRecord, ...]:
        return tuple(self._records[task_id] for task_id in sorted(self._records))

    def add_batch(self, batch: TaskBatch) -> None:
        for task_id in batch.task_ids:
            if task_id in self._records:
                raise ValueError(f"task_id={task_id} arrived more than once")
            self._records[task_id] = TaskLifecycleRecord(
                task_id, TaskLifecycleState.PENDING, updated_step=batch.arrival_step
            )

    def apply_dispatch(self, plan: DispatchPlan) -> None:
        for task_id in plan.deferred_task_ids:
            record = self.record(task_id)
            self._transition(record, TaskLifecycleState.DEFERRED, plan.created_step)
        for assignment in plan.assignments:
            record = self.record(assignment.task_id)
            next_state = TaskLifecycleState.LOCKED if assignment.locked else TaskLifecycleState.ASSIGNED
            self._transition(record, next_state, plan.created_step, assignment.agv_id)

    def apply_environment_events(self, events: Iterable[dict], step: int) -> None:
        for event in events:
            event_type = event.get("type")
            task_id = event.get("shelf_id")
            agv_id = event.get("agv_id")
            if event_type == "AGV_DEAD":
                for record in self.records():
                    if record.owner_agv_id == agv_id and record.state not in {
                        TaskLifecycleState.COMPLETED,
                        TaskLifecycleState.FAILED,
                    }:
                        self._transition(record, TaskLifecycleState.FAILED, step)
            elif task_id is not None and task_id in self._records:
                record = self.record(task_id)
                if event_type == "SHELF_LOADED":
                    self._transition(record, TaskLifecycleState.PICKED, step, agv_id)
                elif event_type == "PICKING_STARTED":
                    self._transition(record, TaskLifecycleState.DELIVERING, step, agv_id)
                elif event_type == "PICKING_COMPLETED":
                    self._transition(record, TaskLifecycleState.COMPLETED, step, agv_id)

    def _transition(
        self,
        record: TaskLifecycleRecord,
        target: TaskLifecycleState,
        step: int,
        owner_agv_id: Optional[int] = None,
    ) -> None:
        legal = {
            TaskLifecycleState.PENDING: {
                TaskLifecycleState.ASSIGNED,
                TaskLifecycleState.LOCKED,
                TaskLifecycleState.DEFERRED,
            },
            TaskLifecycleState.DEFERRED: {
                TaskLifecycleState.ASSIGNED,
                TaskLifecycleState.DEFERRED,
            },
            TaskLifecycleState.ASSIGNED: {
                TaskLifecycleState.ASSIGNED,
                TaskLifecycleState.LOCKED,
                TaskLifecycleState.PICKED,
                TaskLifecycleState.FAILED,
            },
            TaskLifecycleState.LOCKED: {
                TaskLifecycleState.PICKED,
                TaskLifecycleState.FAILED,
            },
            TaskLifecycleState.PICKED: {
                TaskLifecycleState.DELIVERING,
                TaskLifecycleState.FAILED,
            },
            TaskLifecycleState.DELIVERING: {
                TaskLifecycleState.COMPLETED,
                TaskLifecycleState.FAILED,
            },
            TaskLifecycleState.COMPLETED: set(),
            TaskLifecycleState.FAILED: set(),
        }
        if target not in legal[record.state]:
            raise ValueError(
                f"cannot transition task_id={record.task_id} from {record.state} to {target}"
            )
        owner = owner_agv_id if owner_agv_id is not None else record.owner_agv_id
        if record.owner_agv_id is not None and owner is not None and owner != record.owner_agv_id:
            raise ValueError(f"task_id={record.task_id} ownership is locked")
        self._records[record.task_id] = TaskLifecycleRecord(
            record.task_id, target, owner, step
        )
