import pytest

from rware_llm.interfaces import (
    AgentSnapshot,
    DispatchAssignment,
    DispatchPlan,
    PlannerSnapshot,
    RoutePlan,
    SpaceTimeReservation,
    TaskBatch,
    validate_dispatch_plan,
)
from rware_llm.task_lifecycle import TaskLifecycleRegistry, TaskLifecycleState


def test_task_batch_and_route_plan_round_trip_and_reject_unknown_schema_fields():
    batch = TaskBatch("batch-001", 4, (3, 7), user_priority_override=1.0)
    assert TaskBatch.from_dict(batch.to_dict()) == batch
    route = RoutePlan(1, (4, 5), ((2, 3), (3, 3), (4, 3)), eta_step=9)
    assert RoutePlan.from_dict(route.to_dict()) == route
    with pytest.raises(ValueError, match="unknown fields"):
        TaskBatch.from_dict({"batch_id": "a", "arrival_step": 0, "task_ids": [1], "extra": 1})


def test_dispatch_plan_validation_rejects_unknown_and_conflicting_inputs():
    snapshot = PlannerSnapshot(
        step=0,
        grid_size=(5, 5),
        agents=(AgentSnapshot(1, (0, 0), 10.0, None, False, False),),
        shelves=(),
        picking_docks=((1, 1),),
        charging_stations=((4, 4),),
    )
    plan = DispatchPlan(
        "invalid",
        0,
        assignments=(DispatchAssignment(2, 9, picker_id=2),),
        path_reservations=(SpaceTimeReservation(2, (2, 2), 0, 2),),
    )

    assert validate_dispatch_plan(plan, snapshot, known_task_ids={1}) == (
        "unknown agv_id=2",
        "unknown task_id=9",
        "unknown picker_id=2",
        "reservation has unknown agv_id=2",
    )


def test_dispatch_plan_round_trip_and_task_lifecycle_enforce_ownership():
    plan = DispatchPlan("p", 2, assignments=(DispatchAssignment(1, 3, locked=True),))
    assert DispatchPlan.from_dict(plan.to_dict()) == plan

    lifecycle = TaskLifecycleRegistry()
    lifecycle.add_batch(TaskBatch("b", 0, (3,)))
    lifecycle.apply_dispatch(plan)
    lifecycle.apply_environment_events(
        [{"type": "SHELF_LOADED", "shelf_id": 3, "agv_id": 1}], step=3
    )
    lifecycle.apply_environment_events(
        [{"type": "PICKING_STARTED", "shelf_id": 3, "agv_id": 1}], step=4
    )
    lifecycle.apply_environment_events(
        [{"type": "PICKING_COMPLETED", "shelf_id": 3, "agv_id": 1}], step=5
    )
    assert lifecycle.record(3).state == TaskLifecycleState.COMPLETED
