"""Regression coverage for the Phase 2 FIFO/static-A* oracle contract."""

from pathlib import Path

import numpy as np
import yaml

from rware_llm.oracle_execution import FifoOraclePlanner
from rware_llm.phase2_runtime import Phase2EpisodeRuntime
from rware_llm.state import WarehouseStateAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with (PROJECT_ROOT / "configs" / "phase2_medium_1ag_oracle.yaml").open(
        encoding="utf-8"
    ) as handle:
        return yaml.safe_load(handle)


def test_fifo_oracle_reveals_only_one_static_astar_waypoint():
    config = load_config()
    runtime = Phase2EpisodeRuntime(config, use_rule_prior=False)
    try:
        runtime.reset(seed=2000)
        assignment = runtime.decision.assignment_for(1)
        assert assignment.target is not None
        assert assignment.target != runtime.env.request_queue[0].home_position
        assert abs(assignment.target[0] - runtime.env.agents[0].x) + abs(
            assignment.target[1] - runtime.env.agents[0].y
        ) == 1
        assert assignment.shelf_id == runtime.env.request_queue[0].id
        assert runtime.planner.diagnostics.path_failures == 0
    finally:
        runtime.close()


def test_phase2_state_is_nine_by_nine_with_fixed_neighbor_padding():
    config = load_config()
    runtime = Phase2EpisodeRuntime(config, use_rule_prior=False)
    try:
        state = runtime.reset(seed=2001)
        assert state.local_grids.shape == (1, 10, 9, 9)
        assert state.actor_vectors.shape == (1, 38)
        # There are no neighboring AGVs in this 1-AGV curriculum condition.
        np.testing.assert_array_equal(state.actor_vectors[0, -17:-2], 0.0)
    finally:
        runtime.close()


def test_stalled_event_is_recorded_after_configured_no_progress_window():
    config = load_config()
    config["diagnostics"]["stalled_steps"] = 1
    runtime = Phase2EpisodeRuntime(config, use_rule_prior=False)
    try:
        state = runtime.reset(seed=2002)
        runtime.step(np.zeros(1, dtype=np.int64))
        transition = runtime.step(np.zeros(1, dtype=np.int64))
        assert any(
            event.event_type.value == "STALLED"
            for event in transition.execution_events
        )
        assert state.action_masks.shape == (1, 6)
    finally:
        runtime.close()


def test_fifo_oracle_keeps_task_ownership_while_agv_is_en_route():
    config = load_config()
    runtime = Phase2EpisodeRuntime(config, use_rule_prior=False)
    try:
        runtime.reset(seed=2003)
        first_assignment = runtime.decision.assignment_for(1)
        snapshot = WarehouseStateAdapter().snapshot(runtime.env)
        next_assignment = runtime.planner.plan(snapshot).assignment_for(1)
        assert next_assignment.shelf_id == first_assignment.shelf_id
    finally:
        runtime.close()


def test_phase2_runtime_disables_rware_automatic_replenishment():
    config = load_config()
    runtime = Phase2EpisodeRuntime(config, use_rule_prior=False)
    try:
        runtime.reset(seed=2004)
        assert runtime.env.continuous_task_generation is False
    finally:
        runtime.close()
