"""Shared Phase 2 oracle episode runtime for MAPPO training and evaluation."""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from rware.heterogeneous import HeterogeneousWarehouse
from rware_llm.dynamic_tasks import PoissonArrivalConfig, PoissonTaskController
from rware_llm.oracle_execution import (
    ExecutionMonitor,
    FifoOraclePlanner,
    count_execution_events,
)
from rware_llm.planner import RuleBasedPriorPolicy
from rware_llm.state import MAPPOState, WarehouseStateAdapter


@dataclass(frozen=True)
class Phase2Transition:
    state: MAPPOState
    decision: object
    rewards: Tuple[float, ...]
    terminated: bool
    truncated: bool
    info: Dict[str, object]
    released_batches: Tuple[object, ...]
    execution_events: Tuple[object, ...]
    collision_blocks: int


def build_phase2_environment(env_config: dict, render_mode=None) -> HeterogeneousWarehouse:
    """Build a Phase 2 RWARE environment from the declarative YAML contract."""

    return HeterogeneousWarehouse(
        size=env_config["size"],
        n_agvs=env_config["n_agvs"],
        n_pickers=env_config["n_pickers"],
        n_chargers=env_config["n_chargers"],
        request_queue_size=env_config["request_queue_size"],
        picking_duration=env_config.get("picking_duration", 2),
        max_steps=env_config["max_steps"],
        max_completed_tasks=env_config["max_completed_tasks"],
        terminate_on_death=env_config.get("terminate_on_death", False),
        allow_manual_unload=env_config.get("allow_manual_unload", False),
        continuous_task_generation=env_config.get("continuous_task_generation", False),
        randomize_initial_requests=env_config.get("randomize_initial_requests", True),
        initial_battery=env_config.get("initial_battery", 10.0),
        max_battery=env_config.get("max_battery", 10.0),
        safe_charge=env_config.get("safe_charge", 5.0),
        render_mode=render_mode,
    )


class Phase2EpisodeRuntime:
    """Drive the fixed FIFO/static-A* oracle while MAPPO owns low-level actions."""

    def __init__(self, config: dict, use_rule_prior: bool):
        self.config = config
        self.env = build_phase2_environment(config["environment"])
        model_config = config["model"]
        self.adapter = WarehouseStateAdapter(
            local_radius=model_config["local_radius"],
            include_phase2_features=model_config.get("include_phase2_features", False),
        )
        planner_config = config["oracle"]
        self.planner = FifoOraclePlanner(
            self.env,
            charge_threshold=planner_config["charge_threshold"],
            plan_horizon=planner_config.get("plan_horizon", 30),
        )
        self.monitor = ExecutionMonitor(
            stalled_steps=config["diagnostics"].get("stalled_steps", 30)
        )
        dynamic_config = config["dynamic_tasks"]
        self.task_controller = PoissonTaskController(
            PoissonArrivalConfig(
                rate_per_step=dynamic_config["rate_per_step"],
                max_pending_requests=dynamic_config["max_pending_requests"],
                policy_version=dynamic_config.get("policy_version", "phase2-poisson-v1"),
            )
        )
        self.prior = (
            RuleBasedPriorPolicy(config["prior"]["confidence"])
            if use_rule_prior
            else None
        )
        self.decision = None
        self.state = None
        self.trace = []
        self._collision_blocks = 0

    def reset(self, seed: int, capture_trace: bool = False) -> MAPPOState:
        self.env.reset(seed=seed)
        released = self.task_controller.reset(self.env, seed)
        self.planner.reset()
        self.monitor.reset()
        self._collision_blocks = 0
        self.trace = []
        event_names = ["TASK_BATCH_RELEASED" for _ in released]
        self.decision = self.planner.plan(self.adapter.snapshot(self.env, event_names))
        self.state = self.adapter.build(self.env, self.decision, self.prior)
        if capture_trace:
            self._append_trace((), (), released, ())
        return self.state

    def step(self, actions, capture_trace: bool = False) -> Phase2Transition:
        if self.state is None or self.decision is None:
            raise RuntimeError("reset() must be called before step()")
        action_values = tuple(int(action) for action in actions)
        prior_positions = tuple((agv.x, agv.y) for agv in self.env.agents)
        prior_mask = self.state.action_masks.copy()
        _, rewards, terminated, truncated, info = self.env.step(action_values)
        collision_blocks = self._count_collision_blocks(
            action_values, prior_positions, prior_mask
        )
        self._collision_blocks += collision_blocks
        self.task_controller.reconcile_request_queue(self.env)
        released = self.task_controller.release_due(self.env, self.env._steps)
        if released:
            info["events"].extend(
                {
                    "type": "TASK_BATCH_RELEASED",
                    "batch_id": batch.batch_id,
                    "task_ids": list(batch.task_ids),
                }
                for batch in released
            )
        event_types = [event["type"] for event in info["events"]]
        self.decision = self.planner.plan(self.adapter.snapshot(self.env, event_types))
        execution_events = self.monitor.observe(self.env, self.decision)
        self.state = self.adapter.build(self.env, self.decision, self.prior)
        if capture_trace:
            self._append_trace(
                action_values, execution_events, released, info["events"]
            )
        return Phase2Transition(
            state=self.state,
            decision=self.decision,
            rewards=tuple(float(reward) for reward in rewards),
            terminated=terminated,
            truncated=truncated,
            info=info,
            released_batches=released,
            execution_events=execution_events,
            collision_blocks=collision_blocks,
        )

    @property
    def released_task_count(self) -> int:
        return sum(len(batch.task_ids) for batch in self.task_controller.batches)

    @property
    def collision_blocks(self) -> int:
        return self._collision_blocks

    def diagnostics(self) -> dict:
        planner = self.planner.diagnostics
        return {
            "released_task_count": self.released_task_count,
            "collision_blocks": self.collision_blocks,
            "oracle_scheduling_failures": planner.scheduling_failures,
            "oracle_path_failures": planner.path_failures,
            "oracle_path_failure_reasons": list(planner.path_failure_reasons),
            "oracle_replans": planner.replans,
            "execution_event_counts": dict(
                sorted(count_execution_events(self.monitor.events).items())
            ),
        }

    def close(self) -> None:
        self.env.close()

    def _count_collision_blocks(self, actions, prior_positions, prior_mask) -> int:
        """Count contested legal FORWARD moves that the environment rejects."""

        collisions = 0
        for index, action in enumerate(actions):
            if action != 1 or not prior_mask[index, 1]:
                continue
            agv = self.env.agents[index]
            if (agv.x, agv.y) == prior_positions[index]:
                collisions += 1
        return collisions

    def _append_trace(self, actions, execution_events, released, environment_events) -> None:
        self.trace.append(
            {
                "step": self.env._steps,
                "actions": list(actions),
                "waypoints": {
                    str(agv_id): (
                        list(assignment.target) if assignment.target is not None else None
                    )
                    for agv_id, assignment in self.decision.assignments.items()
                },
                "task_types": {
                    str(agv_id): assignment.task_type.value
                    for agv_id, assignment in self.decision.assignments.items()
                },
                "positions": [[agv.x, agv.y] for agv in self.env.agents],
                "battery": [agv.battery for agv in self.env.agents],
                "environment_events": list(environment_events),
                "released_batches": [batch.to_dict() for batch in released],
                "execution_events": [
                    {
                        "type": event.event_type.value,
                        "step": event.step,
                        "agv_id": event.agv_id,
                        "reason": event.reason,
                    }
                    for event in execution_events
                ],
            }
        )
