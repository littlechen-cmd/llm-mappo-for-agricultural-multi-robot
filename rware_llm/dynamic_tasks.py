"""Seeded dynamic task arrivals for the medium/small RWARE curriculum."""

from dataclasses import dataclass
from math import ceil
from random import Random
from typing import Optional, Tuple

from rware_llm.interfaces import TaskBatch


@dataclass(frozen=True)
class PoissonArrivalConfig:
    """Finite task ingress with exponentially distributed inter-arrival gaps."""

    rate_per_step: float = 1.0 / 20.0
    max_pending_requests: int = 3
    policy_version: str = "phase1-poisson-v1"

    def __post_init__(self) -> None:
        if self.rate_per_step <= 0.0:
            raise ValueError("rate_per_step must be positive")
        if self.max_pending_requests < 1:
            raise ValueError("max_pending_requests must be at least one")
        if not self.policy_version:
            raise ValueError("policy_version must not be empty")


class PoissonTaskController:
    """Own task release order without changing RWARE's physical shelf model.

    The underlying environment is configured with ``continuous_task_generation``
    disabled. RWARE may still append a replacement shelf after a completion;
    :meth:`reconcile_request_queue` removes that unplanned request so all visible
    work originates in an auditable :class:`TaskBatch`.
    """

    def __init__(self, config: Optional[PoissonArrivalConfig] = None):
        self.config = config or PoissonArrivalConfig()
        self._random = Random()
        self._batches: Tuple[TaskBatch, ...] = ()
        self._released_task_ids = set()
        self._next_arrival_step = 0
        self._batch_index = 0

    @property
    def batches(self) -> Tuple[TaskBatch, ...]:
        return self._batches

    def reset(self, env, seed: int) -> Tuple[TaskBatch, ...]:
        self._random = Random(seed)
        self._batches = ()
        self._released_task_ids = set()
        self._next_arrival_step = 0
        self._batch_index = 0
        env.request_queue.clear()
        return self.release_due(env, step=0)

    def release_due(self, env, step: int) -> Tuple[TaskBatch, ...]:
        if step < 0:
            raise ValueError("step must be non-negative")
        released = []
        while (
            self._next_arrival_step <= step
            and len(env.request_queue) < self.config.max_pending_requests
        ):
            task = self._next_task(env)
            if task is None:
                break
            self._batch_index += 1
            batch = TaskBatch(
                batch_id=f"poisson-{self._batch_index:04d}",
                arrival_step=self._next_arrival_step,
                task_ids=(task.id,),
                policy_version=self.config.policy_version,
            )
            env.request_queue.append(task)
            self._released_task_ids.add(task.id)
            self._batches += (batch,)
            released.append(batch)
            self._next_arrival_step += max(
                1, ceil(self._random.expovariate(self.config.rate_per_step))
            )
        return tuple(released)

    def reconcile_request_queue(self, env) -> None:
        """Remove automatic replenishment that was not released by this stream."""

        env.request_queue[:] = [
            shelf
            for shelf in env.request_queue
            if shelf.active and shelf.id in self._released_task_ids
        ]

    def _next_task(self, env):
        carrying_ids = {
            agv.carrying_shelf.id
            for agv in env.agents
            if agv.carrying_shelf is not None
        }
        candidates = [
            shelf
            for shelf in env.shelfs
            if shelf.active
            and shelf.id not in self._released_task_ids
            and shelf.id not in carrying_ids
            and shelf not in env.request_queue
        ]
        return self._random.choice(candidates) if candidates else None
