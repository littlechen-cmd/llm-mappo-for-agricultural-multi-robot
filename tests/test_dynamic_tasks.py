from rware.heterogeneous import HeterogeneousWarehouse
from rware_llm.dynamic_tasks import PoissonArrivalConfig, PoissonTaskController


def _environment():
    env = HeterogeneousWarehouse(
        size="medium",
        n_agvs=3,
        n_pickers=2,
        n_chargers=3,
        request_queue_size=1,
        continuous_task_generation=False,
    )
    env.reset(seed=1)
    return env


def test_poisson_task_arrivals_are_seeded_and_release_only_unseen_shelves():
    config = PoissonArrivalConfig(rate_per_step=0.2, max_pending_requests=20)
    first_env = _environment()
    second_env = _environment()
    try:
        first = PoissonTaskController(config)
        second = PoissonTaskController(config)
        first.reset(first_env, seed=77)
        second.reset(second_env, seed=77)
        first.release_due(first_env, step=100)
        second.release_due(second_env, step=100)

        assert first.batches == second.batches
        assert first.batches[0].arrival_step == 0
        assert all(
            later.arrival_step > earlier.arrival_step
            for earlier, later in zip(first.batches, first.batches[1:])
        )
        task_ids = [task_id for batch in first.batches for task_id in batch.task_ids]
        assert len(task_ids) == len(set(task_ids))
        assert {shelf.id for shelf in first_env.request_queue} == set(task_ids)
    finally:
        first_env.close()
        second_env.close()


def test_reconcile_removes_unplanned_rware_replenishment():
    env = _environment()
    try:
        controller = PoissonTaskController()
        released = controller.reset(env, seed=9)
        planned = env.request_queue[0]
        unplanned = next(shelf for shelf in env.shelfs if shelf is not planned)
        env.request_queue.append(unplanned)

        controller.reconcile_request_queue(env)

        assert [shelf.id for shelf in env.request_queue] == list(released[0].task_ids)
    finally:
        env.close()
