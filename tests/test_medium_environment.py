import pytest

from rware.heterogeneous import HeterogeneousWarehouse
from rware_llm.planner import RuleBasedPriorPolicy, RulePlanner
from rware_llm.state import WarehouseStateAdapter


@pytest.mark.parametrize("n_agvs,n_pickers", [(1, 2), (3, 2), (5, 4)])
def test_medium_layout_supports_requested_agv_picker_charger_matrix(
    n_agvs, n_pickers
):
    env = HeterogeneousWarehouse(
        size="medium",
        n_agvs=n_agvs,
        n_pickers=n_pickers,
        n_chargers=n_agvs,
        request_queue_size=n_agvs,
    )
    observations, _ = env.reset(seed=7)

    assert env.size == "medium"
    assert env.n_agents == n_agvs
    assert len(env.picking_robots) == n_pickers
    assert len(env.picking_docks) == n_pickers
    assert len(env.charging_stations) == n_agvs
    assert observations[0].shape == (12,)
    assert len({(agv.x, agv.y) for agv in env.agents}) == n_agvs
    assert not set(env.picking_docks) & set(env.charging_stations)

    adapter = WarehouseStateAdapter()
    decision = RulePlanner().plan(adapter.snapshot(env))
    state = adapter.build(env, decision, RuleBasedPriorPolicy())
    assert state.action_masks.shape == (n_agvs, env.action_space[0].n)


def test_generated_layout_rejects_charger_count_that_differs_from_agv_count():
    with pytest.raises(ValueError, match="n_chargers to equal n_agvs"):
        HeterogeneousWarehouse(
            size="medium", n_agvs=3, n_pickers=2, n_chargers=1
        )


def test_seeded_initial_requests_are_repeatable_and_change_across_seeds():
    env = HeterogeneousWarehouse(
        size="medium",
        n_agvs=3,
        n_pickers=2,
        n_chargers=3,
        request_queue_size=3,
        randomize_initial_requests=True,
    )

    env.reset(seed=101)
    first = tuple(shelf.id for shelf in env.request_queue)
    env.reset(seed=101)
    repeated = tuple(shelf.id for shelf in env.request_queue)
    env.reset(seed=202)
    changed = tuple(shelf.id for shelf in env.request_queue)

    assert first == repeated
    assert first != changed
