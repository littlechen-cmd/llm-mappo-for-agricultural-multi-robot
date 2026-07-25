import pytest

from rware.heterogeneous import (
    HeterogeneousAction,
    HeterogeneousWarehouse,
    WAREHOUSE_SIZES,
    make_rware_style_layout,
)


@pytest.fixture
def env():
    environment = HeterogeneousWarehouse(max_steps=100)
    environment.reset(seed=3)
    return environment


def test_initial_state_and_spaces(env):
    observations, info = env.reset(seed=4)

    assert env.n_agents == 2
    assert len(env.action_space) == 2
    assert env.action_space[0].n == len(HeterogeneousAction)
    assert env.observation_space.contains(observations)
    assert all(agent.battery == pytest.approx(1.0) for agent in env.agents)
    assert env.picking_robots[0].x == 0
    assert env.picking_robots[0].y == env.grid_size[0] - 1
    assert info["active_requests"] == [1]


def test_charging_requires_a_charging_station(env):
    agv = env.agents[0]
    agv.x, agv.y = env.picking_docks[0]
    agv.battery = 0.5
    env.step([HeterogeneousAction.CHARGE.value, HeterogeneousAction.NOOP.value])
    assert agv.battery == pytest.approx(0.5 - env.standby_drain)

    agv.x, agv.y = env.charging_stations[0]
    agv.battery = 0.5
    env.step([HeterogeneousAction.CHARGE.value, HeterogeneousAction.NOOP.value])
    assert agv.battery == pytest.approx(0.5 + env.charge_rate)


def test_picking_locks_agv_for_exact_duration_and_completes_task(env):
    agv = env.agents[0]
    shelf = env.request_queue[0]
    agv.x, agv.y = env.picking_docks[0]
    agv.carrying_shelf = shelf
    shelf.x, shelf.y = agv.x, agv.y

    _, rewards, _, _, info = env.step(
        [HeterogeneousAction.NOOP.value, HeterogeneousAction.NOOP.value]
    )
    assert rewards[0] == pytest.approx(0.0)
    assert agv.picking_remaining == 2
    assert info["events"][-1]["type"] == "PICKING_STARTED"

    env.step([HeterogeneousAction.FORWARD.value, HeterogeneousAction.NOOP.value])
    assert agv.picking_remaining == 1
    assert (agv.x, agv.y) == env.picking_docks[0]

    _, rewards, _, _, info = env.step(
        [HeterogeneousAction.NOOP.value, HeterogeneousAction.NOOP.value]
    )
    assert agv.picking_remaining == 0
    assert agv.carrying_shelf is None
    assert shelf.active
    assert (shelf.x, shelf.y) == shelf.home_position
    assert shelf in env.request_queue
    assert rewards[0] == pytest.approx(env.pick_reward)
    assert any(event["type"] == "PICKING_COMPLETED" for event in info["events"])
    assert any(event["type"] == "REQUEST_GENERATED" for event in info["events"])
    assert len(env.request_queue) == env.request_queue_size


def test_death_is_one_time_penalty_and_dead_agv_blocks(env):
    agv = env.agents[0]
    agv.battery = env.standby_drain

    _, rewards, terminated, _, info = env.step(
        [HeterogeneousAction.NOOP.value, HeterogeneousAction.NOOP.value]
    )
    assert agv.dead
    assert rewards[0] == pytest.approx(env.death_penalty)
    assert not terminated
    assert info["dead_agvs"] == [agv.id]

    x, y = agv.x, agv.y
    _, rewards, _, _, _ = env.step(
        [HeterogeneousAction.FORWARD.value, HeterogeneousAction.NOOP.value]
    )
    assert (agv.x, agv.y) == (x, y)
    assert rewards[0] == pytest.approx(0.0)


def test_death_can_terminate_episode_when_configured():
    env = HeterogeneousWarehouse(terminate_on_death=True)
    env.reset(seed=1)
    env.agents[0].battery = env.standby_drain

    _, _, terminated, _, _ = env.step(
        [HeterogeneousAction.NOOP.value, HeterogeneousAction.NOOP.value]
    )
    assert terminated


@pytest.mark.parametrize("size", WAREHOUSE_SIZES)
def test_rware_style_size_layouts_reserve_service_area_and_rack_highways(size):
    env = HeterogeneousWarehouse(size=size, n_agvs=2)
    observations, _ = env.reset(seed=2)
    shelf_rows, shelf_columns = WAREHOUSE_SIZES[size]
    expected_rows = 9 * shelf_rows + 2
    expected_columns = 3 * shelf_columns + 1

    assert env.grid_size == (expected_rows, expected_columns)
    assert len(env.shelfs) > 0
    assert env.observation_space.contains(observations)
    assert [(picker.x, picker.y) for picker in env.picking_robots] == [
        (0, expected_rows - 1)
    ]
    assert env.picking_docks == [(1, expected_rows - 1)]
    expected_chargers = [
        (x, expected_rows - 1) for x in range(expected_columns - env.n_agents, expected_columns)
    ]
    assert env.charging_stations == expected_chargers
    assert [(agent.x, agent.y) for agent in env.agents] == expected_chargers

    # RWARE-style vertical and horizontal highway cells must never hold shelves.
    for shelf in env.shelfs:
        assert shelf.x % 3 != 0
        assert shelf.y % 9 != 0


def test_rware_style_layout_rejects_too_many_agvs_for_service_row():
    with pytest.raises(ValueError, match="supports at most"):
        make_rware_style_layout("tiny", n_agvs=9)


def test_continuous_task_generation_keeps_n_active_requests():
    env = HeterogeneousWarehouse(size="tiny", n_agvs=2, request_queue_size=2)
    env.reset(seed=5)
    completed_shelf = env.request_queue[0]
    agv = env.agents[0]
    agv.x, agv.y = env.picking_docks[0]
    agv.carrying_shelf = completed_shelf
    completed_shelf.x, completed_shelf.y = agv.x, agv.y

    env.step([HeterogeneousAction.NOOP.value, HeterogeneousAction.NOOP.value])
    env.step([HeterogeneousAction.NOOP.value, HeterogeneousAction.NOOP.value])
    _, _, _, _, info = env.step(
        [HeterogeneousAction.NOOP.value, HeterogeneousAction.NOOP.value]
    )

    assert len(env.request_queue) == 2
    assert completed_shelf in env.request_queue
    assert (completed_shelf.x, completed_shelf.y) == completed_shelf.home_position
    assert info["generated_requests"] == 1
    assert any(event["type"] == "REQUEST_GENERATED" for event in info["events"])
