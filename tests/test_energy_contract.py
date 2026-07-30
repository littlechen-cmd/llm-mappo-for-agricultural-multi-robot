import pytest

from rware.heterogeneous import HeterogeneousAction, HeterogeneousWarehouse


def test_energy_contract_uses_total_battery_of_ten_and_normalized_observation():
    env = HeterogeneousWarehouse(size="tiny", n_agvs=1)
    observations, _ = env.reset(seed=1)

    assert env.agents[0].battery == 10.0
    assert observations[0][3] == 1.0

    env.agents[0].battery = 5.0
    assert env.get_observations()[0][3] == 0.5


def test_energy_contract_charges_each_action_by_confirmed_rate():
    env = HeterogeneousWarehouse(size="tiny", n_agvs=1)
    env.reset(seed=1)
    agv = env.agents[0]

    assert env._energy_drain(agv, HeterogeneousAction.FORWARD, True) == pytest.approx(
        0.001
    )
    assert env._energy_drain(agv, HeterogeneousAction.LEFT, True) == pytest.approx(
        0.001
    )
    assert env._energy_drain(agv, HeterogeneousAction.RIGHT, True) == pytest.approx(
        0.001
    )
    assert env._energy_drain(
        agv, HeterogeneousAction.TOGGLE_LOAD, True
    ) == pytest.approx(0.002)

    agv.carrying_shelf = env.request_queue[0]
    assert env._energy_drain(agv, HeterogeneousAction.FORWARD, True) == pytest.approx(
        0.002
    )
    assert env._energy_drain(agv, HeterogeneousAction.LEFT, True) == pytest.approx(
        0.002
    )
    assert env._energy_drain(agv, HeterogeneousAction.RIGHT, True) == pytest.approx(
        0.002
    )
    assert env._energy_drain(
        agv, HeterogeneousAction.TOGGLE_LOAD, True
    ) == pytest.approx(0.002)

    assert env._energy_drain(agv, HeterogeneousAction.NOOP, True) == pytest.approx(
        0.0002
    )
    assert env._energy_drain(agv, HeterogeneousAction.FORWARD, False) == pytest.approx(
        0.0002
    )
    agv.picking_remaining = 1
    assert env._energy_drain(agv, HeterogeneousAction.LEFT, True) == pytest.approx(
        0.0002
    )


def test_successful_charge_has_no_drain_and_respects_capacity():
    env = HeterogeneousWarehouse(size="tiny", n_agvs=1, charge_rate=0.2)
    env.reset(seed=1)
    agv = env.agents[0]
    agv.battery = 9.9

    _, _, _, _, info = env.step([HeterogeneousAction.CHARGE.value])

    assert agv.battery == 10.0
    assert [event["type"] for event in info["events"]] == ["CHARGED"]
