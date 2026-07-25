import numpy as np

from rware.heterogeneous import HeterogeneousAction, HeterogeneousWarehouse
from rware_llm.interfaces import PlannerDecision, TaskAssignment, TaskType
from rware_llm.planner import RuleBasedPriorPolicy, RulePlanner
from rware_llm.rewards import LegalPathRewardShaper
from rware_llm.state import WarehouseStateAdapter


def test_curriculum_ends_after_one_picking_completion():
    env = HeterogeneousWarehouse(size="tiny", n_agvs=1, max_completed_tasks=1)
    env.reset(seed=1)
    agv = env.agents[0]
    shelf = env.request_queue[0]
    agv.x, agv.y = env.picking_docks[0]
    agv.carrying_shelf = shelf
    shelf.x, shelf.y = agv.x, agv.y

    env.step([HeterogeneousAction.NOOP.value])
    env.step([HeterogeneousAction.NOOP.value])
    _, _, terminated, _, info = env.step([HeterogeneousAction.NOOP.value])

    assert terminated
    assert env.completed_tasks == 1
    assert any(event["type"] == "PICKING_COMPLETED" for event in info["events"])


def test_legal_path_shaper_rewards_one_step_of_progress():
    env = HeterogeneousWarehouse(size="tiny", n_agvs=1)
    env.reset(seed=2)
    planner = RulePlanner()
    decision = planner.plan(WarehouseStateAdapter().snapshot(env))
    target = decision.assignment_for(env.agents[0].id).target
    assert target is not None
    assert env.shortest_path_distance(env.agents[0].id, target) is not None
    next_position = env.shortest_path_next_position(env.agents[0].id, target)
    assert next_position is not None

    shaper = LegalPathRewardShaper(progress_scale=0.5)
    shaper.reset(env, decision)
    action = HeterogeneousAction.FORWARD.value
    while env._forward_location(env.agents[0]) != next_position:
        env.step([HeterogeneousAction.LEFT.value])
    env.step([action])

    assert shaper.reward(env, decision) > 0.0


def test_prior_is_masked_and_prefers_load_and_charge_actions():
    env = HeterogeneousWarehouse(size="tiny", n_agvs=1)
    env.reset(seed=3)
    prior = RuleBasedPriorPolicy()
    shelf = env.request_queue[0]
    agv = env.agents[0]
    agv.x, agv.y = shelf.x, shelf.y
    collect = PlannerDecision(
        "collect", 0, 20,
        {agv.id: TaskAssignment(agv.id, TaskType.COLLECT_SHELF, (shelf.x, shelf.y), 1.0, shelf.id)},
    )
    distribution = prior.action_distribution(env, collect)
    assert np.isclose(distribution.sum(), 1.0)
    assert distribution[0, HeterogeneousAction.TOGGLE_LOAD.value] == distribution[0].max()
    assert np.all(distribution[0, ~env.get_action_mask()[0]] == 0.0)

    agv.x, agv.y = env.charging_stations[0]
    charge = PlannerDecision(
        "charge", 0, 20,
        {
            agv.id: TaskAssignment(
                agv.id, TaskType.CHARGE, env.charging_stations[0], 1.0
            )
        },
    )
    distribution = prior.action_distribution(env, charge)
    assert distribution[0, HeterogeneousAction.CHARGE.value] == distribution[0].max()
