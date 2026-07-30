import numpy as np

from rware.heterogeneous import HeterogeneousWarehouse
from rware_llm.planner import RuleBasedPriorPolicy, RulePlanner
from rware_llm.state import WarehouseStateAdapter


def test_rule_prior_completes_tiny_two_agv_curriculum():
    env = HeterogeneousWarehouse(
        size="tiny",
        n_agvs=2,
        request_queue_size=2,
        max_steps=100,
        max_completed_tasks=4,
    )
    adapter = WarehouseStateAdapter()
    planner = RulePlanner(charge_threshold=2.0, reserve_margin=0.1, plan_horizon=20)
    prior = RuleBasedPriorPolicy(confidence=1.0)
    env.reset(seed=200042)
    decision = planner.plan(adapter.snapshot(env))

    while True:
        state = adapter.build(env, decision, prior)
        actions = np.argmax(state.prior_action_probs, axis=1)
        _, _, terminated, truncated, info = env.step(actions.tolist())
        if terminated or truncated:
            break
        if env._steps % 20 == 0 or info["events"]:
            decision = planner.plan(
                adapter.snapshot(env, [event["type"] for event in info["events"]])
            )

    assert env.completed_tasks == 4
