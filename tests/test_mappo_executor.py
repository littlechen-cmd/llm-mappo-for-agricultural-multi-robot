import numpy as np

from rware.heterogeneous import HeterogeneousWarehouse
from rware_llm.mappo import MAPPOConfig, MAPPOExecutor
from rware_llm.mappo.buffer import RolloutBuffer
from rware_llm.planner import RuleBasedPriorPolicy, RulePlanner
from rware_llm.state import WarehouseStateAdapter


def _make_components():
    env = HeterogeneousWarehouse(size="tiny", n_agvs=2, max_steps=20)
    env.reset(seed=5)
    adapter = WarehouseStateAdapter(local_radius=2)
    planner = RulePlanner(plan_horizon=5)
    decision = planner.plan(adapter.snapshot(env))
    state = adapter.build(env, decision, RuleBasedPriorPolicy())
    executor = MAPPOExecutor(
        vector_dim=state.actor_vectors.shape[-1],
        local_channels=state.local_grids.shape[1],
        global_channels=state.global_map.shape[0],
        action_dim=env.action_space[0].n,
        config=MAPPOConfig(update_epochs=1, minibatch_size=4),
    )
    return env, adapter, planner, decision, state, executor


def test_adapter_exposes_local_cnn_global_cnn_and_safe_actions():
    env, _, _, _, state, _ = _make_components()

    assert state.actor_vectors.shape == (env.n_agents, 21)
    assert state.local_grids.shape == (env.n_agents, 10, 5, 5)
    assert state.global_map.shape == (10, *env.grid_size)
    assert state.action_masks.shape == (env.n_agents, env.action_space[0].n)
    assert state.prior_action_probs.shape == (env.n_agents, env.action_space[0].n)
    assert np.allclose(state.prior_action_probs.sum(axis=1), 1.0)
    assert np.all(state.prior_action_probs[~state.action_masks] == 0.0)
    assert np.all(state.action_masks[:, 0])


def test_shared_actor_act_and_ppo_update_complete_on_cpu():
    env, adapter, planner, decision, state, executor = _make_components()
    rollout = RolloutBuffer()

    for _ in range(4):
        output = executor.act(state)
        assert output.actions.shape == (env.n_agents,)
        assert np.all(state.action_masks[np.arange(env.n_agents), output.actions])
        _, rewards, terminated, truncated, info = env.step(output.actions.tolist())
        done = terminated or truncated
        rollout.add(
            state,
            output.actions,
            output.log_probs,
            output.value,
            float(np.mean(rewards)),
            done,
        )
        if done:
            break
        if env._steps % 2 == 0:
            decision = planner.plan(adapter.snapshot(env))
        state = adapter.build(env, decision, RuleBasedPriorPolicy())

    metrics = executor.update(rollout, state.global_map, done)
    assert set(metrics) == {"actor_loss", "critic_loss", "entropy", "prior_loss"}
    assert all(np.isfinite(value) for value in metrics.values())
