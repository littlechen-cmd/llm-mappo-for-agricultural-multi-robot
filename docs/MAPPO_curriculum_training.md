# MAPPO Tiny-1AG Curriculum

## Scope

`configs/mappo_tiny_1ag_curriculum.yaml` is the first curriculum stage. It
uses one AGV, one active request, `picking_duration: 2`, and
`max_completed_tasks: 1`; one completed picking task terminates a successful
episode.

The training reward is `native_team_reward + shaped_task_reward`. Native
picking and death rewards are not changed. Dense path shaping is only granted
for a forward action that reduces legal BFS distance while the AGV stays in the
same collect, deliver, or charge phase; each transition is capped to one grid
edge. A small time cost, a one-time correct-load bonus, and a one-time
pick-start bonus provide task-completion credit without rewarding repeated
load/unload loops. The curriculum masks manual unload while an AGV is carrying
a shelf because the dock automatically completes the handoff.

## Rule Prior

`RuleBasedPriorPolicy` converts a `RulePlanner` assignment to a masked action
distribution. It handles route turning, shelf loading, charger use, and
waiting. Invalid actions always receive zero probability.

The behavior policy is a probability-space convex mixture:
`(1 - prior_mix) * actor_policy + prior_mix * rule_prior`. It is not a logit
bias, so a high prior coefficient cannot be overridden by an arbitrary Actor
logit. PPO evaluates its probability ratio against that same behavior policy.

The default curriculum has three stages: deterministic rule demonstrations and
strong behavioral cloning during `prior_warmup_episodes`; a gradual withdrawal
of behavior guidance over `prior_decay_episodes`; then Actor-only behavior
with persistent `KL(rule_prior || actor_policy)` regularization. The latter
keeps one bad stochastic rollout from erasing an already competent Actor.

## Training

Use a small smoke run locally; the default 2000-episode run is intended for
the CUDA PyTorch environment on the RTX 4080 Super server.

```powershell
D:\Anaconda3\envs\py310\python.exe train\train_mappo.py --config configs\mappo_tiny_1ag_curriculum.yaml --device cpu
```

```bash
python train/train_mappo.py --config configs/mappo_tiny_1ag_curriculum.yaml --device cuda
```

Episode logs separately report shaped/native/task-shaping returns, event counters,
battery mean, current prior coefficients, and Actor/Critic/entropy/KL losses.
At `actor_eval_interval`, the trainer also runs deterministic Actor-only
validation on fixed seeds. `checkpoint_path` always stores the best model by
Actor-only success rate and native return; `latest_checkpoint_path` stores the
last optimizer state for diagnosis.

## Evaluation and Visualization

Actor-only evaluation measures the learned low-level policy:

```powershell
python train\evaluate_mappo.py --config configs\mappo_tiny_1ag_curriculum.yaml --checkpoint artifacts\mappo_tiny_1ag_curriculum.pt --episodes 20 --device cpu
```

Rule-prior visualization opens the environment and validates the modeled task
flow independently of Actor convergence:

```powershell
python train\evaluate_mappo.py --config configs\mappo_tiny_1ag_curriculum.yaml --checkpoint artifacts\mappo_tiny_1ag_curriculum.pt --episodes 1 --render --delay 0.15 --rule-prior-only --device cpu
```

For a curriculum acceptance result, require at least 80% actor-only completion
over 20 fixed evaluation seeds and zero or near-zero deaths. Do not claim the
rule-prior visualization as Actor-only performance.
