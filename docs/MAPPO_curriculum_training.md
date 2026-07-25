# MAPPO Tiny-1AG Curriculum

## Scope

`configs/mappo_tiny_1ag_curriculum.yaml` is the first curriculum stage. It
uses one AGV, one active request, `picking_duration: 2`, and
`max_completed_tasks: 1`; one completed picking task terminates a successful
episode.

The training reward is `native_team_reward + path_progress_reward`. The native
picking and death rewards are not changed. Path shaping is only
`scale * (previous_legal_distance - current_legal_distance)` when both the
high-level plan and target are unchanged. Legal distance is BFS distance that
respects map bounds, fixed pickers, dead AGVs, and the loaded-shelf constraint.
It does not use a straight-line or Manhattan-distance shortcut through blocked
cells.

## Rule Prior

`RuleBasedPriorPolicy` converts a `RulePlanner` assignment to a masked soft
action distribution. It handles route turning, shelf load/unload, charger use,
and waiting. Invalid actions always receive zero probability.

During the early curriculum, the MAPPO behavior policy is guided by this prior
and the training objective includes `KL(rule_prior || actor_policy)`. The
configuration decays both the behavior guidance strength and KL coefficient
over `prior_decay_episodes`. PPO evaluates its probability ratio against the
same mixed policy used to collect each rollout.

## Training

Use a small smoke run locally; the default 2000-episode run is intended for
the CUDA PyTorch environment on the RTX 4080 Super server.

```powershell
D:\Anaconda3\envs\py310\python.exe train\train_mappo.py --config configs\mappo_tiny_1ag_curriculum.yaml --device cpu
```

```bash
python train/train_mappo.py --config configs/mappo_tiny_1ag_curriculum.yaml --device cuda
```

Episode logs separately report shaped/native/path returns, event counters,
battery mean, current prior coefficients, and Actor/Critic/entropy/KL losses.

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
