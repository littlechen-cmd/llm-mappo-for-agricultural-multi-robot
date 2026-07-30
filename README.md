# Dynamic Warehouse LLM-MAPPO

This repository implements a staged research and engineering baseline for a
dynamic robotic warehouse. The control boundary is intentionally explicit:

```text
task ingress / dispatch -> static A* route -> current waypoint -> MAPPO action
```

The current committed baseline is `4a84a3a` on `main`. It includes a recovered
Tiny-2AG baseline, the frozen medium dynamic environment, and the Phase 2
medium-oracle MAPPO training and evaluation pipeline.

## What Is Included

- Vendored RWARE-compatible warehouse environment in `robotic-warehouse/`.
- Battery contract with a physical range of `0.0` to `10.0` and normalized
  Actor observations.
- Dynamic Poisson task arrivals with reproducible seed pools.
- FIFO oracle dispatch, static A*, and a one-waypoint-only MAPPO interface.
- Phase 2 diagnostics for stalled, blocked, and replan-required execution.
- Shared-Actor CTDE MAPPO, PPO checkpointing, TensorBoard metrics, and
  Actor-only held-out evaluation.

Phase 2 does not require an LLM provider, API key, or network call.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `rware_llm/` | MAPPO, state encoding, rewards, task ingress, oracle, and A*. |
| `robotic-warehouse/` | Vendored RWARE source used by this project. |
| `configs/phase2_medium_1ag_oracle.yaml` | CUDA training configuration for the current Phase 2 gate. |
| `train/train_phase2_mappo.py` | Phase 2 training entry point. |
| `train/evaluate_phase2_mappo.py` | Strict held-out Actor-only evaluator. |
| `train/validate_phase2_oracle.py` | Rule-teacher oracle preflight validation. |
| `tests/` | Fast deterministic regression tests. |
| `artifacts/`, `runs/` | Ignored generated checkpoints, reports, and TensorBoard data. |

## 4080 Setup

Commands below assume Windows PowerShell and Python 3.10. Run them from the
repository root after cloning the repository.

```powershell
conda create -n mappo4080 python=3.10 -y
conda activate mappo4080
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch wheel before installing the rest of the
requirements. The command below targets the CUDA 12.4 wheel index. Use the
PyTorch wheel index matching the CUDA runtime supported by the machine when
necessary.

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e robotic-warehouse
```

Verify that the CUDA wheel can use the RTX 4080:

```powershell
python -c "import torch; print({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
```

`cuda_available` must be `True` before starting formal training. If it is
`False`, check the NVIDIA driver, select a matching PyTorch CUDA wheel, and
restart the shell. Do not run the 5000-episode Phase 2 experiment on CPU.

## Quick Validation

Run the regression suite that covers the committed Tiny recovery, Phase 1, and
Phase 2 oracle behavior:

```powershell
python -m pytest tests/test_phase1_protocols.py tests/test_pathfinding.py tests/test_dynamic_tasks.py tests/test_medium_environment.py tests/test_phase1_validation.py tests/test_curriculum_prior_reward.py tests/test_mappo_executor.py tests/test_energy_contract.py tests/test_tiny_2ag_rule_prior.py tests/test_phase2_oracle.py -q
```

Run the deterministic medium-oracle preflight. It uses the rule teacher only;
it validates task ingress and oracle feasibility and is not a MAPPO result.

```powershell
python train/validate_phase2_oracle.py --episodes 20 --output artifacts/phase2_medium_1ag_oracle_baseline.json
```

Expected preflight baseline on held-out seeds `2000-2019`:

```text
full success rate:             95.00%
fixed target completion rate:  99.17%
deaths:                        0
A* path failures:              0
```

The Phase 2 1-AGV environment uses `medium`, `2` pickers, `1` charger, a
fixed target of `6` completed tasks, and a `400`-step cap. The prior 12-task
target was rejected by this oracle preflight as physically infeasible for one
AGV within 400 steps. The evaluator reports both fixed-target completion and
completion of dynamically released work; only fixed-target completion is the
hard gate.

## Phase 2 CUDA Training

Start a fresh formal training run:

```powershell
python train/train_phase2_mappo.py --config configs/phase2_medium_1ag_oracle.yaml --device cuda
```

The configuration uses a reproducible random training pool of seeds
`1000-1099`, held-out seeds `2000-2019`, a maximum of 5000 episodes, and a
rule-prior curriculum that decays to zero. Training outputs are intentionally
ignored by Git:

```text
artifacts/mappo_phase2_medium_1ag_oracle.pt
artifacts/mappo_phase2_medium_1ag_oracle_latest.pt
artifacts/mappo_phase2_medium_1ag_training.json
runs/mappo_phase2_medium_1ag_oracle/
```

To resume a run on the same machine, use the latest full checkpoint. It
restores the Actor, Critic, optimizer, and deterministic training-seed prefix.

```powershell
python train/train_phase2_mappo.py --config configs/phase2_medium_1ag_oracle.yaml --device cuda --resume-checkpoint artifacts/mappo_phase2_medium_1ag_oracle_latest.pt
```

Checkpoints are not present after a fresh Git clone. Copy them separately only
when continuing an existing run; otherwise start the formal run from scratch.

## Actor-Only Held-Out Evaluation

After training, run the 20-seed Actor-only evaluation. It forcibly disables
rule action mixing, so the result measures the learned Actor rather than the
teacher policy.

```powershell
python train/evaluate_phase2_mappo.py --config configs/phase2_medium_1ag_oracle.yaml --checkpoint artifacts/mappo_phase2_medium_1ag_oracle.pt --episodes 20 --seed-start 2000 --output artifacts/phase2_medium_1ag_heldout.json --device cuda
```

The Phase 2 gate requires all of the following before proceeding to 3 AGVs or
Phase 3:

| Gate | Requirement |
| --- | --- |
| Fixed-target task completion | At least 95%. |
| Collision blocks | At most 2 per episode. |
| Deadlock episodes | At most 5%. |
| Training budget | Stable by 5000 episodes or fewer. |
| Seed stability | Success standard deviation at most 10%. |
| Policy isolation | Actor-only evaluation with rule mixing equal to zero. |

The JSON report includes per-seed task, battery, event, oracle-failure, and
execution diagnostics. Failed episodes include a trace of actions, waypoints,
task releases, and execution events.

## Git Workflow

Sync an existing 4080 checkout without overwriting its local work:

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
git rev-parse HEAD
```

Generated `artifacts/`, `runs/`, model checkpoints, and environment files are
ignored. Do not add API keys, `.env` files, or generated training outputs to a
commit.

The repository can contain local candidate scheduling and traffic experiments
that are not part of the committed Phase 2 control path. Do not use them for
the Phase 2 acceptance result unless they are separately reviewed and
committed.
