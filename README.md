# Shared-Control Surgical Robotics in Isaac Lab

This repository provides an Isaac Lab research environment for learning shared control between a human-force agent and a robot-force agent. The task combines trajectory following, obstacle interaction, force-aware collaboration, and safety-constrained control in a vectorized GPU simulation.

Three multi-agent reinforcement-learning methods are implemented on a common physical task:

- **rMAPPO**: recurrent, on-policy multi-agent PPO with centralized critics.
- **MADDPG**: off-policy deterministic actors with centralized critics and replay.
- **EPIGRAPH**: a recurrent, `z`-conditioned safe-MARL method with separate performance and constraint value branches.

All three methods support the same human-force models, force limits, observation scaling, checkpoint metadata, experiment seeds, and Weights & Biases integration.

## Key features

- Vectorized Isaac Lab environments for GPU training.
- Separate human and robot policies with centralized training.
- Three interchangeable human-force models.
- Previous *applied* opponent force included in each agent's observation.
- Per-axis physical-force limits and explicit normalized/physical action conversion.
- Deterministic seed plans with independent random streams.
- Resumable checkpoints containing resolved configuration and runtime metadata.
- Milestone evaluation and compact algorithm-specific W&B logging.
- Unit tests for force composition, replay, recurrent PPO bookkeeping, reproducibility, and EPIGRAPH dynamic programming.

## Task and control interface

The agents jointly control a surgical stylus using Cartesian forces:

| Item | Description |
|---|---|
| Agents | `human`, `robot` |
| Agent action | 3D Cartesian force |
| Agent observation | Task state plus the previous applied force of the other agent |
| Control rate | 60 Hz (`120 Hz` simulation with decimation `2`) |
| Default episode duration | 20 seconds |
| Standard MARL task | `Isaac-Surgical-MARL-Direct-v0` |
| EPIGRAPH task | `Isaac-Surgical-MARL-Epigraph-v0` |

The reward uses a four-zone task representation together with progress, deviation, completion, force-efficiency, and collaborator-awareness terms. EPIGRAPH exposes the same task information as a performance reward and a signed safety constraint for its two value branches.

## Human-force models

Select a mode with `--human_model_type`:

| Mode | Human command | Trainable policies |
|---|---|---|
| `learnable` | Human actor directly produces the human force | Human and robot |
| `fixed_impedance` | Analytic impedance controller only | Robot only |
| `residual_impedance` | Analytic impedance force plus a learned residual | Human residual actor and robot |

Default force semantics are per axis:

- Robot force: bounded by `0.04 N`.
- Learnable human force: bounded by `0.04 N`.
- Impedance prior: bounded by `0.03 N`.
- Learned human residual: bounded by `0.015 N`.
- Final composed human force: safety-clamped to `0.04 N`.

## Repository layout

```text
shared_control_isaac_sim/
├── assets/                         # Robot, tool, obstacle, and trajectory assets
├── scripts/
│   ├── train_{rmappo,maddpg,epigraph}.py
│   ├── play_{rmappo,maddpg,epigraph}.py
│   └── utils/                      # Seed, logging, runner, and checkpoint helpers
├── src/surgical_project/
│   ├── algorithms/marl/
│   │   ├── rmappo/
│   │   ├── maddpg/
│   │   └── epigraph/
│   └── envs/
│       ├── multi_agent/            # Shared rMAPPO/MADDPG environment
│       ├── multi_agent_epigraph/   # EPIGRAPH environment adapter
│       └── single_agent/           # Earlier single-agent/MBRL environment
├── tests/                          # Algorithm and environment regression tests
├── environment.yml
└── setup.py
```

## Requirements

- Linux with an NVIDIA GPU and a working NVIDIA driver.
- Python 3.10.
- Isaac Sim and Isaac Lab available in the active Python environment.
- PyTorch and NumPy.
- Optional: W&B for experiment tracking and `pytest` for tests.

Isaac Lab installation depends on the Isaac Sim release used on the machine. Install and verify Isaac Lab first, then install this repository into that same environment.

## Installation

### Use an existing Isaac Lab environment

The maintainers commonly use an environment named `env_isaaclab`:

```bash
conda activate env_isaaclab
cd /path/to/shared_control_isaac_sim
python -m pip install -e .
python -m pip install wandb pytest
```

### Use the included minimal Conda specification

```bash
conda env create -f environment.yml
conda activate surgical_robot_env
python -m pip install -e .
python -m pip install wandb pytest
```

After installation, verify that the Isaac Lab application can be imported:

```bash
python -c "from isaaclab.app import AppLauncher; print('Isaac Lab import OK')"
```

## Training

The following commands use the same seed, environment count, human model, and training budget for all algorithms. Add `--headless` for non-interactive training.

### rMAPPO

```bash
python -u scripts/train_rmappo.py \
  --human_model_type residual_impedance \
  --seed 42 \
  --num_envs 64 \
  --max_global_steps 250000 \
  --wandb
```

### MADDPG

```bash
python -u scripts/train_maddpg.py \
  --human_model_type residual_impedance \
  --seed 42 \
  --num_envs 64 \
  --max_global_steps 250000 \
  --wandb
```

MADDPG fills its joint replay buffer before training starts. Its replay capacity and warm-up threshold scale with `num_envs`; exploration noise is independent for every `(agent, environment)` pair.

### EPIGRAPH

```bash
python -u scripts/train_epigraph.py \
  --human_model_type residual_impedance \
  --seed 42 \
  --num_envs 64 \
  --max_global_steps 250000 \
  --wandb
```

EPIGRAPH conditions its actors, shared performance critic, and per-agent safety critics on `z`. The default support is `[-450, 300]`; endpoint-biased sampling and normalized `z` encoding are configured in YAML.

To run another human model, replace the mode in any command:

```bash
--human_model_type learnable
--human_model_type fixed_impedance
--human_model_type residual_impedance
```

## Configuration

The default experiment configurations are:

| Algorithm | Configuration |
|---|---|
| rMAPPO | `src/surgical_project/envs/multi_agent/agents/training_params_rmappo.yaml` |
| MADDPG | `src/surgical_project/envs/multi_agent/agents/training_params_maddpg.yaml` |
| EPIGRAPH | `src/surgical_project/envs/multi_agent_epigraph/agents/training_params_epigraph.yaml` |

Use a custom configuration with:

```bash
python -u scripts/train_rmappo.py --config /path/to/config.yaml ...
```

Command-line values for `seed`, `num_envs`, `human_model_type`, and `max_global_steps` override the corresponding new-run defaults. Each run writes its fully resolved configuration and a run manifest next to its checkpoints.

## Outputs and checkpoints

Default run directories are grouped by algorithm and human model:

```text
logs/rmappo_dual/<human_model_type>/<timestamp>/
logs/maddpg_dual/<human_model_type>/<timestamp>/
logs/epigraph/<human_model_type>/<timestamp>/
```

Each run contains:

```text
checkpoints/          # Milestone and final .pth checkpoints
resolved_config.yaml # Exact configuration used for the run
run_manifest.json    # Seed, model mode, environment count, and Git metadata
```

Resume a run by passing its checkpoint:

```bash
python -u scripts/train_rmappo.py \
  --checkpoint logs/rmappo_dual/residual_impedance/<timestamp>/checkpoints/<checkpoint>.pth
```

Use the matching training script for MADDPG or EPIGRAPH. A resumed run restores checkpoint configuration and validates runtime-sensitive settings instead of silently mixing them with a newer default YAML.

## Evaluation

Pass an explicit checkpoint to make the evaluated run unambiguous.

### rMAPPO

```bash
python -u scripts/play_rmappo.py \
  --checkpoint logs/rmappo_dual/residual_impedance/<timestamp>/checkpoints/<checkpoint>.pth \
  --num_envs 1 \
  --num_episodes 1 \
  --deterministic
```

### MADDPG

```bash
python -u scripts/play_maddpg.py \
  --checkpoint logs/maddpg_dual/residual_impedance/<timestamp>/checkpoints/<checkpoint>.pth \
  --num_envs 1 \
  --num_episodes 1 \
  --deterministic
```

MADDPG evaluation also supports `--video`, `--video_length`, `--camera_eye`, and `--camera_lookat`.

### EPIGRAPH

```bash
python -u scripts/play_epigraph.py \
  --checkpoint logs/epigraph/residual_impedance/<timestamp>/checkpoints/<checkpoint>.pth \
  --num_envs 1 \
  --num_episodes 1 \
  --deterministic
```

EPIGRAPH evaluation uses the learned safety critics and root finder to select `z` at each control step. Evaluation outputs can be redirected with `--save_dir`; use `--wandb` to upload evaluation metrics.

## Reproducibility

One experiment seed deterministically derives separate random streams instead of making every subsystem consume one global stream:

- Environment and network initialization.
- rMAPPO minibatch shuffling.
- MADDPG replay sampling and per-agent/per-environment exploration.
- EPIGRAPH policy sampling, minibatch shuffling, and `z` sampling.

This isolates unrelated stochastic operations: for example, replay sampling cannot shift an agent's future exploration sequence. Checkpoints also retain RNG and runtime state where required for continuation.

GPU PhysX execution can still introduce small platform-dependent numerical differences; reproduce experiments with the same code revision, Isaac stack, GPU configuration, resolved YAML, and seed.

## Tests

Run the regression suite inside the Isaac Lab environment:

```bash
pytest -q tests
```

The suite covers, among other things:

- Human impedance/residual composition and force bounds.
- MADDPG replay, human modes, exploration, and seed streams.
- rMAPPO recurrent pre-state, masks, timeout handling, tanh log-probability, and PPO ratios.
- EPIGRAPH reward decomposition, `z` dynamics, lambda-return DP, recurrent minibatches, and fixed-impedance behavior.

A manual simulator integration check is also provided:

```bash
python tests/isaac_runtime_transition_check.py
```

It launches a viewer and checks transition bookkeeping, evaluation masking, applied-force snapshots, and auto-reset behavior.

## Research notes

- `fixed_impedance` intentionally excludes the human Actor from optimization.
- `residual_impedance` learns only the bounded correction; the environment composes it with the analytic prior.
- rMAPPO and EPIGRAPH share the same recurrent PPO backbone hyperparameters so that differences are attributable primarily to the safety formulation.
- MADDPG uses physical actions in replay and converts them back to normalized critic coordinates during training.
- Console tracing is controlled by `logging.enable_console_logging` in each YAML file and is disabled by default.

This is research code. Checkpoint compatibility is validated by the training scripts, and older checkpoints may be rejected when action, observation, or EPIGRAPH semantics have changed.
