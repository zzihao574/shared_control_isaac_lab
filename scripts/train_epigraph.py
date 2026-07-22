#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Epigraph training script (Pure Epigraph: self-contained, milestone-driven)

Key features:
- No dependency on RMAPPO control flow or interfaces
- Default config path (no need to specify --config)
- Manual env instantiation with unified config injection
- Trainer receives full_config, ckpt_dir, max_global_steps
- Training loop: rollout → update → print → maybe_milestone_eval_and_save()
- StepTracer controlled by YAML (env self-prints when enabled)
"""

import os
import sys
import time
import argparse
import copy
import json
import random
import subprocess
import yaml
import numpy as np
import torch

# -------------------------------
# Ensure src/ is on sys.path
# -------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _path in (REPO_ROOT, SRC_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# -------------------------------
# Default Config Path
# -------------------------------
DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),        # scripts/
        "..", "src",
        "surgical_project",
        "envs",
        "multi_agent_epigraph",
        "agents",
        "training_params_epigraph.yaml",
    )
)

# -------------------------------
# 1) Parse args BEFORE AppLauncher
# -------------------------------
parser = argparse.ArgumentParser(description="Train Epigraph (multi-agent, milestone-driven).")
parser.add_argument(
    "--config",
    type=str,
    default=DEFAULT_CONFIG_PATH,
    help=f"Path to YAML config file. Defaults to {DEFAULT_CONFIG_PATH}",
)
parser.add_argument("--num_envs", type=int, default=None,
                    help="Number of parallel environments (default: 48).")
parser.add_argument("--seed", type=int, default=None,
                    help="Random seed (default: YAML seed).")
parser.add_argument(
    "--human_model_type",
    choices=("learnable", "fixed_impedance", "residual_impedance"),
    default=None,
)
parser.add_argument("--max_global_steps", type=int, default=0,
                    help="Training stops when trainer.global_step >= this value (0=use YAML config).")
parser.add_argument("--ckpt_dir", type=str, default=None,
                    help="Checkpoint root (default: logs/epigraph/<human_model_type>).")
parser.add_argument("--checkpoint", type=str, default="",
                    help="Path to checkpoint to resume training from.")
parser.add_argument(
    "--wandb",
    action="store_true",
    help="Enable WandB logging (overrides YAML; disabled by default).",
)

# Isaac App launcher args
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def setup_global_reproducibility(seed: int, strict_determinism: bool = True):
    """Set process RNG state before Isaac creates simulation components."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if not os.path.exists(args.config):
    raise FileNotFoundError(f"Config file not found: {args.config}")
with open(args.config, "r") as config_file:
    _resolved_config = yaml.safe_load(config_file)

if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_semantics = int(checkpoint.get("epigraph_semantics_version", 1))
    if saved_semantics != 10:
        raise ValueError(
            "This checkpoint predates the current EPIGRAPH z-support semantics; "
            "start a new EPIGRAPH run instead of resuming it."
        )
    checkpoint_params = checkpoint.get("params")
    if isinstance(checkpoint_params, dict):
        _resolved_config = copy.deepcopy(checkpoint_params)

configured_seed = int(
    _resolved_config.get("seed", _resolved_config.get("training", {}).get("seed", 42))
)
if args.checkpoint and args.seed is not None and int(args.seed) != configured_seed:
    raise ValueError(
        f"Cannot resume with a different seed: checkpoint={configured_seed}, CLI={args.seed}"
    )
args.seed = configured_seed if args.seed is None else int(args.seed)

configured_model = str(_resolved_config.get("human_model_type", "learnable"))
if args.checkpoint and args.human_model_type is not None and args.human_model_type != configured_model:
    raise ValueError(
        "Cannot resume with a different human_model_type: "
        f"checkpoint={configured_model}, CLI={args.human_model_type}"
    )
if args.human_model_type is not None:
    _resolved_config["human_model_type"] = args.human_model_type
else:
    _resolved_config.setdefault("human_model_type", configured_model)

runtime = _resolved_config.get("runtime", {})
checkpoint_num_envs = int(runtime.get("num_envs", 48))
if args.checkpoint and args.num_envs is not None and int(args.num_envs) != checkpoint_num_envs:
    raise ValueError(
        f"Cannot resume with a different num_envs: checkpoint={checkpoint_num_envs}, CLI={args.num_envs}"
    )
args.num_envs = checkpoint_num_envs if args.checkpoint else int(args.num_envs or 48)
_resolved_config["seed"] = args.seed
_resolved_config.setdefault("training", {})["seed"] = args.seed
setup_global_reproducibility(args.seed, strict_determinism=True)

# -------------------------------
# 2) Launch Isaac App
# -------------------------------
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import unified surgical_project modules (NOT isaaclab_tasks.*)
from surgical_project.envs.multi_agent_epigraph.surgical_epigraph_env_cfg import SurgicalEpigraphEnvCfg
from surgical_project.envs.multi_agent_epigraph.surgical_epigraph_env import SurgicalEpigraphEnv
from surgical_project.algorithms.marl.epigraph.trainer import EpigraphTrainer
from scripts.utils.training_helpers_epigraph import (
    TrainingConfiguration,
    print_training_progress,
)

_active_env = None
_active_trainer = None


def create_environment(config_params: dict, num_envs: int, seed: int):
    """
    Create SurgicalEpigraphEnv manually (no gym.make registry).
    Injects config params into env for reward computation.
    """
    # Create environment config
    env_cfg = SurgicalEpigraphEnvCfg()
    
    # Set parallel envs and seed
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = num_envs
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = seed
    
    # ✅ CRITICAL: Inject YAML params (for reward_components, scaling, etc.)
    env_cfg.params = config_params
    
    # Instantiate environment
    env = SurgicalEpigraphEnv(cfg=env_cfg, render_mode=None)
    
    # Also inject params to unwrapped env (ensures consistency)
    actual_env = getattr(env, "unwrapped", env)
    actual_env.params = config_params
    
    print(f"[ENV] Created SurgicalEpigraphEnv with {num_envs} parallel environments")
    return env


def check_step_tracer_status(config_params: dict):
    """
    Print StepTracer status based on YAML config.
    
    NOTE: Environment initializes StepTracer itself and controls printing.
    This function only provides visibility into the config setting.
    """
    logging_cfg = config_params.get("logging", {})
    enable = bool(logging_cfg.get("enable_console_logging", False))
    
    if enable:
        print_every = int(logging_cfg.get("print_every_steps", 10))
        max_print = int(logging_cfg.get("max_envs_to_print", 2))
        print(f"[STEPTRACER] Console debug is ENABLED by YAML (env will self-print).")
        print(f"[STEPTRACER] Settings: print_every_steps={print_every}, max_envs_to_print={max_print}")
    else:
        print("[STEPTRACER] Console debug is DISABLED (enable_console_logging=false in YAML).")


def get_device():
    """Get device (prefer CUDA if available)."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Main Training Function
# -------------------------------
def main():
    global _active_env, _active_trainer
    checkpoint_root = args.ckpt_dir or os.path.join(
        "logs", "epigraph", str(_resolved_config["human_model_type"])
    )
    print("\n" + "=" * 80)
    print("EPIGRAPH TRAINING (Pure Epigraph: Self-Contained)")
    print("=" * 80)
    print(f"Config:           {args.config}")
    print(f"Num Envs:         {args.num_envs}")
    print(f"Seed:             {args.seed}")
    print(f"Human Model:      {_resolved_config['human_model_type']}")
    max_steps_display = args.max_global_steps if args.max_global_steps > 0 else "YAML config"
    print(f"Max Global Steps: {max_steps_display}")
    print(f"Checkpoint Root:  {checkpoint_root}")
    if args.checkpoint:
        print(f"Resume from:      {args.checkpoint}")
    print("=" * 80 + "\n")
    
    # ===== 1. Load Configuration =====
    config_obj = TrainingConfiguration(copy.deepcopy(_resolved_config))
    config = config_obj.params  # Full YAML dict
    # Override WandB toggle via CLI (default: disabled unless --wandb provided)
    config.setdefault("logging", {})
    config["logging"]["use_wandb"] = bool(args.wandb)

    # ===== Scale milestone episodes by number of environments (align with rMAPPO) =====
    training_monitor_cfg = config.setdefault("training_monitor", {})
    milestone_episodes = list(training_monitor_cfg.get("milestone_episodes", []))
    milestones_already_scaled = bool(
        config.get("runtime", {}).get("milestones_scaled_by_num_envs", False)
    )
    if milestone_episodes and not milestones_already_scaled:
        scaled_milestones = [int(m * args.num_envs) for m in milestone_episodes]
        training_monitor_cfg["milestone_episodes"] = scaled_milestones
        print(f"[MILESTONES] Using scaled milestones (per-env -> total episodes): {scaled_milestones}")
    elif milestones_already_scaled:
        print("[MILESTONES] Restored already-scaled milestones from checkpoint")
    else:
        print("[MILESTONES] No milestone_episodes defined in config.")

    print(f"[CONFIG] Loaded configuration from: {args.config}\n")
    
    # ===== 2. Create Environment =====
    env = create_environment(config, args.num_envs, args.seed)
    _active_env = env
    
    # ===== 4. Check StepTracer Status =====
    check_step_tracer_status(config)
    
    # ===== 5. Extract Algorithm Configs =====
    algo_cfg = config["algorithms"]["epigraph"]
    yaml_max_steps = int(algo_cfg.get("max_global_steps", 150000))
    max_global_steps = args.max_global_steps if args.max_global_steps > 0 else yaml_max_steps
    source = "CLI" if args.max_global_steps > 0 else "YAML"
    print(f"[MAX_STEPS] Using max_global_steps={max_global_steps} (source={source})")
    epi_cfg = config["epigraph"]
    config["runtime"] = {
        "algorithm": "epigraph",
        "num_envs": int(args.num_envs),
        "max_global_steps": int(max_global_steps),
        "milestones_scaled_by_num_envs": True,
    }
    
    device = get_device()
    print(f"[DEVICE] Using device: {device}\n")
    
    # ===== 6. Create Trainer =====
    trainer = EpigraphTrainer(
        env=env,
        device=device,
        algo_cfg=algo_cfg,
        epi_cfg=epi_cfg,
        full_config=config,              # For milestone, wandb, etc.
        ckpt_dir=checkpoint_root,        # Trainer manages checkpoints
        max_global_steps=max_global_steps,
    )
    _active_trainer = trainer
    os.makedirs(trainer.ckpt_run_dir, exist_ok=True)
    resolved_config_path = os.path.join(trainer.ckpt_run_dir, "resolved_config.yaml")
    with open(resolved_config_path, "w") as resolved_file:
        yaml.safe_dump(config, resolved_file, sort_keys=False)
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        git_hash = "unknown"
    manifest = {
        "algorithm": "epigraph",
        "git_hash": git_hash,
        "seed": int(args.seed),
        "human_model_type": config["human_model_type"],
        "num_envs": int(args.num_envs),
        "max_global_steps": int(max_global_steps),
        "z_min": float(epi_cfg["z"]["min"]),
        "z_max": float(epi_cfg["z"]["max"]),
    }
    with open(os.path.join(trainer.ckpt_run_dir, "run_manifest.json"), "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    
    print(f"[TRAINER] EpigraphTrainer initialized\n")
    
    # ===== 7. Resume from Checkpoint (optional) =====
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"[RESUME] Loading checkpoint from: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
        print(f"[RESUME] Resumed at global_step={trainer.global_step}\n")
    
    # ===== 8. Training Loop =====
    print("[TRAIN] Starting training loop...\n")
    t0 = time.time()
    
    while trainer.global_step < max_global_steps:
        # Collect rollout
        rollout_stats = trainer.collect_rollout()
        
        # Update policy and value functions
        update_stats = trainer.update()
        
        # Print progress periodically (every 2000 steps)
        if trainer.global_step % 2000 == 0:
            elapsed = time.time() - t0
            print(f"[STEP {trainer.global_step}] Elapsed: {elapsed/60:.1f} min")
            print_training_progress(
                global_step=trainer.global_step,
                max_steps=max_global_steps,
                rollout_stats=rollout_stats,
                update_stats=update_stats,
                agent_labels=trainer.agent_ids,
            )
        
    # ===== 9. Final Checkpoint =====
    final_ckpt = os.path.join(
        trainer.ckpt_dir,
        f"epigraph_final_step{trainer.global_step}.pth",
    )
    os.makedirs(os.path.dirname(final_ckpt), exist_ok=True)
    trainer.save_checkpoint(final_ckpt)
    print(f"\n[DONE] Training completed at {trainer.global_step} steps")
    print(f"[DONE] Final checkpoint saved: {final_ckpt}")
    
    # ===== 10. Cleanup =====
    env.close()
    _active_env = None
    if trainer.wandb_logger is not None:
        trainer.wandb_logger.finish()
    _active_trainer = None
    print("[TRAIN] Environment closed\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        if _active_trainer is not None and _active_trainer.wandb_logger is not None:
            _active_trainer.wandb_logger.finish()
        if _active_env is not None:
            _active_env.close()
        print("[CLEANUP] Closing simulation app...")
        simulation_app.close()
        print("[CLEANUP] Done.")
