#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
play_epigraph.py - Evaluation script for trained Epigraph dual networks
Features RootFinder integration for z* computation and deterministic policy evaluation.
Compatible with unified --checkpoint interface for milestone checkpoints.

KEY FIX: Added complete eval_step_with_root() function for proper z* solving.
"""

import os
import sys
import torch
import argparse

# === Step 1: Parse arguments BEFORE AppLauncher ===
parser = argparse.ArgumentParser(description="Play trained Epigraph agent.")
parser.add_argument("--config", type=str, required=True,
                    help="Path to training YAML config.")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained checkpoint (.pth) - unified interface.")
parser.add_argument("--task", type=str,
                    default="Isaac-Surgical-MARL-Epigraph-v0",
                    help="Environment task name (must match training).")
parser.add_argument("--seed", type=str, default=42,
                    help="Random seed for reproducibility.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel environments (forced to 1 for eval).")

# Import AppLauncher and add its args
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# === Step 2: Launch Isaac App ===
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# === Step 3: Import after Isaac is initialized ===
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

from utils.training_helpers_epigraph import TrainingConfiguration
import gymnasium as gym

# ----------------------------- #
# Helpers
# ----------------------------- #

def setup_global_reproducibility(seed: int, strict_determinism: bool = True):
    """Setup global random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if strict_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"[SETUP] Global seed set to: {seed}")


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load trained Epigraph checkpoint.
    
    Checkpoint contains:
        - human_actor, human_critic_vl, human_critic_vh
        - robot_actor, robot_critic_vl, robot_critic_vh
        - z_encoder (shared)
        - optim_state, rng_state, counters (optional, ignored for play)
    
    Returns:
        checkpoint: Dictionary with network states
    """
    print(f"[LOAD] Loading checkpoint: {checkpoint_path}")
    
    # PyTorch 2.6+ requires weights_only=False for checkpoints with RNG states
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check for required keys
    required_keys = {
        "z_encoder",
        "critic_vl",
    }
    
    # Check for per-agent keys
    agents = ["human", "robot"]
    for agent in agents:
        required_keys.add(f"actor_{agent}")
        required_keys.add(f"critic_vh_{agent}")
    
    if not required_keys.issubset(ckpt.keys()):
        raise KeyError(
            f"Checkpoint missing required keys. Expected: {required_keys}\n"
            f"Found: {list(ckpt.keys())[:20]}..."
        )
    
    print("[LOAD] Successfully loaded checkpoint")
    
    # Optional metadata logging
    for k in ("global_step", "episodes_done"):
        if k in ckpt:
            print(f"[INFO] {k}: {ckpt[k]}")
    
    return ckpt


def create_trainer_from_checkpoint(env, config, ckpt, device):
    """
    Create EpigraphTrainer and load checkpoint weights.
    """
    # Import trainer
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "surgical_project", "algorithms", "marl", "epigraph"))
        from trainer import EpigraphTrainer
        
        # Create trainer
        trainer = EpigraphTrainer(
            env=env,
            device=device,
            algo_cfg=config.params["algorithms"]["rmappo"],  # Reuse rmappo config
            epi_cfg=config.params["epigraph"],  # Epigraph-specific config
        )
        
        # Load network weights
        trainer.z_encoder.load_state_dict(ckpt["z_encoder"])
        trainer.critic_vl.load_state_dict(ckpt["critic_vl"])
        
        for agent in ["human", "robot"]:
            trainer.actors[agent].load_state_dict(ckpt[f"actor_{agent}"])
            trainer.critics_vh[agent].load_state_dict(ckpt[f"critic_vh_{agent}"])
        
        # Set eval mode
        trainer.set_eval_mode()
        
        print("[LOAD] Trainer created and weights loaded")
        return trainer
        
    except ImportError as e:
        print(f"[ERROR] Could not import EpigraphTrainer: {e}")
        print("[INFO] Using placeholder - evaluation will not work correctly")
        return None


# ----------------------------- #
# Evaluation Step with Root Finding
# ----------------------------- #

@torch.no_grad()
def eval_step_with_root(trainer, obs):
    """
    Single evaluation step with root finding for z*.
    
    Process:
    1. For each agent, solve Vh(o_i, z*) - z* = 0 to get z_i*
    2. Compute z_global = max(z_i*) across agents
    3. Use z_global for all actors to get deterministic actions
    
    Args:
        trainer: EpigraphTrainer instance
        obs: Dict[agent, [B, obs_dim]] - observations
    
    Returns:
        actions: Dict[agent, [B, act_dim]] - actions
        z_global: [B, 1] - global z value
    """
    masks_ones = torch.ones(trainer.num_envs, 1, device=trainer.device)
    z_stars = []
    
    # ========== 1) Solve z* for each agent ==========
    for agent in trainer.agent_ids:
        # Define Vh function for root finding
        def vh_fn(z, obs_a, rnn_a, mask_a):
            """Wrapper function for Vh(obs, z)."""
            z_enc = trainer.z_encoder(z)  # [B, nz]
            vh, _ = trainer.critics_vh[agent](obs_a, z_enc, rnn_a, mask_a)
            return vh
        
        # Solve Vh(o, z*) - z* = 0
        z_star = trainer.root_finder.solve(
            vh_fn,
            obs[agent],
            trainer.rnn_states[agent]["vh"],
            masks_ones
        )
        z_stars.append(z_star)
    
    # ========== 2) Compute global z = max(z_i*) ==========
    z_global = torch.max(torch.stack(z_stars, dim=0), dim=0)[0]  # [B, 1]
    z_enc_global = trainer.z_encoder(z_global)
    
    # ========== 3) Get deterministic actions with shared z_global ==========
    actions = {}
    for agent in trainer.agent_ids:
        # Deterministic action from actor
        a, _, h_next = trainer.actors[agent](
            obs[agent],
            z_enc_global,
            trainer.rnn_states[agent]["actor"],
            masks_ones,
            deterministic=True
        )
        
        # Update RNN state
        trainer.rnn_states[agent]["actor"] = h_next
        actions[agent] = a
    
    return actions, z_global


# ----------------------------- #
# Main
# ----------------------------- #

def main():
    print("=" * 80)
    print("Epigraph Dual Network Evaluation with RootFinder")
    print("=" * 80)

    # Force single environment for evaluation
    args.num_envs = 1
    print(f"[SETUP] num_envs set to: {args.num_envs}")

    # Load configuration
    print(f"[SETUP] Loading config from: {args.config}")
    config = TrainingConfiguration.from_yaml(args.config)
    print(f"[SETUP] Config loaded successfully")

    setup_global_reproducibility(args.seed, strict_determinism=True)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[SETUP] Using device: {device}")

    # Create environment
    print(f"[SETUP] Creating environment: {args.task}")
    env = gym.make(args.task, num_envs=args.num_envs)
    print(f"[SETUP] Environment created successfully")
    
    # Inject config into environment
    if hasattr(env.unwrapped, "params"):
        env.unwrapped.params = config.params
        print(f"[SETUP] Config injected into environment")

    # Load checkpoint
    ckpt = load_checkpoint(args.checkpoint, device)
    
    # Create trainer and load weights
    trainer = create_trainer_from_checkpoint(env, config, ckpt, device)
    
    if trainer is None:
        print("[ERROR] Failed to create trainer - cannot proceed with evaluation")
        return

    # === Reset environment ===
    print(f"[RESET] Resetting environment...")
    obs, _ = env.reset()
    print(f"[RESET] Environment reset complete")
    try:
        print(f"[RESET] Observation keys: {list(obs.keys())}")
        obs_shapes = {k: v.shape for k, v in obs.items()}
        print(f"[RESET] Observation shapes: {obs_shapes}")
    except Exception:
        pass

    # === Evaluation loop ===
    print(f"\n[EVAL] Starting evaluation loop with RootFinder...")
    total_return = 0.0
    steps = 0
    max_steps = 2000
    
    z_global_values = []

    print("\n" + "=" * 80)
    print("Starting evaluation...")
    print("=" * 80 + "\n")

    while simulation_app.is_running() and (max_steps <= 0 or steps < max_steps):
        # ============ EPIGRAPH-SPECIFIC: Solve for z* using RootFinder ============
        actions, z_global = eval_step_with_root(trainer, obs)
        z_global_values.append(z_global.mean().item())
        
        if steps % 100 == 0:
            print(f"[Step {steps}] z_global={z_global[0].item():.4f}")
        
        # ============ Step environment ============
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Average reward across agents
        avg_reward = sum(rewards[aid].item() for aid in ["human", "robot"]) / 2.0
        total_return += avg_reward
        steps += 1

        if steps % 100 == 0:
            print(f"[Step {steps}] Return so far: {total_return:.3f}")

        # Check done
        done = any(terminated[aid].item() or truncated[aid].item() 
                  for aid in ["human", "robot"])
        if done:
            print(f"\n[DONE] Episode terminated at step {steps}")
            break

    # === Final score ===
    final_score = total_return * 1000.0 / max(1, steps)
    z_global_mean = sum(z_global_values) / len(z_global_values) if z_global_values else 0.0

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Steps    : {steps}")
    print(f"Total Return   : {total_return:.3f}")
    print(f"Final Score    : {final_score:.2f}   (total_return * 1000 / steps)")
    print(f"Avg Return/Step: {total_return / max(1, steps):.4f}")
    print(f"Z Global Mean  : {z_global_mean:.4f}")
    print("=" * 80 + "\n")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR OCCURRED")
        print("=" * 80)
        print(f"Exception type   : {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("=" * 80 + "\n")
    finally:
        print("[CLEANUP] Closing simulation app...")
        simulation_app.close()
        print("[CLEANUP] Done")