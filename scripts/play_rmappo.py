#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
play_rmappo.py - Evaluation script for trained rMAPPO dual networks
Compatible with unified --checkpoint interface for milestone checkpoints
Features StepTracer integration for detailed console debugging
"""

import os
import sys
import torch
import argparse

# === Step 1: Parse arguments BEFORE AppLauncher ===
parser = argparse.ArgumentParser(description="Play trained rMAPPO agent.")
parser.add_argument("--config", type=str, required=True,
                    help="Path to training YAML config.")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained checkpoint (.pth) - unified interface.")
parser.add_argument("--task", type=str,
                    default="Isaac-Surgical-MARL-Direct-v0",
                    help="Environment task name (must match training).")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility.")

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

from utils.training_helpers_rmappo import TrainingConfiguration, MetricsHub
from train_rmappo import (
    setup_global_reproducibility,
    setup_environment,
    initialize_rmappo_algorithm,
)

# ----------------------------- #
# Helpers
# ----------------------------- #

def get_hidden_size(config, default: int = 256) -> int:
    """Safely get hidden_size from YAML; fallback to default if missing."""
    try:
        return int(config.params["algorithms"]["rmappo"]["hidden_size"])
    except Exception:
        print(f"[WARN] 'algorithms.rmappo.hidden_size' not found in config. Use default={default}.")
        return default


def load_checkpoint(rmappo_wrapper, checkpoint_path: str):
    """
    Load trained weights into dual rMAPPO networks.
    
    Compatible with unified flat checkpoint format:
        - human_actor, human_critic, robot_actor, robot_critic (required)
        - optim_state, rng_state, counters (optional, ignored for play)
    """
    print(f"[LOAD] Loading checkpoint: {checkpoint_path}")
    
    # FIXED: PyTorch 2.6+ requires weights_only=False for checkpoints with RNG states
    ckpt = torch.load(checkpoint_path, map_location=rmappo_wrapper.device, weights_only=False)

    # Check for flat keys
    flat_keys = {"human_actor", "human_critic", "robot_actor", "robot_critic"}
    
    if not flat_keys.issubset(ckpt.keys()):
        raise KeyError(
            f"Checkpoint missing required keys. Expected: {flat_keys}\n"
            f"Found: {list(ckpt.keys())[:20]}..."
        )
    
    # Load network weights
    rmappo_wrapper.policies["human"].actor.load_state_dict(ckpt["human_actor"])
    rmappo_wrapper.policies["human"].critic.load_state_dict(ckpt["human_critic"])
    rmappo_wrapper.policies["robot"].actor.load_state_dict(ckpt["robot_actor"])
    rmappo_wrapper.policies["robot"].critic.load_state_dict(ckpt["robot_critic"])
    
    print("[LOAD] Successfully loaded both networks (human & robot)")

    # Optional metadata logging
    for k in ("milestone", "score", "global_steps_total", "episodes_done_total"):
        if k in ckpt:
            print(f"[INFO] {k}: {ckpt[k]}")


def inject_step_tracer(env, config, num_envs):
    """
    Inject StepTracer into environment for detailed console debugging.
    Enable/disable controlled by YAML config: logging.enable_console_logging
    """
    actual_env = getattr(env, "unwrapped", env)
    
    from surgical_project.envs.multi_agent.utils import StepTracer
    
    # Read console logging flag from YAML
    enable_logging = config.params.get("logging", {}).get("enable_console_logging", False)
    
    actual_env.step_tracer = StepTracer(
        num_envs=num_envs,
        device=getattr(actual_env, "device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
        enable_console_logging=enable_logging,
        print_every_steps=1  # Print every step during evaluation
    )
    
    if enable_logging:
        print(f"[STEPTRACER] Enabled with print_every_steps=1")
    else:
        print(f"[STEPTRACER] Disabled (set logging.enable_console_logging=true in YAML to enable)")


# ----------------------------- #
# Main
# ----------------------------- #

def main():
    print("=" * 80)
    print("rMAPPO Dual Network Evaluation")
    print("=" * 80)

    # Force single environment for evaluation
    args.num_envs = 1
    print(f"[SETUP] num_envs set to: {args.num_envs}")

    # Load configuration
    print(f"[SETUP] Loading config from: {args.config}")
    config = TrainingConfiguration.from_yaml(args.config)
    print(f"[SETUP] Config loaded successfully")

    setup_global_reproducibility(args.seed, strict_determinism=True)

    # Create environment
    print(f"[SETUP] Creating environment...")
    env, env_cfg = setup_environment(args, config)
    print(f"[SETUP] Environment created successfully")

    # Inject params into environment
    actual_env = getattr(env, 'unwrapped', env)
    if hasattr(actual_env, "params"):
        actual_env.params = config.params
        print(f"[SETUP] Params injected into environment")

    # Inject StepTracer (controlled by YAML)
    inject_step_tracer(env, config, args.num_envs)

    # Initialize rMAPPO wrapper
    print(f"[SETUP] Initializing rMAPPO algorithm...")
    metrics_hub = MetricsHub()
    rmappo = initialize_rmappo_algorithm(env, config, args, metrics_hub)
    print(f"[SETUP] rMAPPO initialized successfully")

    # Load trained weights
    load_checkpoint(rmappo, args.checkpoint)

    # Set evaluation mode
    rmappo.set_eval_mode(True)
    print(f"[SETUP] Evaluation mode enabled")

    # === Reset environment and RNN states ===
    print(f"[RESET] Resetting environment...")
    obs, _ = env.reset()
    print(f"[RESET] Environment reset complete")
    try:
        print(f"[RESET] Observation keys: {list(obs.keys())}")
        obs_shapes = {k: v.shape for k, v in obs.items()}
        print(f"[RESET] Observation shapes: {obs_shapes}")
    except Exception:
        pass

    # Hidden size
    hidden_size = get_hidden_size(config, default=256)
    print(f"[RESET] hidden_size = {hidden_size}")

    # Clear RNN states
    for aid in rmappo.agent_ids:
        rmappo.rnn_states[aid]["actor"] = torch.zeros(1, hidden_size, device=rmappo.device)
        rmappo.rnn_states[aid]["critic"] = torch.zeros(1, hidden_size, device=rmappo.device)
    print(f"[RESET] RNN states initialized for agents: {rmappo.agent_ids}")

    # === Evaluation loop ===
    print(f"\n[EVAL] Starting evaluation loop...")
    total_return = 0.0
    steps = 0
    max_steps = 2000

    print("\n" + "=" * 80)
    print("Starting evaluation...")
    print("=" * 80 + "\n")

    with torch.no_grad():
        while simulation_app.is_running() and (max_steps <= 0 or steps < max_steps):
            # Select deterministic actions
            actions, detail = rmappo.select_actions(
                obs,
                add_noise=False,
                deterministic=True
            )

            # Set detail info for StepTracer
            if hasattr(actual_env, 'set_detail_actor_info'):
                actual_env.set_detail_actor_info(detail)

            # Step environment
            obs, rewards, terminated, truncated, _ = env.step(actions)

            # Trigger StepTracer (if enabled)
            if hasattr(actual_env, 'step_tracer') and actual_env.step_tracer is not None:
                # Force print every step during evaluation
                actual_env.step_tracer.maybe_print_step(
                    actual_env, 
                    rewards, 
                    global_step=steps,
                    force_print=True
                )

            # Average reward across agents
            avg_reward = torch.stack([rewards[aid] for aid in rmappo.agent_ids]).mean()
            total_return += float(avg_reward.item())
            steps += 1

            if steps % 100 == 0:
                print(f"[Step {steps}] Return so far: {total_return:.3f}")

            # Check done
            done_any = None
            for aid in rmappo.agent_ids:
                d = (terminated[aid] | truncated[aid]).to(torch.bool)
                done_any = d if done_any is None else (done_any | d)
            if done_any.item():
                print(f"\n[DONE] Episode terminated at step {steps}")
                break

    # === Final score ===
    final_score = total_return * 1000.0 / max(1, steps)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Steps    : {steps}")
    print(f"Total Return   : {total_return:.3f}")
    print(f"Final Score    : {final_score:.2f}   (total_return * 1000 / steps)")
    print(f"Avg Return/Step: {total_return / max(1, steps):.4f}")
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