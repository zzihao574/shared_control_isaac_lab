#!/usr/bin/env python3

"""Quick test script for the surgical environment."""

import argparse
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test surgical environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

# Add the src directory to Python path to find our modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our environment
import surgical_project.envs.single_agent


def test_environment():
    """Test the surgical environment basic functionality."""
    
    print("Creating surgical environment...")
    
    try:
        # Import configuration first
        from surgical_project.envs.single_agent.surgical_direct_env_cfg import SurgicalDirectEnvCfg
        
        # Create configuration
        env_cfg = SurgicalDirectEnvCfg()
        
        # Override number of environments
        env_cfg.scene.num_envs = args_cli.num_envs
        
        # Create environment with configuration (Isaac Lab style)
        env = gym.make("Isaac-Surgical-Direct-v0", cfg=env_cfg)
        
        print(f"Environment created with {env.unwrapped.num_envs} environments")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print(f"Device: {env.unwrapped.device}")
        print(f"Simulation scale: {env.unwrapped.cfg.simulation_scale}x")
        print(f"Network operates in real-world scale (observations and actions)")
        
        # Reset environment
        print("\nResetting environment...")
        obs_dict, info = env.reset()
        obs = obs_dict["policy"]
        print(f"Observation shape: {obs.shape}")
        print(f"Sample observation (real-world scale): {obs[0]}")
        
        # Show the scaling is working
        scalpel_pos_sim = env.unwrapped._scalpel.data.root_pos_w[0]  # Simulation scale
        scalpel_pos_real = obs[0, :3]  # Real world scale from network observations
        print(f"Scalpel position in simulation: {scalpel_pos_sim}")
        print(f"Scalpel position for network (real-world): {scalpel_pos_real}")
        print(f"Scale ratio: {scalpel_pos_sim / scalpel_pos_real} (should be ~{env.unwrapped.cfg.simulation_scale})")
        
        # Test random actions
        print("\nTesting random actions...")
        print("Note: Actions are in real-world scale, then scaled up 10x for simulation")
        
        for step in range(100):
            # Generate random actions
            actions = torch.rand(env.unwrapped.num_envs, 3, device=env.unwrapped.device) * 2 - 1
            
            # Step environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            obs = obs_dict["policy"]
            
            if step % 20 == 0:
                print(f"Step {step:3d}: avg_reward={rewards.mean().item():.3f}, "
                      f"terminated={terminated.sum().item()}, "
                      f"truncated={truncated.sum().item()}")
                
                # Print some observation statistics
                scalpel_pos = obs[:, :3]  # First 3 elements are position
                print(f"         Scalpel position range: "
                      f"x=[{scalpel_pos[:, 0].min().item():.3f}, {scalpel_pos[:, 0].max().item():.3f}], "
                      f"y=[{scalpel_pos[:, 1].min().item():.3f}, {scalpel_pos[:, 1].max().item():.3f}], "
                      f"z=[{scalpel_pos[:, 2].min().item():.3f}, {scalpel_pos[:, 2].max().item():.3f}]")
            
            # Reset environments that are done
            done = terminated | truncated
            if done.any():
                print(f"         Resetting {done.sum().item()} environments")
        
        print("\nEnvironment test completed successfully!")
        
        # Test reward components
        print("\nTesting reward components...")
        if hasattr(env.unwrapped, 'extras') and 'log' in env.unwrapped.extras:
            for key, value in env.unwrapped.extras['log'].items():
                print(f"  {key}: {value:.4f}")
        
        env.close()
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_environment()
    simulation_app.close()