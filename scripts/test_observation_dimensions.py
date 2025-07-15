#!/usr/bin/env python3

"""Quick test script to check observation dimensions"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from surgical_project.envs.multi_agent.surgical_direct_marl_env import SurgicalDirectMARLEnv
from surgical_project.envs.multi_agent.surgical_direct_marl_env_cfg import SurgicalDirectMARLEnvCfg

def test_observation_dimensions():
    """Test observation dimensions."""
    print("Testing observation dimensions...")
    
    # Create environment config
    cfg = SurgicalDirectMARLEnvCfg()
    cfg.scene.num_envs = 4  # Small number for testing
    
    # Create environment
    env = SurgicalDirectMARLEnv(cfg)
    
    # Reset environment
    obs_dict, info = env.reset()
    
    print(f"\nEnvironment created successfully!")
    print(f"Number of environments: {env.num_envs}")
    print(f"Possible agents: {env.cfg.possible_agents}")
    
    print(f"\nObservation dictionary keys: {list(obs_dict.keys())}")
    
    for agent_name, obs in obs_dict.items():
        print(f"\nAgent: {agent_name}")
        print(f"  Observation tensor shape: {obs.shape}")
        print(f"  Expected from config: {cfg.observation_spaces[agent_name]}")
        print(f"  Sample observation (first env, first 10 values): {obs[0, :10]}")
        
        # Check if dimensions match
        expected_dim = cfg.observation_spaces[agent_name]
        actual_dim = obs.shape[1]
        
        if expected_dim == actual_dim:
            print(f"  ✓ Dimensions match!")
        else:
            print(f"  ✗ Dimension mismatch! Expected: {expected_dim}, Got: {actual_dim}")
    
    # Test step
    print(f"\nTesting environment step...")
    actions = {}
    for agent_name in env.cfg.possible_agents:
        action_dim = env.cfg.action_spaces[agent_name]
        actions[agent_name] = env.sample_actions_random(agent_name) if hasattr(env, 'sample_actions_random') else \
                              torch.zeros(env.num_envs, action_dim, device=env.device)
    
    try:
        obs_dict, rew_dict, term_dict, trunc_dict, info = env.step(actions)
        print(f"  ✓ Environment step successful!")
        
        for agent_name, obs in obs_dict.items():
            print(f"  Agent {agent_name} new obs shape: {obs.shape}")
            
    except Exception as e:
        print(f"  ✗ Environment step failed: {e}")
    
    # Close environment
    env.close()
    print(f"\nTest completed!")

if __name__ == "__main__":
    import torch
    try:
        test_observation_dimensions()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()