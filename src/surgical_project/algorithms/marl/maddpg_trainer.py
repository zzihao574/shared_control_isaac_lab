# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PyTorch MADDPG Trainer for Surgical Human-Robot Collaboration"""

import torch
import numpy as np
import time
import os
from typing import List, Dict, Optional
import pickle

from .maddpg_agent import MADDPGAgent
from surgical_project.envs.multi_agent.surgical_direct_marl_env import SurgicalDirectMARLEnv
from surgical_project.envs.multi_agent.surgical_direct_marl_env_cfg import SurgicalDirectMARLEnvCfg


class MADDPGTrainer:
    """MADDPG Trainer for Surgical Human-Robot Collaboration."""
    
    def __init__(self, config: Dict):
        """Initialize MADDPG trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create environment
        env_cfg = SurgicalDirectMARLEnvCfg()
        env_cfg.scene.num_envs = config.get("num_envs", 512)
        env_cfg.episode_length_s = config.get("episode_length_s", 10.0)
        
        self.env = SurgicalDirectMARLEnv(env_cfg)
        self.max_episode_len = config.get("max_episode_len", int(env_cfg.episode_length_s / self.env.step_dt))
        
        # Agent configuration
        self.agent_names = ["human", "robot"]
        self.num_agents = len(self.agent_names)
        
        # Get actual observation dimensions from environment
        obs_dict, _ = self.env.reset()
        sample_obs = obs_dict[self.agent_names[0]][0]  # Get first agent's first observation
        self.obs_dim = sample_obs.shape[0]  # Actual observation dimension
        self.action_dim = 3  # From environment config
        
        print(f"[INFO] Detected observation dimension: {self.obs_dim}")
        print(f"[INFO] Action dimension: {self.action_dim}")
        
        # Debug: print observation shapes for all agents
        for agent_name in self.agent_names:
            if agent_name in obs_dict:
                obs_shape = obs_dict[agent_name][0].shape
                print(f"[DEBUG] Agent '{agent_name}' observation shape: {obs_shape}")
        
        # Reset environment after getting dimensions
        obs_dict, _ = self.env.reset()
        
        # Training parameters
        self.lr = config.get("lr", 1e-2)
        self.gamma = config.get("gamma", 0.95)
        self.batch_size = config.get("batch_size", 1024)
        self.num_episodes = config.get("num_episodes", 60000)
        self.save_rate = config.get("save_rate", 1000)
        self.num_units = config.get("num_units", 64)
        
        # Create agents
        self.agents = []
        for i, name in enumerate(self.agent_names):
            # Human agent uses local Q-function, robot uses global
            local_q_func = (name == "human")  # Human focuses on local optimization
            
            agent = MADDPGAgent(
                name=name,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                num_agents=self.num_agents,
                agent_index=i,
                lr=self.lr,
                hidden_dim=self.num_units,
                local_q_func=local_q_func,
                device=self.device
            )
            self.agents.append(agent)
        
        # Logging
        self.episode_rewards = []
        self.agent_rewards = [[] for _ in range(self.num_agents)]
        self.final_ep_rewards = []
        self.final_ep_ag_rewards = []
        
        print(f"[INFO] MADDPG Trainer initialized")
        print(f"  - Environment: {self.num_agents} agents, {self.env.num_envs} parallel envs")
        print(f"  - Max episode length: {self.max_episode_len}")
        print(f"  - Device: {self.device}")
        print(f"  - Agents: {[agent.name for agent in self.agents]}")
    
    def train(self) -> None:
        """Main training loop."""
        print(f"[INFO] Starting MADDPG training for {self.num_episodes} episodes")
        
        # Reset environment
        obs_dict, _ = self.env.reset()
        obs_n = [obs_dict[agent.name][0].cpu().numpy() for agent in self.agents]  # Take first env
        
        episode_step = 0
        train_step = 0
        episode_count = 0
        t_start = time.time()
        
        # Initialize episode tracking
        self.episode_rewards.append(0.0)
        for i in range(self.num_agents):
            self.agent_rewards[i].append(0.0)
        
        print("[INFO] Starting training iterations...")
        
        while episode_count < self.num_episodes:
            # Get actions from all agents
            actions_dict = {}
            action_n = []
            
            for i, agent in enumerate(self.agents):
                action = agent.action(obs_n[i], add_noise=True)
                action_n.append(action)
                # Expand action to all environments
                actions_dict[agent.name] = torch.FloatTensor(action).unsqueeze(0).repeat(
                    self.env.num_envs, 1
                ).to(self.device)
            
            # Environment step
            new_obs_dict, rew_dict, done_dict, truncated_dict, info_dict = self.env.step(actions_dict)
            
            # Extract data for first environment
            new_obs_n = [new_obs_dict[agent.name][0].cpu().numpy() for agent in self.agents]
            rew_n = [rew_dict[agent.name][0].cpu().numpy() for agent in self.agents]
            done_n = [done_dict[agent.name][0].cpu().numpy() for agent in self.agents]
            
            episode_step += 1
            done = any(done_n) or any(truncated_dict[agent.name][0].cpu().numpy() for agent in self.agents)
            terminal = (episode_step >= self.max_episode_len)
            
            # Store experience for each agent
            for i, agent in enumerate(self.agents):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done or terminal)
            
            # Update observations
            obs_n = new_obs_n
            
            # Update episode rewards
            for i, rew in enumerate(rew_n):
                self.episode_rewards[-1] += rew
                self.agent_rewards[i][-1] += rew
            
            # Episode end handling
            if done or terminal:
                # Reset environment
                obs_dict, _ = self.env.reset()
                obs_n = [obs_dict[agent.name][0].cpu().numpy() for agent in self.agents]
                
                episode_step = 0
                episode_count += 1
                
                # Start new episode tracking
                self.episode_rewards.append(0.0)
                for i in range(self.num_agents):
                    self.agent_rewards[i].append(0.0)
            
            train_step += 1
            
            # Update all agents
            losses = []
            for agent in self.agents:
                agent.preupdate()
            
            for agent in self.agents:
                loss = agent.update(self.agents, train_step)
                if loss is not None:
                    losses.append(loss)
            
            # Logging and saving
            if terminal and (episode_count % self.save_rate == 0):
                # Save models
                self.save_models(f"episode_{episode_count}")
                
                # Print training progress
                recent_episodes = min(self.save_rate, len(self.episode_rewards) - 1)
                mean_reward = np.mean(self.episode_rewards[-recent_episodes:])
                
                print(f"Steps: {train_step}, Episodes: {episode_count}, "
                      f"Mean episode reward: {mean_reward:.3f}")
                
                agent_mean_rewards = []
                for i in range(self.num_agents):
                    agent_mean_reward = np.mean(self.agent_rewards[i][-recent_episodes:])
                    agent_mean_rewards.append(agent_mean_reward)
                    print(f"  {self.agents[i].name}: {agent_mean_reward:.3f}")
                
                if losses:
                    mean_loss = np.mean([loss[0] for loss in losses])  # Q-loss
                    print(f"  Mean Q-loss: {mean_loss:.6f}")
                
                print(f"  Time: {time.time() - t_start:.2f}s")
                t_start = time.time()
                
                # Track final episode rewards
                self.final_ep_rewards.append(mean_reward)
                for i, agent_reward in enumerate(agent_mean_rewards):
                    self.final_ep_ag_rewards.append(agent_reward)
        
        # Save final results
        self.save_training_results()
        print(f"[INFO] Training completed after {episode_count} episodes")
    
    def save_models(self, checkpoint_name: str) -> None:
        """Save all agent models."""
        save_dir = self.config.get("save_dir", "/tmp/surgical_maddpg/")
        os.makedirs(save_dir, exist_ok=True)
        
        for agent in self.agents:
            filepath = os.path.join(save_dir, f"{checkpoint_name}_{agent.name}.pth")
            agent.save(filepath)
    
    def load_models(self, checkpoint_name: str) -> None:
        """Load all agent models."""
        save_dir = self.config.get("save_dir", "/tmp/surgical_maddpg/")
        
        for agent in self.agents:
            filepath = os.path.join(save_dir, f"{checkpoint_name}_{agent.name}.pth")
            if os.path.exists(filepath):
                agent.load(filepath)
                print(f"[INFO] Loaded model for agent {agent.name}")
            else:
                print(f"[WARNING] Model file not found: {filepath}")
    
    def save_training_results(self) -> None:
        """Save training curves and results."""
        plots_dir = self.config.get("plots_dir", "./learning_curves/")
        exp_name = self.config.get("exp_name", "surgical_maddpg")
        
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save reward curves
        rew_file = os.path.join(plots_dir, f"{exp_name}_rewards.pkl")
        with open(rew_file, 'wb') as f:
            pickle.dump(self.final_ep_rewards, f)
        
        agrew_file = os.path.join(plots_dir, f"{exp_name}_agrewards.pkl")
        with open(agrew_file, 'wb') as f:
            pickle.dump(self.final_ep_ag_rewards, f)
        
        print(f"[INFO] Training results saved to {plots_dir}")
    
    def evaluate(self, num_episodes: int = 10, render: bool = False) -> Dict:
        """Evaluate trained agents.
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render the environment
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"[INFO] Evaluating agents for {num_episodes} episodes")
        
        episode_rewards = []
        agent_rewards = [[] for _ in range(self.num_agents)]
        success_count = 0
        collision_count = 0
        
        for episode in range(num_episodes):
            obs_dict, _ = self.env.reset()
            obs_n = [obs_dict[agent.name][0].cpu().numpy() for agent in self.agents]
            
            episode_reward = 0.0
            agent_episode_rewards = [0.0 for _ in range(self.num_agents)]
            episode_step = 0
            
            while episode_step < self.max_episode_len:
                # Get actions (no exploration noise)
                actions_dict = {}
                for i, agent in enumerate(self.agents):
                    action = agent.action(obs_n[i], add_noise=False)
                    actions_dict[agent.name] = torch.FloatTensor(action).unsqueeze(0).repeat(
                        self.env.num_envs, 1
                    ).to(self.device)
                
                # Environment step
                new_obs_dict, rew_dict, done_dict, truncated_dict, info_dict = self.env.step(actions_dict)
                
                # Extract rewards
                rew_n = [rew_dict[agent.name][0].cpu().numpy() for agent in self.agents]
                done_n = [done_dict[agent.name][0].cpu().numpy() for agent in self.agents]
                
                # Update rewards
                for i, rew in enumerate(rew_n):
                    episode_reward += rew
                    agent_episode_rewards[i] += rew
                
                # Update observations
                obs_n = [new_obs_dict[agent.name][0].cpu().numpy() for agent in self.agents]
                
                episode_step += 1
                
                # Check termination
                if any(done_n) or any(truncated_dict[agent.name][0].cpu().numpy() for agent in self.agents):
                    break
                
                if render:
                    self.env.render()
            
            # Record results
            episode_rewards.append(episode_reward)
            for i in range(self.num_agents):
                agent_rewards[i].append(agent_episode_rewards[i])
            
            # Check success criteria (task completion)
            if hasattr(self.env, 'task_completed') and self.env.task_completed[0].cpu().item():
                success_count += 1
            
            # Check collision
            if hasattr(self.env, '_scalpel'):
                scalpel_pos = self.env._scalpel.data.root_pos_w[0]
                collision = self.env._check_collision_with_constraint(scalpel_pos.unsqueeze(0))[0]
                if collision:
                    collision_count += 1
            
            print(f"  Episode {episode + 1}: Total={episode_reward:.2f}, "
                  f"Human={agent_episode_rewards[0]:.2f}, Robot={agent_episode_rewards[1]:.2f}")
        
        # Calculate metrics
        results = {
            "mean_episode_reward": np.mean(episode_rewards),
            "std_episode_reward": np.std(episode_rewards),
            "success_rate": success_count / num_episodes,
            "collision_rate": collision_count / num_episodes,
        }
        
        for i, agent in enumerate(self.agents):
            results[f"{agent.name}_mean_reward"] = np.mean(agent_rewards[i])
            results[f"{agent.name}_std_reward"] = np.std(agent_rewards[i])
        
        print(f"\n[INFO] Evaluation Results:")
        print(f"  Mean Episode Reward: {results['mean_episode_reward']:.3f} ± {results['std_episode_reward']:.3f}")
        print(f"  Success Rate: {results['success_rate']:.1%}")
        print(f"  Collision Rate: {results['collision_rate']:.1%}")
        for agent in self.agents:
            print(f"  {agent.name.capitalize()} Reward: {results[f'{agent.name}_mean_reward']:.3f}")
        
        return results
    
    def close(self):
        """Close the environment."""
        self.env.close()