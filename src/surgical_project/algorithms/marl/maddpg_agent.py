# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PyTorch MADDPG Agent Implementation"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional

from .replay_buffer import ReplayBuffer


class Actor(nn.Module):
    """Actor network for MADDPG agent."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Actions bounded to [-1, 1]
        return x


class Critic(nn.Module):
    """Critic network for MADDPG agent."""
    
    def __init__(self, obs_dim: int, action_dim: int, num_agents: int, 
                 hidden_dim: int = 64, local_q_func: bool = False):
        super(Critic, self).__init__()
        self.local_q_func = local_q_func
        
        if local_q_func:
            # Local Q-function: only use own observation and action
            input_dim = obs_dim + action_dim
        else:
            # Global Q-function: use all agents' observations and actions
            input_dim = obs_dim * num_agents + action_dim * num_agents
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if self.local_q_func:
            # Use only local observation and action
            x = torch.cat([obs, actions], dim=1)
        else:
            # Use all observations and actions
            x = torch.cat([obs, actions], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class MADDPGAgent:
    """MADDPG Agent implementation in PyTorch."""
    
    def __init__(self, name: str, obs_dim: int, action_dim: int, num_agents: int,
                 agent_index: int, lr: float = 1e-2, hidden_dim: int = 64,
                 local_q_func: bool = False, device: str = "cuda"):
        """Initialize MADDPG agent.
        
        Args:
            name: Agent name
            obs_dim: Observation dimension
            action_dim: Action dimension  
            num_agents: Total number of agents
            agent_index: Index of this agent
            lr: Learning rate
            hidden_dim: Hidden layer dimension
            local_q_func: Whether to use local Q-function
            device: Device to run on
        """
        self.name = name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.agent_index = agent_index
        self.device = device
        self.local_q_func = local_q_func
        
        # Create networks
        self.actor = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(obs_dim, action_dim, num_agents, hidden_dim, local_q_func).to(device)
        
        # Create target networks
        self.target_actor = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.target_critic = Critic(obs_dim, action_dim, num_agents, hidden_dim, local_q_func).to(device)
        
        # Initialize target networks to match main networks
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)
        
        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(int(1e6))
        
        # Training parameters
        self.gamma = 0.95
        self.tau = 0.01  # Soft update parameter
        self.batch_size = 1024
        self.max_replay_buffer_len = self.batch_size * 25  # From original code
        
        # Exploration noise
        self.exploration_noise = 0.1
        
        print(f"[INFO] MADDPG Agent '{name}' initialized")
        print(f"  - Observation dim: {obs_dim}")
        print(f"  - Action dim: {action_dim}")
        print(f"  - Local Q-function: {local_q_func}")
        print(f"  - Device: {device}")
    
    def hard_update(self, target: nn.Module, source: nn.Module):
        """Hard update: copy parameters from source to target."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update: slowly update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def action(self, obs: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Get action for given observation.
        
        Args:
            obs: Observation
            add_noise: Whether to add exploration noise
            
        Returns:
            Action array
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]
        
        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = action + noise
            
        return np.clip(action, -1.0, 1.0)
    
    def target_action(self, obs: np.ndarray) -> np.ndarray:
        """Get target action for given observation."""
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        
        with torch.no_grad():
            action = self.target_actor(obs_tensor).cpu().numpy()
            
        return action
    
    def experience(self, obs: np.ndarray, action: np.ndarray, reward: float,
                  new_obs: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.add(obs, action, reward, new_obs, done)
    
    def preupdate(self):
        """Prepare for update (placeholder for consistency with original code)."""
        pass
    
    def update(self, agents: List['MADDPGAgent'], step: int) -> Optional[List[float]]:
        """Update agent networks.
        
        Args:
            agents: List of all agents
            step: Current training step
            
        Returns:
            List of losses [q_loss, p_loss, mean_target_q, mean_reward, mean_target_q_next, std_target_q]
        """
        # Only update every 100 steps and if buffer is large enough
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            return None
        if step % 100 != 0:
            return None
        
        # Sample from replay buffer
        sample_indices = self.replay_buffer.make_index(self.batch_size)
        
        # Collect replay samples from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        
        for i, agent in enumerate(agents):
            obs, act, rew, obs_next, done = agent.replay_buffer.sample_index(sample_indices)
            obs_n.append(torch.FloatTensor(obs).to(self.device))
            obs_next_n.append(torch.FloatTensor(obs_next).to(self.device))
            act_n.append(torch.FloatTensor(act).to(self.device))
        
        # Get own experience
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(sample_indices)
        reward = torch.FloatTensor(rew).to(self.device)
        done_mask = torch.FloatTensor(1 - done).to(self.device)
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        action_tensor = torch.FloatTensor(act).to(self.device)
        obs_next_tensor = torch.FloatTensor(obs_next).to(self.device)
        
        # ===== Update Critic =====
        # Get target actions for next states
        target_actions_next = []
        for i, agent in enumerate(agents):
            target_action = agent.target_action(obs_next_n[i].cpu().numpy())
            target_actions_next.append(torch.FloatTensor(target_action).to(self.device))
        
        # Prepare input for critic
        if self.local_q_func:
            # Local Q-function: use only own observation and action
            obs_critic = obs_tensor
            act_critic = action_tensor
            obs_next_critic = obs_next_tensor
            act_next_critic = target_actions_next[self.agent_index]
        else:
            # Global Q-function: use all agents' observations and actions
            obs_critic = torch.cat(obs_n, dim=1)
            act_critic = torch.cat(act_n, dim=1)
            obs_next_critic = torch.cat(obs_next_n, dim=1)
            act_next_critic = torch.cat(target_actions_next, dim=1)
        
        # Calculate target Q-value
        with torch.no_grad():
            target_q_next = self.target_critic(obs_next_critic, act_next_critic).squeeze()
            target_q = reward + self.gamma * done_mask * target_q_next
        
        # Current Q-value
        current_q = self.critic(obs_critic, act_critic).squeeze()
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # ===== Update Actor =====
        # Get current actions
        if self.local_q_func:
            policy_actions = self.actor(obs_tensor)
            actor_loss = -self.critic(obs_tensor, policy_actions).mean()
        else:
            # For global Q-function, need to construct full action vector
            policy_actions_n = []
            for i, agent in enumerate(agents):
                if i == self.agent_index:
                    policy_actions_n.append(self.actor(obs_n[i]))
                else:
                    # Use the actions from replay buffer for other agents
                    policy_actions_n.append(act_n[i])
            
            policy_actions_full = torch.cat(policy_actions_n, dim=1)
            actor_loss = -self.critic(obs_critic, policy_actions_full).mean()
        
        # Actor regularization
        if not self.local_q_func:
            policy_actions = self.actor(obs_tensor)
        actor_reg = torch.mean(torch.square(policy_actions))
        actor_loss = actor_loss + actor_reg * 1e-3
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)
        
        # Return losses for logging
        return [
            critic_loss.item(),
            actor_loss.item(), 
            target_q.mean().item(),
            reward.mean().item(),
            target_q_next.mean().item(),
            target_q.std().item()
        ]
    
    def save(self, filepath: str):
        """Save agent networks."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent networks."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])