"""
DDPG agent with residual connections and state normalization for shared MADDPG.
Features deterministic actor, centralized critic, and configurable network architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any
from .networks import Actor, Critic

class DDPGAgent:
    """
    Deep Deterministic Policy Gradient agent for shared network MADDPG.
    
    Features:
    - Deterministic actor with normalized actions [-1,1]
    - Centralized critic expecting normalized action inputs
    - Fixed state scaling injected into networks
    - Soft target network updates
    - Batch processing for shared network architecture
    - Gradient norm statistics for training diagnostics
    - YAML-configurable network architecture
    - Residual connections for improved gradient flow
    """
    
    def __init__(self, agent_id: str, state_dim: int, action_dim: int, 
                 total_state_dim: int, total_action_dim: int, params: Dict[str, Any], device: torch.device):
        """Initialize DDPG Agent for shared network architecture."""
        self.agent_id = agent_id
        self.device = device
        
        # Load hyperparameters
        maddpg_cfg = params.get('maddpg_config', {})
        self.lr_actor = float(maddpg_cfg.get('lr_actor', 0.001))
        self.lr_critic = float(maddpg_cfg.get('lr_critic', 0.001))
        self.tau = float(maddpg_cfg.get('tau', 0.002))
        
        # Load network configuration
        net_cfg = params.get('networks', {})
        actor_cfg = net_cfg.get('actor', {})
        critic_cfg = net_cfg.get('critic', {})

        # Network architecture parameters - removed dropout, added residual connections
        actor_hidden_layers = actor_cfg.get('hidden_layers', [256, 256])
        actor_bypass_layers = actor_cfg.get('input_bypass_layers', [])  # Residual connections
        actor_ortho = bool(actor_cfg.get('orthogonal_init', False))
        actor_gain_h = float(actor_cfg.get('ortho_gain_hidden', 1.0))
        actor_gain_o = float(actor_cfg.get('ortho_gain_output', 0.01))

        critic_hidden_layers = critic_cfg.get('hidden_layers', [256, 256])
        critic_bypass_layers = critic_cfg.get('input_bypass_layers', [])  # Residual connections
        critic_ortho = bool(critic_cfg.get('orthogonal_init', False))
        critic_gain_h = float(critic_cfg.get('ortho_gain_hidden', 1.0))
        critic_gain_o = float(critic_cfg.get('ortho_gain_output', 1.0))
        
        # Read state scaling configuration
        obs_scaling = params.get('obs_scaling', {})
        factors = obs_scaling.get('factors', [1.0] * state_dim)
        
        if len(factors) != state_dim:
            print(f"[WARNING] {agent_id} - Factors length mismatch: got {len(factors)}, expected {state_dim}")
            factors = [1.0] * state_dim
        
        # Actor uses single agent scaling factors
        obs_factors_actor = torch.tensor(factors, dtype=torch.float32, device=device)
        
        # Critic uses all agents' scaling factors (repeat single agent factors)
        num_agents = total_state_dim // state_dim
        factors_critic = factors * num_agents
        obs_factors_critic = torch.tensor(factors_critic, dtype=torch.float32, device=device)
        
        print(f"[DDPG AGENT] {agent_id}:")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  Total state dim: {total_state_dim}, Total action dim: {total_action_dim}")
        print(f"  Actor layers: {actor_hidden_layers}, bypass: {actor_bypass_layers}")
        print(f"  Critic layers: {critic_hidden_layers}, bypass: {critic_bypass_layers}")
        print(f"  Orthogonal init - Actor: {actor_ortho}, Critic: {critic_ortho}")
        print(f"  State scaling factors: {factors}")
        print(f"  LR - Actor: {self.lr_actor}, Critic: {self.lr_critic}")
        print(f"  Target update rate (tau): {self.tau}")
        
        # Initialize Actor network (with state scaling and residual connections)
        self.actor = Actor(
            state_dim, action_dim,
            hidden_layers=actor_hidden_layers,
            input_bypass_layers=actor_bypass_layers,  # Residual connections
            orthogonal_init=actor_ortho,
            ortho_gain_hidden=actor_gain_h,
            ortho_gain_output=actor_gain_o,
            obs_factors=obs_factors_actor,
        ).to(device)

        self.actor_target = Actor(
            state_dim, action_dim,
            hidden_layers=actor_hidden_layers,
            input_bypass_layers=actor_bypass_layers,  # Residual connections
            orthogonal_init=actor_ortho,
            ortho_gain_hidden=actor_gain_h,
            ortho_gain_output=actor_gain_o,
            obs_factors=obs_factors_actor,
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Initialize Critic network (centralized training, with state scaling and residual connections)
        self.critic = Critic(
            total_state_dim, total_action_dim,
            hidden_layers=critic_hidden_layers,
            input_bypass_layers=critic_bypass_layers,  # Residual connections
            orthogonal_init=critic_ortho,
            ortho_gain_hidden=critic_gain_h,
            ortho_gain_output=critic_gain_o,
            obs_factors=obs_factors_critic,
        ).to(device)

        self.critic_target = Critic(
            total_state_dim, total_action_dim,
            hidden_layers=critic_hidden_layers,
            input_bypass_layers=critic_bypass_layers,  # Residual connections
            orthogonal_init=critic_ortho,
            ortho_gain_hidden=critic_gain_h,
            ortho_gain_output=critic_gain_o,
            obs_factors=obs_factors_critic,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        print(f"[DDPG AGENT] {agent_id} initialized with residual connections and state normalization")

    def soft_update(self) -> None:
        """Soft update target networks using Polyak averaging."""
        # Update Actor target network
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        
        # Update Critic target network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )