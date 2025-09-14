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
    
    FINAL VERSION: Unified action selection interface supporting batch processing.
    Features:
    - Stochastic actor with mean and variance outputs
    - Centralized critic for multi-agent coordination  
    - Soft target network updates
    - Force constraint compliance
    - Batch processing support for shared network architecture
    - Gradient norm statistics for training diagnostics
    - Configurable network architecture via YAML
    """
    
    def __init__(self, agent_id: str, state_dim: int, action_dim: int, 
                 total_state_dim: int, total_action_dim: int, params: Dict[str, Any], device: torch.device):
        """
        Initialize DDPG agent for shared network architecture.
        
        Args:
            agent_id: Unique identifier for this agent
            state_dim: Dimension of this agent's observation space
            action_dim: Dimension of this agent's action space
            total_state_dim: Combined observation dimension of all agents
            total_action_dim: Combined action dimension of all agents
            params: Configuration parameters dictionary
            device: PyTorch device for computations
        """
        self.agent_id = agent_id
        self.device = device
        
        # Load hyperparameters
        maddpg_cfg = params.get('maddpg_config', {})
        self.lr_actor = float(maddpg_cfg.get('lr_actor', 0.001))
        self.lr_critic = float(maddpg_cfg.get('lr_critic', 0.001))
        self.tau = float(maddpg_cfg.get('tau', 0.002))
        hidden_dim = int(maddpg_cfg.get('hidden_units', 512))
        
        # Get agent-specific force constraints
        constraints = params.get('constraints', {})
        if 'robot' in agent_id.lower():
            max_action = constraints.get('max_robot_force', 0.04)
        else:
            max_action = constraints.get('max_human_force', 0.04)
            
        self.max_action = max_action
        
        # Load network configuration
        net_cfg = params.get('networks', {})
        actor_cfg = net_cfg.get('actor', {})
        critic_cfg = net_cfg.get('critic', {})

        hidden_dim = int(params.get('maddpg_config', {}).get('hidden_units', 512))  # 兼容旧字段

        actor_hidden_layers = actor_cfg.get('hidden_layers', [hidden_dim, hidden_dim])
        actor_dropout = float(actor_cfg.get('dropout_p', 0.0))
        actor_ortho = bool(actor_cfg.get('orthogonal_init', False))
        actor_gain_h = float(actor_cfg.get('ortho_gain_hidden', 1.0))
        actor_gain_o = float(actor_cfg.get('ortho_gain_output', 0.01))
        actor_std_scale = float(actor_cfg.get('std_scale', 1.0))  # 新

        critic_hidden_layers = critic_cfg.get('hidden_layers', [hidden_dim, hidden_dim])
        critic_dropout = float(critic_cfg.get('dropout_p', 0.0))
        critic_ortho = bool(critic_cfg.get('orthogonal_init', False))
        critic_gain_h = float(critic_cfg.get('ortho_gain_hidden', 1.0))
        critic_gain_o = float(critic_cfg.get('ortho_gain_output', 1.0))
        
        print(f"[DDPG AGENT] {agent_id}:")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  Total state dim: {total_state_dim}, Total action dim: {total_action_dim}")
        print(f"  Max action: {max_action}")
        print(f"  Actor layers: {actor_hidden_layers}, dropout: {actor_dropout}")
        print(f"  Critic layers: {critic_hidden_layers}, dropout: {critic_dropout}")
        print(f"  Orthogonal init - Actor: {actor_ortho}, Critic: {critic_ortho}")
        print(f"  Std scale: {actor_std_scale}")
        print(f"  LR - Actor: {self.lr_actor}, Critic: {self.lr_critic}")
        print(f"  Target update rate (tau): {self.tau}")
        
        # Initialize actor networks
        self.actor = Actor(
            state_dim, action_dim,
            hidden_dimension=hidden_dim,
            max_action_magnitude=max_action,
            hidden_layers=actor_hidden_layers,
            dropout_p=actor_dropout,
            orthogonal_init=actor_ortho,
            ortho_gain_hidden=actor_gain_h,
            ortho_gain_output=actor_gain_o,
            std_scale=actor_std_scale
        ).to(device)

        self.actor_target = Actor(
            state_dim, action_dim,
            hidden_dimension=hidden_dim,
            max_action_magnitude=max_action,
            hidden_layers=actor_hidden_layers,
            dropout_p=actor_dropout,
            orthogonal_init=actor_ortho,
            ortho_gain_hidden=actor_gain_h,
            ortho_gain_output=actor_gain_o,
            std_scale=actor_std_scale
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Initialize critic networks (centralized training)
        self.critic = Critic(
            total_state_dim, total_action_dim,
            hidden_dimension=hidden_dim,
            hidden_layers=critic_hidden_layers,
            dropout_p=critic_dropout,
            orthogonal_init=critic_ortho,
            ortho_gain_hidden=critic_gain_h,
            ortho_gain_output=critic_gain_o
        ).to(device)

        self.critic_target = Critic(
            total_state_dim, total_action_dim,
            hidden_dimension=hidden_dim,
            hidden_layers=critic_hidden_layers,
            dropout_p=critic_dropout,
            orthogonal_init=critic_ortho,
            ortho_gain_hidden=critic_gain_h,
            ortho_gain_output=critic_gain_o
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        print(f"[DDPG AGENT] {agent_id} initialized successfully")

    def update_actor(self, loss: torch.Tensor) -> Dict[str, float]:
        """
        Update actor network using provided loss.
        
        Args:
            loss: Actor loss tensor (should be scalar)
            
        Returns:
            Dictionary containing training statistics
        """
        # Zero gradients
        self.actor_optimizer.zero_grad()
        
        # Backpropagate loss
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        
        # Update parameters
        self.actor_optimizer.step()
        
        return {
            'actor_loss': loss.item()
        }

    def update_critic(self, states: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Update critic network using Huber loss.
        
        Args:
            states: Concatenated states from all agents
            actions: Concatenated actions from all agents
            targets: Target Q-values
            
        Returns:
            Dictionary containing training statistics
        """
        # Forward pass through critic
        q_values = self.critic(states, actions)
        
        # Calculate Huber loss (smooth L1) for better robustness to outliers
        critic_loss = F.smooth_l1_loss(q_values, targets)
        
        # Zero gradients
        self.critic_optimizer.zero_grad()
        
        # Backpropagate loss
        critic_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        
        # Update parameters
        self.critic_optimizer.step()
        
        return {
            'critic_loss': critic_loss.item(),
            'q_mean': q_values.mean().item(),
            'q_std': q_values.std().item(),
            'target_mean': targets.mean().item(),
            'target_std': targets.std().item()
        }

    def soft_update(self) -> None:
        """
        Perform soft update of target networks using Polyak averaging.
        
        Updates both actor and critic target networks using the formula:
        target_param = tau * param + (1 - tau) * target_param
        """
        # Update actor target network
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        
        # Update critic target network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )