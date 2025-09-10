import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any
from .networks import Actor, Critic

class DDPGAgent:
    """
    Deep Deterministic Policy Gradient agent for multi-agent environments.
    
    Features:
    - Stochastic actor with mean and variance outputs
    - Centralized critic for multi-agent coordination
    - Soft target network updates
    - Force constraint compliance
    - Huber loss for improved critic stability
    - Actor debug information collection for console display
    """
    
    def __init__(self, agent_id: str, state_dim: int, action_dim: int, 
                 total_state_dim: int, total_action_dim: int, params: Dict[str, Any], device: torch.device):
        """
        Initialize DDPG agent.
        
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
        self.lr_actor = float(maddpg_cfg.get('lr_actor', 0.01))
        self.lr_critic = float(maddpg_cfg.get('lr_critic', 0.01))
        self.tau = float(maddpg_cfg.get('tau', 0.01))
        hidden_dim = int(maddpg_cfg.get('hidden_units', 64))
        
        # Get agent-specific force constraints
        constraints = params.get('constraints', {})
        if 'robot' in agent_id.lower():
            max_action = constraints.get('max_robot_force', 0.02)
        else:
            max_action = constraints.get('max_human_force', 0.02)
        
        print(f"[INFO] Initializing DDPG Agent: {agent_id}")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  Max action magnitude: {max_action}")
        print(f"  Hidden units: {hidden_dim}")
        print(f"  Learning rates - Actor: {self.lr_actor}, Critic: {self.lr_critic}")
        print(f"  Target update rate (tau): {self.tau}")
        
        # Initialize actor networks
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action_magnitude=max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action_magnitude=max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Initialize critic networks (centralized training)
        self.critic = Critic(total_state_dim, total_action_dim, hidden_dim).to(device)
        self.critic_target = Critic(total_state_dim, total_action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        print(f"[INFO] DDPG Agent {agent_id} initialized successfully")

    def select_action(self, observation: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select action using stochastic policy (legacy interface).
        
        Args:
            observation: Agent's observation as numpy array
            add_noise: Whether to add exploration noise
            
        Returns:
            Action as numpy array
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std = self.actor(obs_tensor)
        
        action = mean
        if add_noise:
            noise = std * torch.randn_like(mean)
            action += noise
            
        return action.cpu().numpy().flatten()
    
    def select_action_with_debug(self, observation: np.ndarray, add_noise: bool = True) -> Dict[str, np.ndarray]:
        """
        Select action using stochastic policy and return detailed debug information.
        
        This method provides the same functionality as select_action but additionally
        returns the mean action (deterministic policy output) and noise separately
        for debugging and console display purposes.
        
        Args:
            observation: Agent's observation as numpy array
            add_noise: Whether to add exploration noise
            
        Returns:
            Dictionary containing:
                - 'action': Final action (mean + noise)
                - 'mean': Deterministic policy output (no noise)
                - 'noise': Exploration noise that was added
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get stochastic policy outputs
            mean, std = self.actor(obs_tensor)
            
            # Calculate noise and final action
            if add_noise:
                noise = std * torch.randn_like(mean)
                action = mean + noise
            else:
                noise = torch.zeros_like(mean)
                action = mean.clone()
        
        # Return debug information for console display
        return {
            'action': action.cpu().numpy().flatten(),
            'mean': mean.cpu().numpy().flatten(),
            'noise': noise.cpu().numpy().flatten()
        }
    
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
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        Get information about network architectures and parameters.
        
        Returns:
            Dictionary containing network information
        """
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'agent_id': self.agent_id,
            'actor_params': count_parameters(self.actor),
            'critic_params': count_parameters(self.critic),
            'total_params': count_parameters(self.actor) + count_parameters(self.critic),
            'learning_rates': {
                'actor': self.lr_actor,
                'critic': self.lr_critic
            },
            'tau': self.tau,
            'device': str(self.device)
        }
    
    def save_networks(self, filepath: str) -> None:
        """
        Save all network states to file.
        
        Args:
            filepath: Path to save the networks
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'agent_id': self.agent_id,
            'hyperparameters': {
                'lr_actor': self.lr_actor,
                'lr_critic': self.lr_critic,
                'tau': self.tau
            }
        }, filepath)
        print(f"[INFO] Networks saved for agent {self.agent_id}: {filepath}")
    
    def load_networks(self, filepath: str) -> None:
        """
        Load all network states from file.
        
        Args:
            filepath: Path to load the networks from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load network states
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        
        # Load optimizer states
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        print(f"[INFO] Networks loaded for agent {self.agent_id}: {filepath}")
    
    def set_train_mode(self) -> None:
        """Set all networks to training mode."""
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()
    
    def set_eval_mode(self) -> None:
        """Set all networks to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"DDPGAgent(id='{self.agent_id}', device='{self.device}')"