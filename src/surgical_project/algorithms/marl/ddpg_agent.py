import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any
from .networks import Actor, Critic

class DDPGAgent:
    def __init__(self, agent_id: str, state_dim: int, action_dim: int, 
                 total_state_dim: int, total_action_dim: int, params: Dict[str, Any], device: torch.device):
        self.agent_id = agent_id
        self.device = device
        
        maddpg_cfg = params.get('maddpg_config', {})
        self.lr_actor = float(maddpg_cfg.get('lr_actor', 0.01))
        self.lr_critic = float(maddpg_cfg.get('lr_critic', 0.01))
        self.tau = float(maddpg_cfg.get('tau', 0.01))
        hidden_dim = int(maddpg_cfg.get('num_units', 64))
        
        # 从环境约束获取合理的最大action
        constraints = params.get('constraints', {})
        if 'robot' in agent_id.lower():
            max_action = constraints.get('max_robot_force', 0.02)
        else:
            max_action = constraints.get('max_human_force', 0.02)
        
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action_magnitude=max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action_magnitude=max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(total_state_dim, total_action_dim, hidden_dim).to(device)
        self.critic_target = Critic(total_state_dim, total_action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def select_action(self, observation: np.ndarray, add_noise: bool = True) -> np.ndarray:
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std = self.actor(obs_tensor)
        
        action = mean
        if add_noise:
            action += std * torch.randn_like(mean)
            
        return action.cpu().numpy().flatten()
    
    def update_actor(self, loss: torch.Tensor) -> Dict[str, float]:
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        return {'actor_loss': loss.item()}

    def update_critic(self, states: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_values, targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        return {'critic_loss': critic_loss.item()}

    def soft_update(self) -> None:
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_state(self) -> Dict[str, Any]:
        """Saves network and optimizer states."""
        return {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }
    
    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Loads network and optimizer states."""
        self.actor.load_state_dict(state_dict['actor_state_dict'])
        self.critic.load_state_dict(state_dict['critic_state_dict'])
        self.actor_target.load_state_dict(state_dict['actor_target_state_dict'])
        self.critic_target.load_state_dict(state_dict['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer_state_dict'])