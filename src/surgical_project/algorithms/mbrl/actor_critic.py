"""Surgical Robot Actor-Critic with Dynamics Identifier Network"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SurgicalActor(nn.Module):
    """Surgical Robot Actor Network"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        # Simple MLP network for actual observation dimension
        self.network = nn.ModuleList()
        dims = [obs_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.network.append(nn.Linear(dims[i], dims[i+1]))
            
        # Output layers
        self.action_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = obs
        for layer in self.network:
            x = F.relu(layer(x))
            
        # Generate action distribution parameters
        action_mean = torch.tanh(self.action_head(x))  # Bounded actions
        action_std = torch.exp(self.log_std.clamp(-20, 2))
        
        return action_mean, action_std

class SurgicalCritic(nn.Module):
    """Surgical Robot Critic Network"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        # Q-critic network
        self.q_layers = nn.ModuleList()
        dims = [obs_dim + action_dim] + hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            self.q_layers.append(nn.Linear(dims[i], dims[i+1]))
            
        # V-critic network
        self.v_layers = nn.ModuleList()
        dims = [obs_dim] + hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            self.v_layers.append(nn.Linear(dims[i], dims[i+1]))
    
    def forward_q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        for i, layer in enumerate(self.q_layers[:-1]):
            x = F.relu(layer(x))
        q_value = self.q_layers[-1](x)
        return q_value
    
    def forward_v(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        for i, layer in enumerate(self.v_layers[:-1]):
            x = F.relu(layer(x))
        v_value = self.v_layers[-1](x)
        return v_value

class DynamicsIdentifierNetwork(nn.Module):
    """Dynamics Identifier Network - Equations (34–36) from the paper"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Identifier network: input (z, u) -> output z˙  
        input_dim = state_dim + action_dim
        self.network = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [state_dim]
        
        for i in range(len(dims) - 1):
            self.network.append(nn.Linear(dims[i], dims[i+1]))
            
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Equation (34) in the paper: z˙ = W^T_id S_id(z, u) + ε_id"""
        x = torch.cat([state, action], dim=-1)
        
        for i, layer in enumerate(self.network[:-1]):
            x = F.relu(layer(x))
            
        # Output state derivative
        state_dot = self.network[-1](x)
        return state_dot
    
    def predict_next_state(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """Predict next state: z_{k+1} = z_k + dt * z˙"""
        state_dot = self.forward(state, action)
        next_state = state + dt * state_dot
        return next_state

class SurgicalActorCritic(nn.Module):
    """Complete Actor-Critic Network with Dynamics Identifier"""
    def __init__(self, obs_dim: int, action_dim: int, state_dim: int = None):
        super().__init__()
        
        # If state_dim is not specified, assume it equals obs_dim
        if state_dim is None:
            state_dim = obs_dim
            
        self.actor = SurgicalActor(obs_dim, action_dim)
        self.critic = SurgicalCritic(obs_dim, action_dim)
        self.identifier = DynamicsIdentifierNetwork(state_dim, action_dim)
        
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        mean, std = self.actor(obs)
        if deterministic:
            return mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            return torch.clamp(action, -1.0, 1.0)
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        mean, std = self.actor(obs)
        dist = torch.distributions.Normal(mean, std)
        
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        q_value = self.critic.forward_q(obs, actions)
        v_value = self.critic.forward_v(obs)
        
        return log_prob, entropy, q_value, v_value
    
    def identify_dynamics(self, state: torch.Tensor, action: torch.Tensor):
        """Identify system dynamics"""
        return self.identifier(state, action)
