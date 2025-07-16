"""Simplified Surgical Robot Actor-Critic with Dynamics Identifier Network"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SurgicalActor(nn.Module):
    """Simplified Surgical Robot Actor Network"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list = [256, 128]):
        super().__init__()
        
        # Build network layers
        layers = []
        dims = [obs_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        
        self.backbone = nn.Sequential(*layers)
        
        # Output layers
        self.action_mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(obs)
        
        # Generate action distribution parameters
        action_mean = torch.tanh(self.action_mean(x))  # Bounded actions [-1, 1]
        action_std = torch.exp(self.log_std.clamp(-20, 2))
        
        return action_mean, action_std

class SurgicalCritic(nn.Module):
    """Simplified Surgical Robot Critic Network"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list = [256, 128]):
        super().__init__()
        
        # Q-critic network (state-action value)
        q_dims = [obs_dim + action_dim] + hidden_dims + [1]
        q_layers = []
        for i in range(len(q_dims) - 1):
            q_layers.append(nn.Linear(q_dims[i], q_dims[i+1]))
            if i < len(q_dims) - 2:  # No activation on output layer
                q_layers.append(nn.ReLU())
        self.q_network = nn.Sequential(*q_layers)
        
        # V-critic network (state value)
        v_dims = [obs_dim] + hidden_dims + [1]
        v_layers = []
        for i in range(len(v_dims) - 1):
            v_layers.append(nn.Linear(v_dims[i], v_dims[i+1]))
            if i < len(v_dims) - 2:  # No activation on output layer
                v_layers.append(nn.ReLU())
        self.v_network = nn.Sequential(*v_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)
    
    def forward_q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.q_network(x)
    
    def forward_v(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v_network(obs)

class DynamicsIdentifierNetwork(nn.Module):
    """Simplified Dynamics Identifier Network - Based on paper Equations (34–36)"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Identifier network: input (z, u) -> output z˙  
        input_dim = state_dim + action_dim
        dims = [input_dim] + hidden_dims + [state_dim]
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation on output layer
                layers.append(nn.ReLU())
                
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.1)  # Smaller gain for dynamics
            nn.init.constant_(m.bias, 0.0)
            
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Equation (34) in the paper: z˙ = W^T_id S_id(z, u) + ε_id"""
        x = torch.cat([state, action], dim=-1)
        state_dot = self.network(x)
        return state_dot
    
    def predict_next_state(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """Predict next state: z_{k+1} = z_k + dt * z˙"""
        state_dot = self.forward(state, action)
        next_state = state + dt * state_dot
        return next_state

class SurgicalActorCritic(nn.Module):
    """Complete Simplified Actor-Critic Network with Dynamics Identifier"""
    def __init__(self, obs_dim: int, action_dim: int, state_dim: int = None):
        super().__init__()
        
        # If state_dim is not specified, assume it equals obs_dim
        if state_dim is None:
            state_dim = obs_dim
            
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
            
        self.actor = SurgicalActor(obs_dim, action_dim)
        self.critic = SurgicalCritic(obs_dim, action_dim)
        self.identifier = DynamicsIdentifierNetwork(state_dim, action_dim)
        
        print(f"[INFO] SurgicalActorCritic initialized:")
        print(f"  - Obs dim: {obs_dim}, Action dim: {action_dim}, State dim: {state_dim}")
        print(f"  - Actor params: {sum(p.numel() for p in self.actor.parameters()):,}")
        print(f"  - Critic params: {sum(p.numel() for p in self.critic.parameters()):,}")
        print(f"  - Identifier params: {sum(p.numel() for p in self.identifier.parameters()):,}")
        
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Get action from actor network"""
        mean, std = self.actor(obs)
        if deterministic:
            return mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            return torch.clamp(action, -1.0, 1.0)
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for training"""
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