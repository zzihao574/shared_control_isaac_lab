"""论文对齐的手术机器人Actor-Critic网络"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SurgicalCritic(nn.Module):
    """论文方程(29)的Critic网络: Γ = W_c^T S_c(z̄) + ε_c"""
    def __init__(self, augmented_state_dim: int, hidden_dims: list = [256, 128]):
        super().__init__()
        
        layers = []
        dims = [augmented_state_dim] + hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        self.value_network = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)
        
    def forward(self, augmented_state: torch.Tensor) -> torch.Tensor:
        if augmented_state.dim() == 1:
            augmented_state = augmented_state.unsqueeze(0)
        
        return self.value_network(augmented_state)

class SurgicalActor(nn.Module):
    """论文方程(50)的Actor网络: u = Ŵ_a^T S_a(Z_a)"""
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: list = [256, 128]):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [action_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)
            
    def forward(self, input_state: torch.Tensor) -> torch.Tensor:
        if input_state.dim() == 1:
            input_state = input_state.unsqueeze(0)
        
        network_output = self.network(input_state)
        return torch.tanh(network_output)

class DynamicsIdentifierNetwork(nn.Module):
    """论文方程(34)的动力学识别网络: ż = W_id^T S_id(z, u) + ε_id"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        super().__init__()
        
        layers = []
        input_dim = state_dim + action_dim
        dims = [input_dim] + hidden_dims + [state_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.1)
            nn.init.constant_(m.bias, 0.0)
            
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

class SurgicalActorCritic(nn.Module):
    """论文Actor-Critic网络，使用神经网络近似求解HJB方程"""
    def __init__(self, state_dim: int, action_dim: int, augmented_state_dim: int, 
                 Q_weights: torch.Tensor = None, R_weights: torch.Tensor = None):
        super().__init__()
        
        self.state_dim = state_dim  # z ∈ R^9
        self.action_dim = action_dim  # u ∈ R^3
        self.augmented_state_dim = augmented_state_dim  # z̄ ∈ R^12
        
        self.critic = SurgicalCritic(augmented_state_dim)
        self.actor = SurgicalActor(augmented_state_dim, action_dim)
        self.identifier = DynamicsIdentifierNetwork(state_dim, action_dim)
        
    def get_action(self, augmented_state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action = self.actor(augmented_state)
        
        if not deterministic:
            noise = torch.randn_like(action) * 0.01
            action = action + noise
                
        return torch.clamp(action, -1.0, 1.0)
    
    def evaluate_value(self, augmented_state: torch.Tensor) -> torch.Tensor:
        return self.critic(augmented_state)
    
    def compute_actor_loss(self, augmented_state: torch.Tensor) -> torch.Tensor:
        value = self.critic(augmented_state)
        return -value.mean()
    
    def compute_critic_loss(self, augmented_state: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
        predicted_value = self.critic(augmented_state)
        return F.mse_loss(predicted_value.squeeze(), target_value.squeeze())
    
    def compute_identifier_loss(self, state: torch.Tensor, action: torch.Tensor, 
                              next_state: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        true_state_dot = (next_state - state) / dt
        pred_state_dot = self.identifier(state, action)
        return F.mse_loss(pred_state_dot, true_state_dot)