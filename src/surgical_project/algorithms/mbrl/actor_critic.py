"""Paper-aligned surgical robot RBF Actor-Critic networks - complete version with YAML configuration"""

import torch
import torch.nn as nn
import numpy as np


class RBFLayer(nn.Module):
    """Single-layer RBF network as described in paper"""
    def __init__(self, input_dim: int, num_nodes: int, gaussian_width: float, 
                 mu_range: list = [-1.0, 1.0], mu_step: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.gaussian_width = gaussian_width
        
        # Initialize RBF centers μi according to paper setup
        # μc,ij = μa,ij = μid,ij = -1 + 0.2i
        centers = []
        for i in range(num_nodes):
            center = []
            for j in range(input_dim):
                mu_val = mu_range[0] + mu_step * i
                # Clamp to range
                mu_val = max(mu_range[0], min(mu_range[1], mu_val))
                center.append(mu_val)
            centers.append(center)
        
        self.centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32), requires_grad=False)
        self.width = gaussian_width
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF activations: si(Z) = exp(-(Z-μi)^T(Z-μi)/ηi^2)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(1)  # (batch, 1, input_dim)
        centers_expanded = self.centers.unsqueeze(0)  # (1, num_nodes, input_dim)
        
        # Compute squared distances
        diff = x_expanded - centers_expanded  # (batch, num_nodes, input_dim)
        squared_dist = torch.sum(diff**2, dim=2)  # (batch, num_nodes)
        
        # Apply Gaussian function
        rbf_output = torch.exp(-squared_dist / (self.width**2))
        
        return rbf_output


class SurgicalCritic(nn.Module):
    """Paper equation (29) Critic network: Γ = W_c^T S_c(z̄) + ε_c"""
    def __init__(self, augmented_state_dim: int, network_cfg: dict = None):
        super().__init__()
        
        # Extract RBF configuration
        if network_cfg is None:
            network_cfg = {}
        
        rbf_cfg = network_cfg.get('rbf_network', {})
        critic_cfg = rbf_cfg.get('critic', {})
        
        num_nodes = critic_cfg.get('nodes', 10)
        gaussian_width = critic_cfg.get('gaussian_width', 15.0)
        mu_range = rbf_cfg.get('mu_range', [-1.0, 1.0])
        mu_step = rbf_cfg.get('mu_step', 0.2)
        
        # Single RBF layer
        self.rbf_layer = RBFLayer(
            input_dim=augmented_state_dim,
            num_nodes=num_nodes,
            gaussian_width=gaussian_width,
            mu_range=mu_range,
            mu_step=mu_step
        )
        
        # Output layer (no activation for value function)
        self.output_layer = nn.Linear(num_nodes, 1)
        
        # Initialize output weights to small values
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)
        
    def forward(self, augmented_state: torch.Tensor) -> torch.Tensor:
        if augmented_state.dim() == 1:
            augmented_state = augmented_state.unsqueeze(0)
        
        rbf_features = self.rbf_layer(augmented_state)
        value = self.output_layer(rbf_features)
        
        return value


class SurgicalActor(nn.Module):
    """Paper equation (50) Actor network: u = Ŵ_a^T S_a(Z_a)"""
    def __init__(self, actor_input_dim: int, action_dim: int, network_cfg: dict = None):
        super().__init__()
        
        # Extract RBF configuration
        if network_cfg is None:
            network_cfg = {}
        
        rbf_cfg = network_cfg.get('rbf_network', {})
        actor_cfg = rbf_cfg.get('actor', {})
        
        num_nodes = actor_cfg.get('nodes', 10)
        gaussian_width = actor_cfg.get('gaussian_width', 10.0)
        mu_range = rbf_cfg.get('mu_range', [-1.0, 1.0])
        mu_step = rbf_cfg.get('mu_step', 0.2)
        
        # Single RBF layer
        self.rbf_layer = RBFLayer(
            input_dim=actor_input_dim,
            num_nodes=num_nodes,
            gaussian_width=gaussian_width,
            mu_range=mu_range,
            mu_step=mu_step
        )
        
        # Output layer with tanh activation for bounded control
        self.output_layer = nn.Linear(num_nodes, action_dim)
        
        # Initialize output weights to small values
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)
            
    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        if actor_input.dim() == 1:
            actor_input = actor_input.unsqueeze(0)
        
        rbf_features = self.rbf_layer(actor_input)
        control_output = self.output_layer(rbf_features)
        
        # Apply tanh for bounded output
        return control_output


class DynamicsIdentifierNetwork(nn.Module):
    """Paper equation (34) dynamics identifier network: ż = W_id^T S_id(z, u) + ε_id"""
    def __init__(self, state_dim: int, action_dim: int, network_cfg: dict = None):
        super().__init__()
        
        # Extract RBF configuration
        if network_cfg is None:
            network_cfg = {}
        
        rbf_cfg = network_cfg.get('rbf_network', {})
        identifier_cfg = rbf_cfg.get('identifier', {})
        
        num_nodes = identifier_cfg.get('nodes', 10)
        gaussian_width = identifier_cfg.get('gaussian_width', 100.0)
        mu_range = rbf_cfg.get('mu_range', [-1.0, 1.0])
        mu_step = rbf_cfg.get('mu_step', 0.2)
        
        input_dim = state_dim + action_dim
        
        # Single RBF layer
        self.rbf_layer = RBFLayer(
            input_dim=input_dim,
            num_nodes=num_nodes,
            gaussian_width=gaussian_width,
            mu_range=mu_range,
            mu_step=mu_step
        )
        
        # Output layer (no activation for dynamics prediction)
        self.output_layer = nn.Linear(num_nodes, state_dim)
        
        # Initialize with smaller weights for dynamics identification
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.output_layer.bias, 0.0)
            
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        combined_input = torch.cat([state, action], dim=-1)
        rbf_features = self.rbf_layer(combined_input)
        state_derivative = self.output_layer(rbf_features)
        
        return state_derivative


class SurgicalActorCritic(nn.Module):
    """Paper Actor-Critic network using RBF approximation to solve HJB equation - configuration supported"""
    def __init__(self, state_dim: int, action_dim: int, augmented_state_dim: int, 
                 actor_input_dim: int, network_cfg: dict = None):
        super().__init__()
        
        self.state_dim = state_dim  # z ∈ R^9
        self.action_dim = action_dim  # u ∈ R^3
        self.augmented_state_dim = augmented_state_dim  # z̄ ∈ R^12
        self.actor_input_dim = actor_input_dim  # Za ∈ R^18 = [q, q̇, ẋr, ẍr]
        
        # Pass network configuration to all sub-networks
        self.critic = SurgicalCritic(augmented_state_dim, network_cfg)
        self.actor = SurgicalActor(actor_input_dim, action_dim, network_cfg)
        self.identifier = DynamicsIdentifierNetwork(state_dim, action_dim, network_cfg)
        
        # Extract exploration parameters from configuration
        self.action_noise_std = network_cfg.get('exploration_noise', 0.01) if network_cfg else 0.01
        
    def get_action(self, actor_input: torch.Tensor, deterministic: bool = False, 
                   exploration_noise: float = None) -> torch.Tensor:
        """Get control action from actor network"""
        action = self.actor(actor_input)
        
        if not deterministic:
            # Use configured exploration noise
            noise_std = exploration_noise if exploration_noise is not None else self.action_noise_std
            noise = torch.randn_like(action) * noise_std
            action = action + noise

        return torch.clamp(action, -1.0, 1.0)  # 优化

    def evaluate_value(self, augmented_state: torch.Tensor) -> torch.Tensor:
        """Evaluate value function using critic network"""
        return self.critic(augmented_state)
    
    def compute_actor_loss(self, augmented_state: torch.Tensor) -> torch.Tensor:
        """Compute actor loss for policy gradient"""
        value = self.critic(augmented_state)
        return -value.mean()