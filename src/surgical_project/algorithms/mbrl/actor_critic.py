"""Paper-aligned RBF Actor-Critic networks with PE noise and clear time notation"""

import torch
import torch.nn as nn
import numpy as np


class RBFLayer(nn.Module):
    """RBF layer: si(Z) = exp(-(Z-μi)^T(Z-μi)/ηi^2)"""
    def __init__(self, input_dim: int, num_nodes: int, gaussian_width: float, 
                 mu_range: list = [-1.0, 1.0], mu_step: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.width = gaussian_width
        
        # RBF centers: μi = -1 + 0.2*i
        centers = []
        for i in range(num_nodes):
            center = []
            for j in range(input_dim):
                mu_val = mu_range[0] + mu_step * i
                mu_val = max(mu_range[0], min(mu_range[1], mu_val))
                center.append(mu_val)
            centers.append(center)
        
        self.centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32), requires_grad=False)
    
    def forward(self, x_now: torch.Tensor) -> torch.Tensor:
        """Compute RBF activations at time t"""
        if x_now.dim() == 1:
            x_now = x_now.unsqueeze(0)
        
        x_expanded = x_now.unsqueeze(1)  # (num_envs, 1, input_dim)
        centers_expanded = self.centers.unsqueeze(0)  # (1, num_nodes, input_dim)
        
        diff = x_expanded - centers_expanded  # (num_envs, num_nodes, input_dim)
        squared_dist = torch.sum(diff**2, dim=2)  # (num_envs, num_nodes)
        
        rbf_output = torch.exp(-squared_dist / (self.width**2))
        return rbf_output


class SurgicalCritic(nn.Module):
    """Critic network: Γ̂(t) = Ŵc^T Sc(z̄(t))"""
    def __init__(self, augmented_state_dim: int, params: dict):
        super().__init__()
        
        net_cfg = params.get('network_config', {})
        num_nodes = net_cfg.get('critic_nodes', 10)
        gaussian_width = net_cfg.get('critic_gaussian_width', 15.0)
        mu_range = net_cfg.get('mu_range', [-1.0, 1.0])
        mu_step = net_cfg.get('mu_step', 0.2)
        
        self.rbf_layer = RBFLayer(
            input_dim=augmented_state_dim,
            num_nodes=num_nodes,
            gaussian_width=gaussian_width,
            mu_range=mu_range,
            mu_step=mu_step
        )
        
        self.output_layer = nn.Linear(num_nodes, 1)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)
    
    def forward(self, z_bar_now: torch.Tensor) -> torch.Tensor:
        """Forward pass at time t"""
        if z_bar_now.dim() == 1:
            z_bar_now = z_bar_now.unsqueeze(0)
        
        rbf_features_now = self.rbf_layer(z_bar_now)
        value_now = self.output_layer(rbf_features_now)
        return value_now


class SurgicalActor(nn.Module):
    """Actor network: û(t) = Ŵa^T Sa(Za(t))"""
    def __init__(self, actor_input_dim: int, action_dim: int, params: dict):
        super().__init__()
        
        net_cfg = params.get('network_config', {})
        num_nodes = net_cfg.get('actor_nodes', 10)
        gaussian_width = net_cfg.get('actor_gaussian_width', 10.0)
        mu_range = net_cfg.get('mu_range', [-1.0, 1.0])
        mu_step = net_cfg.get('mu_step', 0.2)
        
        self.rbf_layer = RBFLayer(
            input_dim=actor_input_dim,
            num_nodes=num_nodes,
            gaussian_width=gaussian_width,
            mu_range=mu_range,
            mu_step=mu_step
        )
        
        self.output_layer = nn.Linear(num_nodes, action_dim)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)
    
    def forward(self, Za_now: torch.Tensor) -> torch.Tensor:
        """Forward pass at time t"""
        if Za_now.dim() == 1:
            Za_now = Za_now.unsqueeze(0)
        
        rbf_features_now = self.rbf_layer(Za_now)
        control_output_now = self.output_layer(rbf_features_now)
        return control_output_now


class DynamicsIdentifier(nn.Module):
    """Dynamics identifier: ż̂(t) = Ŵid^T Sid(ẑ(t),u(t))"""
    def __init__(self, state_dim: int, action_dim: int, params: dict):
        super().__init__()
        
        net_cfg = params.get('network_config', {})
        num_nodes = net_cfg.get('identifier_nodes', 10)
        gaussian_width = net_cfg.get('identifier_gaussian_width', 100.0)
        mu_range = net_cfg.get('mu_range', [-1.0, 1.0])
        mu_step = net_cfg.get('mu_step', 0.2)
        
        input_dim = state_dim + action_dim
        
        self.rbf_layer = RBFLayer(
            input_dim=input_dim,
            num_nodes=num_nodes,
            gaussian_width=gaussian_width,
            mu_range=mu_range,
            mu_step=mu_step
        )
        
        self.output_layer = nn.Linear(num_nodes, state_dim)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.output_layer.bias, 0.0)
    
    def forward(self, z_hat_now: torch.Tensor, u_now: torch.Tensor) -> torch.Tensor:
        """Forward pass at time t"""
        if z_hat_now.dim() == 1:
            z_hat_now = z_hat_now.unsqueeze(0)
        if u_now.dim() == 1:
            u_now = u_now.unsqueeze(0)
        
        combined_input_now = torch.cat([z_hat_now, u_now], dim=-1)
        rbf_features_now = self.rbf_layer(combined_input_now)
        z_hat_dot_now = self.output_layer(rbf_features_now)
        return z_hat_dot_now


class SurgicalActorCritic(nn.Module):
    """Complete Actor-Critic system with PE noise and clear exploration control"""
    def __init__(self, params: dict):
        super().__init__()
        
        self.state_dim = params.get('state_dim', 9)
        self.action_dim = params.get('action_dim', 3)
        self.augmented_state_dim = params.get('augmented_state_dim', 12)
        self.actor_input_dim = params.get('actor_input_dim', 18)
        
        # Networks
        self.critic = SurgicalCritic(self.augmented_state_dim, params)
        self.actor = SurgicalActor(self.actor_input_dim, self.action_dim, params)
        self.identifier = DynamicsIdentifier(self.state_dim, self.action_dim, params)
        
        # Exploration parameters
        exploration_cfg = params.get('exploration', {})
        self.exploration_noise = exploration_cfg.get('exploration_noise', 0.0005)
        
        # PE parameters
        pe_cfg = exploration_cfg.get('persistent_excitation', {})
        self.pe_enabled = pe_cfg.get('enabled', True)
        self.pe_amplitude = pe_cfg.get('amplitude', 0.0005)
        self.pe_num_freq = pe_cfg.get('num_frequencies', 10)
        self.pe_base_freq = pe_cfg.get('base_frequency', 1.0)
        self.pe_time_now = 0.0  # PE time at current step
    
    def get_action_now(self, Za_now: torch.Tensor, dt: float, add_exploration: bool = True) -> torch.Tensor:
        """Get action at time t with optional exploration noise"""
        actor_output_now = self.actor(Za_now)
        
        if add_exploration:
            # Exploration noise
            if self.exploration_noise > 0:
                noise_now = torch.randn_like(actor_output_now) * self.exploration_noise
                actor_output_now = actor_output_now + noise_now
            
            # PE signal: 0.0005 * Σ(i=1 to 10) sin(i*t)
            if self.pe_enabled:
                pe_signal_now = self._generate_pe_signal_now(dt, actor_output_now.shape)
                actor_output_now = actor_output_now + pe_signal_now
        
        return torch.clamp(actor_output_now, -1.0, 1.0)
    
    def _generate_pe_signal_now(self, dt: float, shape: tuple) -> torch.Tensor:
        """Generate PE signal at time t: 0.0005 * Σ(i=1 to 10) sin(i*t)"""
        self.pe_time_now += dt
        
        pe_sum_now = 0.0
        for i in range(1, self.pe_num_freq + 1):
            pe_sum_now += torch.sin(torch.tensor(i * self.pe_base_freq * self.pe_time_now))
        
        pe_signal_now = self.pe_amplitude * pe_sum_now
        return pe_signal_now * torch.ones(shape, device=next(self.parameters()).device)
    
    def evaluate_value_now(self, z_bar_now: torch.Tensor) -> torch.Tensor:
        """Evaluate value function at time t"""
        return self.critic(z_bar_now)
    
    def predict_dynamics_now(self, z_hat_now: torch.Tensor, u_now: torch.Tensor) -> torch.Tensor:
        """Predict state derivative at time t"""
        return self.identifier(z_hat_now, u_now)
    
    def reset_pe_time(self):
        """Reset PE time for episode start"""
        self.pe_time_now = 0.0