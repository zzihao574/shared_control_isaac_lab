"""
Neural network architectures for MADDPG with state normalization and residual connections.
Features fixed state scaling, deterministic actor, centralized critic, and configurable bypasses.
"""

import torch
import torch.nn as nn
from typing import Sequence, Optional

class FixedScaler(nn.Module):
    """Lightweight fixed scaler with per-dimension constant multiplication."""
    def __init__(self, dim: int, factors: torch.Tensor):
        super().__init__()
        assert factors.numel() == dim, f"Factors size {factors.numel()} != dim {dim}"
        self.register_buffer("factors", factors.view(1, dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factors

class Actor(nn.Module):
    """
    Actor network with dynamic input dimension detection and residual bypass connections.
    Features fixed state scaling, tanh activation ensuring output ∈ [-1,1], and configurable bypasses.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_layers: Sequence[int] = (128, 128, 128),
                 input_bypass_layers: Sequence[int] = None,
                 orthogonal_init: bool = True,
                 ortho_gain_hidden: float = 1.414,
                 ortho_gain_output: float = 0.01,
                 obs_factors: Optional[torch.Tensor] = None):
        super().__init__()
        
        # State scaler
        if obs_factors is not None:
            self.state_scale = FixedScaler(state_dim, obs_factors)
        else:
            self.state_scale = nn.Identity()
        
        self.input_dim = state_dim
        self.hidden_layers = hidden_layers
        self.input_bypass_layers = set(input_bypass_layers) if input_bypass_layers else set()
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.bypass_configs = {}  # Store bypass configuration for each layer
        
        last_dim = self.input_dim
        for i, hidden_dim in enumerate(hidden_layers):
            if i in self.input_bypass_layers:
                # Calculate connection configuration for this layer
                fc_dim = hidden_dim - self.input_dim  # FC part output dimension
                
                if fc_dim <= 0:
                    # Hidden layer too small for bypass, fallback to normal FC
                    print(f"  [WARNING] Layer {i}: hidden_dim({hidden_dim}) <= input_dim({self.input_dim}), using normal FC")
                    layer = nn.Linear(last_dim, hidden_dim)
                else:
                    # Create partial connection layer
                    layer = nn.Linear(last_dim, fc_dim)
                    self.bypass_configs[i] = {
                        'fc_dim': fc_dim,
                        'bypass_dim': self.input_dim,
                        'total_dim': hidden_dim
                    }
                    print(f"  Layer {i}: {last_dim} -> {fc_dim} (FC) + {self.input_dim} (bypass) = {hidden_dim}")
            else:
                # Normal fully connected layer
                layer = nn.Linear(last_dim, hidden_dim)
                print(f"  Layer {i}: {last_dim} -> {hidden_dim} (normal FC)")
            
            self.layers.append(layer)
            last_dim = hidden_dim
            
            # Orthogonal initialization
            if orthogonal_init:
                nn.init.orthogonal_(layer.weight, gain=ortho_gain_hidden)
                nn.init.constant_(layer.bias, 0.0)
        
        # Output head
        self.mean_head = nn.Linear(last_dim, action_dim)
        if orthogonal_init:
            nn.init.orthogonal_(self.mean_head.weight, gain=ortho_gain_output)
            nn.init.constant_(self.mean_head.bias, 0.0)
        
        # Print architecture summary
        self._print_architecture_summary()
    
    def _print_architecture_summary(self):
        print(f"\n[Actor Network Architecture]")
        print(f"  Input dimension: {self.input_dim}")
        print(f"  Hidden layers: {self.hidden_layers}")
        print(f"  Bypass layers: {sorted(self.input_bypass_layers) if self.input_bypass_layers else 'None'}")
        if self.bypass_configs:
            print(f"  Bypass configurations:")
            for layer_idx, config in sorted(self.bypass_configs.items()):
                print(f"    Layer {layer_idx}: {config['fc_dim']} (FC) + {config['bypass_dim']} (bypass)")

    def forward(self, state: torch.Tensor):
        """Forward pass with dynamic input bypass handling."""
        # Scale and save original input
        original_input = self.state_scale(state)
        x = original_input
        
        # Pass through each layer
        for i, layer in enumerate(self.layers):
            if i in self.bypass_configs:
                # Partial connection layer: FC output + input bypass
                config = self.bypass_configs[i]
                
                # FC part
                fc_output = layer(x)  # [batch, fc_dim]
                
                # Concatenate FC output and original input
                x = torch.cat([fc_output, original_input], dim=-1)  # [batch, hidden_dim]
                
                # Apply activation
                x = torch.relu(x)
            else:
                # Normal fully connected layer
                x = torch.relu(layer(x))
        
        # Output normalized actions
        return torch.tanh(self.mean_head(x))

class Critic(nn.Module):
    """
    Critic network with dynamic state+action joint input processing and residual bypasses.
    Features fixed state scaling, direct normalized action acceptance, and configurable bypasses.
    """
    
    def __init__(self,
                 total_state_dimension: int,
                 total_action_dimension: int,
                 hidden_layers: Sequence[int] = (128, 128, 128),
                 input_bypass_layers: Sequence[int] = None,
                 orthogonal_init: bool = True,
                 ortho_gain_hidden: float = 1.414,
                 ortho_gain_output: float = 0.01,
                 obs_factors: Optional[torch.Tensor] = None):
        super().__init__()
        
        # State scaler
        if obs_factors is not None:
            self.state_scale = FixedScaler(total_state_dimension, obs_factors)
        else:
            self.state_scale = nn.Identity()
        
        # Critic input is state + action
        self.input_dim = total_state_dimension + total_action_dimension
        self.state_dim = total_state_dimension
        self.action_dim = total_action_dimension
        self.hidden_layers = hidden_layers
        self.input_bypass_layers = set(input_bypass_layers) if input_bypass_layers else set()
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.bypass_configs = {}
        
        last_dim = self.input_dim
        for i, hidden_dim in enumerate(hidden_layers):
            if i in self.input_bypass_layers:
                fc_dim = hidden_dim - self.input_dim
                
                if fc_dim <= 0:
                    print(f"  [WARNING] Layer {i}: hidden_dim({hidden_dim}) <= input_dim({self.input_dim}), using normal FC")
                    layer = nn.Linear(last_dim, hidden_dim)
                else:
                    layer = nn.Linear(last_dim, fc_dim)
                    self.bypass_configs[i] = {
                        'fc_dim': fc_dim,
                        'bypass_dim': self.input_dim,
                        'total_dim': hidden_dim
                    }
                    print(f"  Layer {i}: {last_dim} -> {fc_dim} (FC) + {self.input_dim} (bypass) = {hidden_dim}")
            else:
                layer = nn.Linear(last_dim, hidden_dim)
                print(f"  Layer {i}: {last_dim} -> {hidden_dim} (normal FC)")
            
            self.layers.append(layer)
            last_dim = hidden_dim
            
            if orthogonal_init:
                nn.init.orthogonal_(layer.weight, gain=ortho_gain_hidden)
                nn.init.constant_(layer.bias, 0.0)
        
        # Q-value output head
        self.q_head = nn.Linear(last_dim, 1)
        if orthogonal_init:
            nn.init.orthogonal_(self.q_head.weight, gain=ortho_gain_output)
            nn.init.constant_(self.q_head.bias, 0.0)
        
        self._print_architecture_summary()
    
    def _print_architecture_summary(self):
        print(f"\n[Critic Network Architecture]")
        print(f"  Input dimension: {self.input_dim} (state:{self.state_dim} + action:{self.action_dim})")
        print(f"  Hidden layers: {self.hidden_layers}")
        print(f"  Bypass layers: {sorted(self.input_bypass_layers) if self.input_bypass_layers else 'None'}")
        if self.bypass_configs:
            print(f"  Bypass configurations:")
            for layer_idx, config in sorted(self.bypass_configs.items()):
                print(f"    Layer {layer_idx}: {config['fc_dim']} (FC) + {config['bypass_dim']} (bypass)")

    def forward(self, states: torch.Tensor, actions_norm: torch.Tensor):
        """Forward pass with joint state-action input processing."""
        s = self.state_scale(states)
        original_input = torch.cat([s, actions_norm], dim=-1)
        x = original_input
        
        for i, layer in enumerate(self.layers):
            if i in self.bypass_configs:
                config = self.bypass_configs[i]
                fc_output = layer(x)
                x = torch.cat([fc_output, original_input], dim=-1)
                x = torch.relu(x)
            else:
                x = torch.relu(layer(x))
        
        return self.q_head(x)