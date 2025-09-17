"""
Neural network architectures for MADDPG with configurable layers and initialization.

Features:
- Configurable hidden layer sizes and dropout
- Orthogonal weight initialization support
- Stochastic actor with mean and std outputs
- Centralized critic for multi-agent training
"""

import torch
import torch.nn as nn
from typing import Sequence, Optional, List

def _build_mlp(in_dim: int, hidden_layers: Sequence[int], out_dim: int,
               dropout_p: float = 0.0, final_activation: Optional[nn.Module] = None) -> nn.Sequential:
    """Build a multi-layer perceptron with configurable architecture."""
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden_layers:
        layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
        if dropout_p and dropout_p > 0.0:
            layers += [nn.Dropout(p=float(dropout_p))]
        last = h
    layers += [nn.Linear(last, out_dim)]
    if final_activation is not None:
        layers += [final_activation]
    return nn.Sequential(*layers)

def _apply_orthogonal_init(module: nn.Module, gain_hidden: float = 1.0, gain_output: float = 1.0):
    """Apply orthogonal initialization to linear layers."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain_hidden)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    if isinstance(module, nn.Sequential) and isinstance(module[-1], nn.Linear):
        nn.init.orthogonal_(module[-1].weight, gain=gain_output)
        if module[-1].bias is not None:
            nn.init.zeros_(module[-1].bias)

class Actor(nn.Module):
    """
    Stochastic actor network with configurable architecture.
    
    Features:
    - Outputs both mean and standard deviation for actions
    - Configurable hidden layers and dropout
    - Optional orthogonal weight initialization
    - Force constraint compliance via tanh activation
    """
    
    def __init__(self,
                 state_dimension: int,
                 action_dimension: int,
                 max_action_magnitude: float = 1.0,
                 hidden_layers: Sequence[int] = (256, 256),        # Default architecture
                 dropout_p: float = 0.0,                           # Dropout probability
                 orthogonal_init: bool = False,                    # Orthogonal initialization flag
                 ortho_gain_hidden: float = 1.0,                   # Hidden layer gain
                 ortho_gain_output: float = 0.01,                  # Output layer gain
                 std_scale: float = 1.0):                          # Standard deviation scaling
        super().__init__()
        self.max_action = max_action_magnitude
        self.std_scale = float(std_scale)
        
        out_dim = action_dimension * 2  # mean + log_std
        self.net = _build_mlp(state_dimension, hidden_layers, out_dim, dropout_p=dropout_p)
        
        if orthogonal_init:
            _apply_orthogonal_init(self.net, gain_hidden=ortho_gain_hidden, gain_output=ortho_gain_output)

    def forward(self, state: torch.Tensor):
        """Forward pass returning action mean and standard deviation."""
        output = self.net(state)
        action_mean, log_std = torch.chunk(output, 2, dim=-1)
        action_mean = torch.tanh(action_mean) * self.max_action
        action_std  = torch.exp(log_std.clamp(-20, 0)) * self.max_action * self.std_scale
        return action_mean, action_std

class Critic(nn.Module):
    """
    Centralized critic network for multi-agent training.
    
    Features:
    - Takes concatenated states and actions as input
    - Configurable hidden layers and dropout
    - Optional orthogonal weight initialization
    - Outputs single Q-value
    """
    
    def __init__(self,
                 total_state_dimension: int,
                 total_action_dimension: int,
                 hidden_layers: Sequence[int] = (256, 256),        # Default architecture
                 dropout_p: float = 0.0,
                 orthogonal_init: bool = False,
                 ortho_gain_hidden: float = 1.0,
                 ortho_gain_output: float = 1.0):
        super().__init__()
        in_dim = total_state_dimension + total_action_dimension
        self.net = _build_mlp(in_dim, hidden_layers, 1, dropout_p=dropout_p)
        if orthogonal_init:
            _apply_orthogonal_init(self.net, gain_hidden=ortho_gain_hidden, gain_output=ortho_gain_output)

    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        """Forward pass with concatenated states and actions."""
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)