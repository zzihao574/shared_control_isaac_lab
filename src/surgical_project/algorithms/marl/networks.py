"""
Neural network architectures for MADDPG algorithm.
(MODIFIED VERSION - Simplified, NetworkAnalyzer removed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Actor(nn.Module):
    def __init__(self, state_dimension: int, action_dimension: int, hidden_dimension: int = 64, max_action_magnitude: float = 1.0):
        super(Actor, self).__init__()
        self.input_layer = nn.Linear(state_dimension, hidden_dimension)
        self.hidden_layer = nn.Linear(hidden_dimension, hidden_dimension)
        self.output_layer = nn.Linear(hidden_dimension, action_dimension * 2)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.input_layer(state))
        x = F.relu(self.hidden_layer(x))
        output = self.output_layer(x)
        action_mean, log_std = torch.chunk(output, 2, dim=-1)
        action_std = torch.exp(log_std.clamp(-20, 2)) # Clamp for stability
        return action_mean, action_std

class Critic(nn.Module):
    def __init__(self, total_state_dimension: int, total_action_dimension: int, hidden_dimension: int = 64):
        super(Critic, self).__init__()
        input_dimension = total_state_dimension + total_action_dimension
        self.input_layer = nn.Linear(input_dimension, hidden_dimension)
        self.hidden_layer = nn.Linear(hidden_dimension, hidden_dimension)
        self.output_layer = nn.Linear(hidden_dimension, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        network_input = torch.cat([states, actions], dim=-1)
        x = F.relu(self.input_layer(network_input))
        x = F.relu(self.hidden_layer(x))
        q_value = self.output_layer(x)
        return q_value