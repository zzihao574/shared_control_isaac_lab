"""
Neural network architectures for MADDPG with state normalization and unified action scaling.

Features:
- FixedScaler for state normalization without offsets
- Deterministic actor with normalized action output [-1,1]
- Centralized critic expecting normalized actions
- Configurable layers and orthogonal initialization
"""

import torch
import torch.nn as nn
from typing import Sequence, Optional, List

class FixedScaler(nn.Module):
    """轻量固定缩放器（逐维常数乘法）"""
    def __init__(self, dim: int, factors: torch.Tensor):
        super().__init__()
        assert factors.numel() == dim, f"Factors size {factors.numel()} != dim {dim}"
        self.register_buffer("factors", factors.view(1, dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factors

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
    确定性Actor网络，输出归一化动作。
    
    Features:
    - 状态固定缩放 (FixedScaler)
    - 仅输出动作均值（去掉std头）
    - tanh激活确保输出 ∈ [-1,1]
    - 可配置网络架构和正交初始化
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_layers: Sequence[int] = (128, 128, 128),
                 dropout_p: float = 0.0,
                 orthogonal_init: bool = True,
                 ortho_gain_hidden: float = 1.414,
                 ortho_gain_output: float = 0.01,
                 obs_factors: Optional[torch.Tensor] = None):
        super().__init__()
        
        # 状态缩放器（如果提供缩放因子）
        if obs_factors is not None:
            self.state_scale = FixedScaler(state_dim, obs_factors)
        else:
            self.state_scale = nn.Identity()
        
        # 网络主体
        layers = []
        last = state_dim
        for h in hidden_layers:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            if dropout_p > 0:
                layers += [nn.Dropout(p=float(dropout_p))]
            last = h
        self.body = nn.Sequential(*layers)
        
        # 动作输出头（仅均值）
        self.mean_head = nn.Linear(last, action_dim)
        
        # 正交初始化
        if orthogonal_init:
            for m in self.body:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=ortho_gain_hidden)
                    nn.init.constant_(m.bias, 0.0)
            nn.init.orthogonal_(self.mean_head.weight, gain=ortho_gain_output)
            nn.init.constant_(self.mean_head.bias, 0.0)

    def forward(self, state: torch.Tensor):
        """前向传播：输出归一化动作 ∈ [-1,1]"""
        s = self.state_scale(state)
        raw = self.mean_head(self.body(s))
        return torch.tanh(raw)  # a_norm ∈ [-1,1]

class Critic(nn.Module):
    """
    集中式Critic网络，期望归一化动作输入。
    
    Features:
    - 状态固定缩放 (FixedScaler)
    - 直接接受归一化动作（不在内部做 /a_max）
    - 可配置网络架构和正交初始化
    - 输出单个Q值
    """
    
    def __init__(self,
                 total_state_dimension: int,
                 total_action_dimension: int,
                 hidden_layers: Sequence[int] = (128, 128, 128),
                 dropout_p: float = 0.0,
                 orthogonal_init: bool = True,
                 ortho_gain_hidden: float = 1.414,
                 ortho_gain_output: float = 0.01,
                 obs_factors: Optional[torch.Tensor] = None):
        super().__init__()
        
        # 状态缩放器（如果提供缩放因子）
        if obs_factors is not None:
            self.state_scale = FixedScaler(total_state_dimension, obs_factors)
        else:
            self.state_scale = nn.Identity()
        
        # 网络主体
        in_dim = total_state_dimension + total_action_dimension
        layers = []
        last = in_dim
        for h in hidden_layers:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            if dropout_p > 0:
                layers += [nn.Dropout(p=float(dropout_p))]
            last = h
        self.body = nn.Sequential(*layers)
        
        # Q值输出头
        self.q_head = nn.Linear(last, 1)
        
        # 正交初始化
        if orthogonal_init:
            for m in self.body:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=ortho_gain_hidden)
                    nn.init.constant_(m.bias, 0.0)
            nn.init.orthogonal_(self.q_head.weight, gain=ortho_gain_output)
            nn.init.constant_(self.q_head.bias, 0.0)

    def forward(self, states: torch.Tensor, actions_norm: torch.Tensor):
        """前向传播：期望states为物理量，actions_norm为归一化动作"""
        s = self.state_scale(states)  # 状态缩放
        x = torch.cat([s, actions_norm], dim=-1)  # Critic期望a_norm
        h = self.body(x)
        return self.q_head(h)