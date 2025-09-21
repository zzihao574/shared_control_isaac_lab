"""
Neural network architectures for MADDPG with state normalization and residual connections.
OPTIMIZED: Removed dropout, added residual bypass connections, removed _build_mlp.

Features:
- FixedScaler for state normalization without offsets
- Deterministic actor with normalized action output [-1,1]  
- Centralized critic expecting normalized actions
- Residual bypass connections for improved gradient flow
- Configurable layers and orthogonal initialization
- No dropout (removed for simplification)
"""

import torch
import torch.nn as nn
from typing import Sequence, Optional

class FixedScaler(nn.Module):
    """轻量固定缩放器（逐维常数乘法）"""
    def __init__(self, dim: int, factors: torch.Tensor):
        super().__init__()
        assert factors.numel() == dim, f"Factors size {factors.numel()} != dim {dim}"
        self.register_buffer("factors", factors.view(1, dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factors

class Actor(nn.Module):
    """
    Actor网络，动态检测输入维度并在指定层实现残差连接。
    OPTIMIZED: 移除dropout，添加可配置的残差bypass连接
    
    Features:
    - 状态固定缩放 (FixedScaler)
    - 仅输出动作均值（去掉std头）
    - tanh激活确保输出 ∈ [-1,1]
    - 可配置的残差bypass连接层
    - 移除dropout以简化架构
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
        
        # 状态缩放器
        if obs_factors is not None:
            self.state_scale = FixedScaler(state_dim, obs_factors)
        else:
            self.state_scale = nn.Identity()
        
        self.input_dim = state_dim
        self.hidden_layers = hidden_layers
        self.input_bypass_layers = set(input_bypass_layers) if input_bypass_layers else set()
        
        # 构建网络层
        self.layers = nn.ModuleList()
        self.bypass_configs = {}  # 存储每层的bypass配置
        
        last_dim = self.input_dim
        for i, hidden_dim in enumerate(hidden_layers):
            if i in self.input_bypass_layers:
                # 计算这一层的连接配置
                fc_dim = hidden_dim - self.input_dim  # 全连接部分的输出维度
                
                if fc_dim <= 0:
                    # 隐藏层太小，无法做bypass，回退到正常全连接
                    print(f"  [WARNING] Layer {i}: hidden_dim({hidden_dim}) <= input_dim({self.input_dim}), using normal FC")
                    layer = nn.Linear(last_dim, hidden_dim)
                else:
                    # 创建部分连接层
                    layer = nn.Linear(last_dim, fc_dim)
                    self.bypass_configs[i] = {
                        'fc_dim': fc_dim,
                        'bypass_dim': self.input_dim,
                        'total_dim': hidden_dim
                    }
                    print(f"  Layer {i}: {last_dim} -> {fc_dim} (FC) + {self.input_dim} (bypass) = {hidden_dim}")
            else:
                # 正常的全连接层
                layer = nn.Linear(last_dim, hidden_dim)
                print(f"  Layer {i}: {last_dim} -> {hidden_dim} (normal FC)")
            
            self.layers.append(layer)
            last_dim = hidden_dim
            
            # 正交初始化
            if orthogonal_init:
                nn.init.orthogonal_(layer.weight, gain=ortho_gain_hidden)
                nn.init.constant_(layer.bias, 0.0)
        
        # 输出头
        self.mean_head = nn.Linear(last_dim, action_dim)
        if orthogonal_init:
            nn.init.orthogonal_(self.mean_head.weight, gain=ortho_gain_output)
            nn.init.constant_(self.mean_head.bias, 0.0)
        
        # 打印架构摘要
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
        """前向传播，动态处理输入直通"""
        # 缩放并保存原始输入
        original_input = self.state_scale(state)
        x = original_input
        
        # 通过每一层
        for i, layer in enumerate(self.layers):
            if i in self.bypass_configs:
                # 部分连接层：全连接输出 + 输入直通
                config = self.bypass_configs[i]
                
                # 全连接部分
                fc_output = layer(x)  # [batch, fc_dim]
                
                # 拼接全连接输出和原始输入
                x = torch.cat([fc_output, original_input], dim=-1)  # [batch, hidden_dim]
                
                # 应用激活函数
                x = torch.relu(x)
            else:
                # 正常全连接层
                x = torch.relu(layer(x))
        
        # 输出归一化动作
        return torch.tanh(self.mean_head(x))

class Critic(nn.Module):
    """
    Critic网络，动态处理状态+动作的联合输入，支持残差连接。
    OPTIMIZED: 移除dropout，添加可配置的残差bypass连接
    
    Features:
    - 状态固定缩放 (FixedScaler)
    - 直接接受归一化动作（不在内部做 /a_max）
    - 可配置的残差bypass连接层
    - 移除dropout以简化架构
    - 输出单个Q值
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
        
        # 状态缩放器
        if obs_factors is not None:
            self.state_scale = FixedScaler(total_state_dimension, obs_factors)
        else:
            self.state_scale = nn.Identity()
        
        # Critic的输入是状态+动作
        self.input_dim = total_state_dimension + total_action_dimension
        self.state_dim = total_state_dimension
        self.action_dim = total_action_dimension
        self.hidden_layers = hidden_layers
        self.input_bypass_layers = set(input_bypass_layers) if input_bypass_layers else set()
        
        # 构建网络层
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
        
        # Q值输出头
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
        """前向传播"""
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