"""论文对齐的手术机器人Actor-Critic网络 - 完整版本，支持YAML配置"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurgicalCritic(nn.Module):
    """论文方程(29)的Critic网络: Γ = W_c^T S_c(z̄) + ε_c"""
    def __init__(self, augmented_state_dim: int, network_cfg: dict = None):
        super().__init__()
        
        # 从配置获取网络架构参数
        if network_cfg is None:
            network_cfg = {}
        
        critic_cfg = network_cfg.get('critic', {})
        hidden_dims = critic_cfg.get('hidden_dims', [256, 128])
        activation = critic_cfg.get('activation', 'relu')
        
        layers = []
        dims = [augmented_state_dim] + hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # 不在最后一层添加激活函数
                if activation.lower() == 'relu':
                    layers.append(nn.ReLU())
                elif activation.lower() == 'elu':
                    layers.append(nn.ELU())
                elif activation.lower() == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    layers.append(nn.ReLU())  # 默认使用ReLU
        
        self.value_network = nn.Sequential(*layers)
        
        # 初始化方式配置
        initializer_cfg = network_cfg.get('initializer', {})
        init_method = initializer_cfg.get('name', 'orthogonal')
        init_gain = initializer_cfg.get('gain', 1.0)
        
        if init_method == 'orthogonal':
            self.apply(lambda m: self._orthogonal_init(m, init_gain))
        elif init_method == 'xavier':
            self.apply(lambda m: self._xavier_init(m, init_gain))
        else:
            self.apply(lambda m: self._orthogonal_init(m, init_gain))  # 默认
        
    def _orthogonal_init(self, m, gain=1.0):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            nn.init.constant_(m.bias, 0.0)
    
    def _xavier_init(self, m, gain=1.0):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=gain)
            nn.init.constant_(m.bias, 0.0)
        
    def forward(self, augmented_state: torch.Tensor) -> torch.Tensor:
        if augmented_state.dim() == 1:
            augmented_state = augmented_state.unsqueeze(0)
        
        return self.value_network(augmented_state)


class SurgicalActor(nn.Module):
    """论文方程(50)的Actor网络: u = Ŵ_a^T S_a(Z_a)"""
    def __init__(self, input_dim: int, action_dim: int, network_cfg: dict = None):
        super().__init__()
        
        # 从配置获取网络架构参数
        if network_cfg is None:
            network_cfg = {}
        
        actor_cfg = network_cfg.get('actor', {})
        hidden_dims = actor_cfg.get('hidden_dims', [256, 128])
        activation = actor_cfg.get('activation', 'relu')
        output_activation = actor_cfg.get('output_activation', 'tanh')
        
        layers = []
        dims = [input_dim] + hidden_dims + [action_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # 隐藏层激活函数
                if activation.lower() == 'relu':
                    layers.append(nn.ReLU())
                elif activation.lower() == 'elu':
                    layers.append(nn.ELU())
                elif activation.lower() == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
        
        # 输出激活函数
        if output_activation.lower() == 'tanh':
            self.output_activation = torch.tanh
        elif output_activation.lower() == 'sigmoid':
            self.output_activation = torch.sigmoid
        else:
            self.output_activation = torch.tanh  # 默认
        
        # 初始化方式配置
        initializer_cfg = network_cfg.get('initializer', {})
        init_method = initializer_cfg.get('name', 'orthogonal')
        init_gain = initializer_cfg.get('gain', 1.0)
        
        if init_method == 'orthogonal':
            self.apply(lambda m: self._orthogonal_init(m, init_gain))
        elif init_method == 'xavier':
            self.apply(lambda m: self._xavier_init(m, init_gain))
        else:
            self.apply(lambda m: self._orthogonal_init(m, init_gain))
    
    def _orthogonal_init(self, m, gain=1.0):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            nn.init.constant_(m.bias, 0.0)
    
    def _xavier_init(self, m, gain=1.0):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=gain)
            nn.init.constant_(m.bias, 0.0)
            
    def forward(self, input_state: torch.Tensor) -> torch.Tensor:
        if input_state.dim() == 1:
            input_state = input_state.unsqueeze(0)
        
        network_output = self.network(input_state)
        return self.output_activation(network_output)


class DynamicsIdentifierNetwork(nn.Module):
    """论文方程(34)的动力学识别网络: ż = W_id^T S_id(z, u) + ε_id"""
    def __init__(self, state_dim: int, action_dim: int, network_cfg: dict = None):
        super().__init__()
        
        # 从配置获取网络架构参数
        if network_cfg is None:
            network_cfg = {}
        
        identifier_cfg = network_cfg.get('identifier', {})
        hidden_dims = identifier_cfg.get('hidden_dims', [128, 128])
        activation = identifier_cfg.get('activation', 'relu')
        
        layers = []
        input_dim = state_dim + action_dim
        dims = [input_dim] + hidden_dims + [state_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                if activation.lower() == 'relu':
                    layers.append(nn.ReLU())
                elif activation.lower() == 'elu':
                    layers.append(nn.ELU())
                elif activation.lower() == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    layers.append(nn.ReLU())
                
        self.network = nn.Sequential(*layers)
        
        # 动力学网络使用更小的初始化增益
        initializer_cfg = network_cfg.get('initializer', {})
        init_method = initializer_cfg.get('name', 'orthogonal')
        init_gain = initializer_cfg.get('identifier_gain', 0.1)  # 动力学网络专用增益
        
        if init_method == 'orthogonal':
            self.apply(lambda m: self._orthogonal_init(m, init_gain))
        elif init_method == 'xavier':
            self.apply(lambda m: self._xavier_init(m, init_gain))
        else:
            self.apply(lambda m: self._orthogonal_init(m, init_gain))
    
    def _orthogonal_init(self, m, gain=0.1):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            nn.init.constant_(m.bias, 0.0)
    
    def _xavier_init(self, m, gain=0.1):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=gain)
            nn.init.constant_(m.bias, 0.0)
            
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class SurgicalActorCritic(nn.Module):
    """论文Actor-Critic网络，使用神经网络近似求解HJB方程 - 支持配置"""
    def __init__(self, state_dim: int, action_dim: int, augmented_state_dim: int, network_cfg: dict = None):
        super().__init__()
        
        self.state_dim = state_dim  # z ∈ R^9
        self.action_dim = action_dim  # u ∈ R^3
        self.augmented_state_dim = augmented_state_dim  # z̄ ∈ R^12
        
        # 传递网络配置给各个子网络
        self.critic = SurgicalCritic(augmented_state_dim, network_cfg)
        self.actor = SurgicalActor(augmented_state_dim, action_dim, network_cfg)
        self.identifier = DynamicsIdentifierNetwork(state_dim, action_dim, network_cfg)
        
        # 从配置获取探索参数
        self.action_noise_std = network_cfg.get('action_noise_std', 0.01) if network_cfg else 0.01
        
    def get_action(self, augmented_state: torch.Tensor, deterministic: bool = False, 
                   exploration_noise: float = None) -> torch.Tensor:
        action = self.actor(augmented_state)
        
        if not deterministic:
            # 使用配置的探索噪声
            noise_std = exploration_noise if exploration_noise is not None else self.action_noise_std
            noise = torch.randn_like(action) * noise_std
            action = action + noise
                
        return torch.clamp(action, -1.0, 1.0)
    
    def evaluate_value(self, augmented_state: torch.Tensor) -> torch.Tensor:
        return self.critic(augmented_state)
    
    def compute_actor_loss(self, augmented_state: torch.Tensor) -> torch.Tensor:
        value = self.critic(augmented_state)
        return -value.mean()