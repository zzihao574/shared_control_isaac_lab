import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any
from .networks import Actor, Critic

class DDPGAgent:
    """
    Deep Deterministic Policy Gradient agent for shared network MADDPG.
    OPTIMIZED: Removed dropout parameters, added residual connections support.
    
    Features:
    - 确定性Actor，输出归一化动作[-1,1]
    - 集中式Critic，期望归一化动作输入
    - 状态固定缩放注入网络内部
    - 软更新目标网络
    - 批处理支持共享网络架构
    - 梯度范数统计用于训练诊断
    - 通过YAML配置网络架构
    - 残差连接支持改善梯度流动
    """
    
    def __init__(self, agent_id: str, state_dim: int, action_dim: int, 
                 total_state_dim: int, total_action_dim: int, params: Dict[str, Any], device: torch.device):
        """
        为共享网络架构初始化DDPG Agent。
        
        Args:
            agent_id: 此Agent的唯一标识符
            state_dim: 此Agent观测空间的维度
            action_dim: 此Agent动作空间的维度
            total_state_dim: 所有Agent观测维度之和
            total_action_dim: 所有Agent动作维度之和
            params: 配置参数字典
            device: PyTorch计算设备
        """
        self.agent_id = agent_id
        self.device = device
        
        # 加载超参数
        maddpg_cfg = params.get('maddpg_config', {})
        self.lr_actor = float(maddpg_cfg.get('lr_actor', 0.001))
        self.lr_critic = float(maddpg_cfg.get('lr_critic', 0.001))
        self.tau = float(maddpg_cfg.get('tau', 0.002))
        
        # 加载网络配置
        net_cfg = params.get('networks', {})
        actor_cfg = net_cfg.get('actor', {})
        critic_cfg = net_cfg.get('critic', {})

        # 网络架构参数 - OPTIMIZED: 移除dropout，添加residual连接支持
        actor_hidden_layers = actor_cfg.get('hidden_layers', [256, 256])
        actor_bypass_layers = actor_cfg.get('input_bypass_layers', [])  # NEW: Residual connections
        actor_ortho = bool(actor_cfg.get('orthogonal_init', False))
        actor_gain_h = float(actor_cfg.get('ortho_gain_hidden', 1.0))
        actor_gain_o = float(actor_cfg.get('ortho_gain_output', 0.01))

        critic_hidden_layers = critic_cfg.get('hidden_layers', [256, 256])
        critic_bypass_layers = critic_cfg.get('input_bypass_layers', [])  # NEW: Residual connections
        critic_ortho = bool(critic_cfg.get('orthogonal_init', False))
        critic_gain_h = float(critic_cfg.get('ortho_gain_hidden', 1.0))
        critic_gain_o = float(critic_cfg.get('ortho_gain_output', 1.0))
        
        # 读取状态缩放配置
        obs_scaling = params.get('obs_scaling', {})
        factors = obs_scaling.get('factors', [1.0] * state_dim)
        
        if len(factors) != state_dim:
            print(f"[WARNING] {agent_id} - Factors length mismatch: got {len(factors)}, expected {state_dim}")
            factors = [1.0] * state_dim
        
        # Actor使用单agent的缩放因子
        obs_factors_actor = torch.tensor(factors, dtype=torch.float32, device=device)
        
        # Critic使用所有agent的缩放因子（重复单agent的因子）
        num_agents = total_state_dim // state_dim
        factors_critic = factors * num_agents
        obs_factors_critic = torch.tensor(factors_critic, dtype=torch.float32, device=device)
        
        print(f"[DDPG AGENT] {agent_id}:")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  Total state dim: {total_state_dim}, Total action dim: {total_action_dim}")
        print(f"  Actor layers: {actor_hidden_layers}, bypass: {actor_bypass_layers}")
        print(f"  Critic layers: {critic_hidden_layers}, bypass: {critic_bypass_layers}")
        print(f"  Orthogonal init - Actor: {actor_ortho}, Critic: {critic_ortho}")
        print(f"  State scaling factors: {factors}")
        print(f"  LR - Actor: {self.lr_actor}, Critic: {self.lr_critic}")
        print(f"  Target update rate (tau): {self.tau}")
        
        # 初始化Actor网络（注入状态缩放和残差连接）
        self.actor = Actor(
            state_dim, action_dim,
            hidden_layers=actor_hidden_layers,
            input_bypass_layers=actor_bypass_layers,  # NEW: Residual connections
            orthogonal_init=actor_ortho,
            ortho_gain_hidden=actor_gain_h,
            ortho_gain_output=actor_gain_o,
            obs_factors=obs_factors_actor,
        ).to(device)

        self.actor_target = Actor(
            state_dim, action_dim,
            hidden_layers=actor_hidden_layers,
            input_bypass_layers=actor_bypass_layers,  # NEW: Residual connections
            orthogonal_init=actor_ortho,
            ortho_gain_hidden=actor_gain_h,
            ortho_gain_output=actor_gain_o,
            obs_factors=obs_factors_actor,
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # 初始化Critic网络（集中式训练，注入状态缩放和残差连接）
        self.critic = Critic(
            total_state_dim, total_action_dim,
            hidden_layers=critic_hidden_layers,
            input_bypass_layers=critic_bypass_layers,  # NEW: Residual connections
            orthogonal_init=critic_ortho,
            ortho_gain_hidden=critic_gain_h,
            ortho_gain_output=critic_gain_o,
            obs_factors=obs_factors_critic,
        ).to(device)

        self.critic_target = Critic(
            total_state_dim, total_action_dim,
            hidden_layers=critic_hidden_layers,
            input_bypass_layers=critic_bypass_layers,  # NEW: Residual connections
            orthogonal_init=critic_ortho,
            ortho_gain_hidden=critic_gain_h,
            ortho_gain_output=critic_gain_o,
            obs_factors=obs_factors_critic,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        print(f"[DDPG AGENT] {agent_id} initialized with residual connections and state normalization")

    def soft_update(self) -> None:
        """使用Polyak平均进行目标网络软更新。"""
        # 更新Actor目标网络
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        
        # 更新Critic目标网络
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )