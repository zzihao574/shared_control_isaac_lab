"""
rMAPPO package initialization.
Simplified implementation for continuous actions with RNN.
Clear naming: Policy (networks) + Trainer (algorithm).
"""

from .r_mappo_core import RMAPPOPolicy, RMAPPOTrainer, R_Actor, R_Critic
from .rollout_buffer import SharedRolloutBuffer

__all__ = [
    'RMAPPOPolicy',        # 策略类：管理网络和优化器
    'RMAPPOTrainer',       # 训练器：PPO算法更新逻辑
    'R_Actor',             # Actor网络
    'R_Critic',            # Critic网络
    'SharedRolloutBuffer', # 轨迹缓冲区
]