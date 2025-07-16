# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""手术机器人强化学习算法配置包 - 论文对齐版本"""

# 导入基于模型的强化学习算法
try:
    from .mbrl import (
        # Actor-Critic 组件
        SurgicalActor,
        SurgicalCritic,
        DynamicsIdentifierNetwork,
        HJBSolver,
        SurgicalActorCritic,
        # 共享控制组件
        SharedControlTrainer,
        HumanImpedanceModel,
        PaperCostFunction,
        AdaptiveSharedControl,
        ReplayBuffer,
    )
    
    _MBRL_AVAILABLE = True
    print("[INFO] MBRL algorithms loaded successfully")
    
except ImportError as e:
    print(f"[WARNING] Failed to import MBRL algorithms: {e}")
    _MBRL_AVAILABLE = False

# 导出所有可用的组件
__all__ = []

if _MBRL_AVAILABLE:
    __all__.extend([
        # Actor-Critic 组件
        "SurgicalActor",
        "SurgicalCritic",
        "DynamicsIdentifierNetwork", 
        "HJBSolver",
        "SurgicalActorCritic",
        # 共享控制组件
        "SharedControlTrainer",
        "HumanImpedanceModel",
        "PaperCostFunction",
        "AdaptiveSharedControl", 
        "ReplayBuffer",
    ])

# 版本信息
__version__ = "1.0.0-paper-aligned"