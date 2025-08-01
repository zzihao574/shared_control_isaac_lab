# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-agent environments for surgical robot training."""

import gymnasium as gym

from .surgical_direct_marl_env import SurgicalDirectMARLEnv
from .surgical_direct_marl_env_cfg import SurgicalDirectMARLEnvCfg

# Register environment with gymnasium
gym.register(
    id="Isaac-Surgical-MARL-Direct-v0",
    entry_point="surgical_project.envs.multi_agent.surgical_direct_marl_env:SurgicalDirectMARLEnv",
    kwargs={
        "env_cfg_entry_point": "surgical_project.envs.multi_agent.surgical_direct_marl_env_cfg:SurgicalDirectMARLEnvCfg",
    },
)

print(f"[INFO] 成功注册 Isaac-Surgical-MARL-Direct-v0")
print(f"[INFO] 环境特性:")
print(f"  - 人机协作双智能体控制")
print(f"  - 与MBRL环境物理参数对齐")
print(f"  - 距离驱动的轨迹切换")
print(f"  - 信任机制和协作奖励")
print(f"  - 差异化智能体观测和奖励")

__all__ = ["SurgicalDirectMARLEnv", "SurgicalDirectMARLEnvCfg"]