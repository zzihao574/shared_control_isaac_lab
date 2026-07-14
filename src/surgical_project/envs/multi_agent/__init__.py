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
    disable_env_checker=True,
)

print(f"[INFO] Successfully registered Isaac-Surgical-MARL-Direct-v0")
print(f"[INFO] Environment features:")
print(f"  - Human-robot collaborative dual-agent control")
print(f"  - Physics parameters aligned with MBRL environment")
print(f"  - Distance-driven trajectory switching")
print(f"  - Trust mechanism and collaborative rewards")
print(f"  - Differentiated agent observations and rewards")

__all__ = ["SurgicalDirectMARLEnv", "SurgicalDirectMARLEnvCfg"]
