# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Surgical Human-Robot Shared Control Environment
"""

import gymnasium as gym
from . import agents 

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Surgical-Direct-v0",
    entry_point="surgical_project.envs.single_agent.surgical_direct_env:SurgicalDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "surgical_project.envs.single_agent.surgical_direct_env_cfg:SurgicalDirectEnvCfg",
    },
)

print(f"[INFO] Successfully registered Isaac-Surgical-Direct-v0")
print(f"[INFO] Environment features:")
print(f"  - Paper-aligned human-robot shared control")
print(f"  - CBF (Control Barrier Function) constraints")  
print(f"  - Distance-based trajectory switching")
print(f"  - Human impedance model (Equation 6)")
print(f"  - Extended cost function (Equation 13 + CBF)")
print(f"  - Simple YAML configuration support")