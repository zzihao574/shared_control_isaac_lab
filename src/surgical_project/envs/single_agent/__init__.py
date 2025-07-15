# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Surgical Human-Robot Shared Control Environment.
"""

import gymnasium as gym

##
# Register Gym environments.
##

try:
    # Print debug info about registration
    module_name = __name__
    print(f"[DEBUG] Registering environment from module: {module_name}")
    
    gym.register(
        id="Isaac-Surgical-Direct-v0",
        entry_point="surgical_project.envs.single_agent.surgical_direct_env:SurgicalDirectEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "surgical_project.envs.single_agent.surgical_direct_env_cfg:SurgicalDirectEnvCfg",
        },
    )
    
    print(f"[INFO] Successfully registered Isaac-Surgical-Direct-v0")
    
except Exception as e:
    print(f"[ERROR] Failed to register environment: {e}")
    import traceback
    traceback.print_exc()