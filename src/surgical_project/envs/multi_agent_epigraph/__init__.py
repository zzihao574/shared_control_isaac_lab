# SPDX-License-Identifier: BSD-3-Clause
"""
Epigraph Multi-Agent Surgical Environment Package.
Registers "Isaac-Surgical-MARL-Epigraph-v0" for Gymnasium.
"""

import gymnasium as gym
from .surgical_epigraph_env import SurgicalEpigraphEnv
from .surgical_epigraph_env_cfg import SurgicalEpigraphEnvCfg

# Register Epigraph environment
gym.register(
    id="Isaac-Surgical-MARL-Epigraph-v0",
    entry_point="surgical_project.envs.multi_agent_epigraph.surgical_epigraph_env:SurgicalEpigraphEnv",
    disable_env_checker=True,
)

__all__ = ["SurgicalEpigraphEnv", "SurgicalEpigraphEnvCfg"]