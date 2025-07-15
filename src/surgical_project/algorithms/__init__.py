# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Algorithms for surgical robot training.

This package contains both single-agent (MBRL) and multi-agent (MARL) algorithms
for surgical robot training with human-robot collaboration.
"""

# Import single-agent algorithms from mbrl subpackage
from .mbrl import (
    SurgicalActor,
    SurgicalCritic,
    DynamicsIdentifierNetwork,
    SurgicalActorCritic,
    SharedControlTrainer,
    HumanDynamicsModel,
    ReplayBuffer,
)

# Import multi-agent algorithms from marl subpackage
from .marl import (
    MADDPGTrainer,
    MADDPGAgent,
    Actor,
    Critic,
    ReplayBuffer as MARLReplayBuffer,
)

__all__ = [
    # Single-agent (MBRL) algorithms
    "SurgicalActor",
    "SurgicalCritic",
    "DynamicsIdentifierNetwork",
    "SurgicalActorCritic", 
    "SharedControlTrainer",
    "HumanDynamicsModel",
    "ReplayBuffer",
    # Multi-agent (MADDPG) algorithms
    "MADDPGTrainer",
    "MADDPGAgent",
    "Actor",
    "Critic", 
    "MARLReplayBuffer",
]