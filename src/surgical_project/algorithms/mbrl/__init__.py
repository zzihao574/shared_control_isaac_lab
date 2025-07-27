# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Model-based reinforcement learning algorithms - paper-aligned version, integrating CBF constraints"""

# Import from actor_critic.py
from .actor_critic import (
    SurgicalActor,
    SurgicalCritic,
    DynamicsIdentifierNetwork,
    SurgicalActorCritic,
)

# Import from shared_control.py
from .shared_control import (
    SharedControlTrainer,
)

__all__ = [
    # Actor-Critic network components
    "SurgicalActor",
    "SurgicalCritic", 
    "DynamicsIdentifierNetwork",
    "SurgicalActorCritic",
    # Main trainer
    "SharedControlTrainer",
]