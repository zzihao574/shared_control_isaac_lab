# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Surgical Direct MARL Environment with Human-Robot Collaboration."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class SurgicalDirectMARLEnvCfg(DirectMARLEnvCfg):
    """Configuration for Surgical Direct MARL Environment."""
    
    # Environment settings
    episode_length_s = 10.0  # 10 seconds per episode
    decimation = 2  # Control frequency decimation
    
    # Multi-agent settings
    possible_agents = ["human", "robot"]  # Two agents: human and robot
    
    # Agent-specific action and observation spaces
    action_spaces = {
        "human": 3,   # xyz forces from human (haptic device)
        "robot": 3,   # xyz forces from robot (control policy)
    }
    
    observation_spaces = {
        "human": 21,   # Updated: actual observation dimension  
        "robot": 21,   # Updated: actual observation dimension
    }
    
    # Global state space (centralized training)
    state_space = 30  # Updated: combined state information for centralized critic
    
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,  # 120 Hz simulation
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply", 
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.1,
        ),
    )
    
    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,  # Reduced for MARL training
        env_spacing=3.0, 
        replicate_physics=True
    )
    
    # Scalpel (controlled by both agents)
    scalpel = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Scalpel",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/zzh/workspace/surgical_robot_project/assets/models/usd/scalpei_simple.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,  # No gravity
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.15),  # Start 150mm above constraint center
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )
    
    # Constraint (cone-shaped)
    constraint = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Constraint", 
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/zzh/workspace/surgical_robot_project/assets/models/usd/ConeConstraint.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Cannot be moved
                disable_gravity=True,
            ),
            scale=(0.05, 0.05, 0.05),  # Scale down by 0.05
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),  # At world origin
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # Ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0, 
            restitution=0.0,
        ),
    )
    
    # Force and control parameters
    max_force = {
        "human": 8.0,   # Human can apply higher force (haptic feedback)
        "robot": 5.0,   # Robot has controlled force limit
    }
    force_scale = 1.0
    
    # Agent interaction parameters
    interaction_coupling = 0.3  # How much agents influence each other
    force_sharing_ratio = {
        "human": 0.6,   # Human contributes 60% of total force
        "robot": 0.4,   # Robot contributes 40% of total force
    }
    
    # Reward scales (agent-specific)
    reward_scales = {
        "human": {
            "collision_penalty": -8.0,
            "distance_reward": 1.2,
            "smoothness_penalty": -0.05,
            "collaboration_reward": 3.0,
            "task_completion": 60.0,
            "intention_alignment": 2.0,
        },
        "robot": {
            "collision_penalty": -10.0,
            "distance_reward": 1.0,
            "smoothness_penalty": -0.1,
            "collaboration_reward": 2.5,
            "task_completion": 50.0,
            "adaptation_reward": 1.5,
        }
    }
    
    # Simulation scaling factor
    simulation_scale = 10.0  # Simulation is 10x larger than real world
    
    # Task parameters (in simulation scale)
    target_height = 0.05  # Target height inside constraint
    collision_threshold = 0.001  # Collision detection threshold
    
    # Constraint geometry parameters (in simulation scale, after 0.05 scaling)
    constraint_inner_radius_min = 0.1   # 100mm in sim = 10mm real
    constraint_inner_radius_max = 0.25  # 250mm in sim = 25mm real
    constraint_outer_radius_min = 0.15  # 150mm in sim = 15mm real
    constraint_outer_radius_max = 0.3   # 300mm in sim = 30mm real
    constraint_height = 0.2598          # 259.8mm in sim = 25.98mm real
    
    # Human dynamics parameters (for simulation)
    human_dynamics = {
        "stiffness": 201.0,
        "damping": 21.0,
        "noise_std": 0.1,          # Human input noise
        "reaction_delay": 0.05,    # 50ms reaction delay
        "intention_update_rate": 10.0,  # Hz
    }
    
    # Robot dynamics parameters
    robot_dynamics = {
        "control_frequency": 50.0,  # Hz
        "adaptation_rate": 0.1,
        "learning_rate": 0.001,
    }
    
    # Collaboration parameters
    collaboration = {
        "trust_factor": 0.8,        # Initial trust between agents
        "trust_decay": 0.95,        # Trust decay on conflicts
        "trust_recovery": 1.02,     # Trust recovery on success
        "conflict_threshold": 2.0,  # Force difference threshold for conflict
        "alignment_bonus": 5.0,     # Reward bonus for aligned actions
    }
    
    # Task completion criteria
    task_completion = {
        "target_radius": 0.01,      # 10mm tolerance around target
        "completion_time": 50,      # Steps to stay in target (1 second)
        "max_completion_bonus": 100.0,
        "partial_completion_steps": [10, 25, 40],  # Progressive rewards
        "partial_completion_rewards": [10.0, 25.0, 40.0],
    }
    
    # Observation noise models (optional)
    observation_noise_model = {
        "human": None,  # Human observations can have noise
        "robot": None,  # Robot observations are precise
    }
    
    # Action noise models (for realism)
    action_noise_model = {
        "human": None,  # Human actions have inherent noise
        "robot": None,  # Robot actions are precise
    }