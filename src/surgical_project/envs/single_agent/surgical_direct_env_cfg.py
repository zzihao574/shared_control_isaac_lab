# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Surgical Direct Environment with Human-Robot Shared Control."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class SurgicalDirectEnvCfg(DirectRLEnvCfg):
    """Configuration for Surgical Direct Environment."""
    
    # Environment settings
    episode_length_s = 10.0  # 10 seconds per episode
    decimation = 2  # Control frequency decimation
    action_space = 3  # xyz forces only
    observation_space = 15  # Position(3) + Velocity(3) + Constraint distances(6) + Human intent(3)
    state_space = 0
    
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
        num_envs=1024, 
        env_spacing=3.0, 
        replicate_physics=True
    )
    
    # Scalpel (represented as sphere)
    scalpel = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Scalpel",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/zzh/workspace/surgical_robot_project/assets/models/usd/scalpei_simple.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,  # No gravity as requested
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.15),  # Start 150mm above constraint center
            rot=(1.0, 0.0, 0.0, 0.0),  # No rotation
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
            # Scale down by 0.05 as requested
            scale=(0.05, 0.05, 0.05),
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
    max_force = 5.0  # Maximum force that can be applied (N)
    force_scale = 1.0  # Force scaling factor
    
    # Reward scales
    collision_penalty_scale = -10.0
    distance_reward_scale = 1.0
    smoothness_reward_scale = -0.1
    human_collaboration_scale = 2.0
    task_completion_scale = 50.0
    
    # Simulation scaling factor - simulation is 10x larger than real world
    simulation_scale = 10.0  # Simulation to real world scale factor
    
    # Task parameters (in simulation scale, 10x real world)
    target_height = 0.05  # Target height inside constraint (50mm in sim = 5mm real)
    collision_threshold = 0.001  # Collision detection threshold (1mm in sim = 0.1mm real)
    
    # Constraint geometry parameters (in simulation scale, after 0.05 scaling)
    constraint_inner_radius_min = 0.1  # 100mm in sim = 10mm real
    constraint_inner_radius_max = 0.25  # 250mm in sim = 25mm real
    constraint_outer_radius_min = 0.15  # 150mm in sim = 15mm real
    constraint_outer_radius_max = 0.3  # 300mm in sim = 30mm real
    constraint_height = 0.2598  # 259.8mm in sim = 25.98mm real
    
    # Human dynamics model parameters
    human_stiffness = 201.0
    human_damping = 21.0
    interaction_force_threshold = 0.1