# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simplified Configuration for Surgical Direct Environment."""

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
    """Simplified Configuration for Surgical Direct Environment."""
    
    # Environment settings
    episode_length_s = 8.0  # 8 seconds per episode
    decimation = 2  # Control frequency decimation
    action_space = 3  # xyz forces only
    observation_space = 19  # Simplified: pos(3) + vel(3) + target_pos(3) + target_vel(3) + constraint(5) + workspace(2)
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
        num_envs=512, 
        env_spacing=2.0, 
        replicate_physics=True
    )
    
    # Scalpel (represented as sphere) - real-world scale
    scalpel = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Scalpel",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/zzh/workspace/surgical_robot_project/assets/models/usd/scalpel_simple.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,  # No gravity
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.015),  # Start 15mm above constraint center
            rot=(1.0, 0.0, 0.0, 0.0),  # No rotation
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )
    
    # Constraint - real-world scale
    constraint = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Constraint", 
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/zzh/workspace/surgical_robot_project/assets/models/usd/ConeConstraint.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Cannot be moved
                disable_gravity=True,
            ),
            scale=(0.017, 0.017, 0.0125),
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
    
    # Force and control parameters - real-world scale
    max_force = 1.0  # Reduced maximum force (N)
    force_scale = 1.0  # Force scaling factor
    
    # Reward scales (balanced for reasonable rewards)
    collision_penalty_scale = -5.0       # Moderate collision penalty
    distance_reward_scale = 2.0
    smoothness_reward_scale = 0.5        # Positive smoothness reward
    human_collaboration_scale = 1.0      # Moderate collaboration reward
    task_completion_scale = 10.0         # Moderate completion reward
    
    # Task parameters (real-world scale)
    target_height = 0.005  # Target height inside constraint (5mm)
    collision_threshold = 0.0001  # Collision detection threshold (0.1mm)
    
    # Constraint geometry parameters (real-world scale)
    constraint_inner_radius_min = 0.01   # 10mm (thin end radius)
    constraint_inner_radius_max = 0.05   # 50mm (thick end radius, 10cm diameter)
    constraint_outer_radius_min = 0.015  # 15mm 
    constraint_outer_radius_max = 0.055  # 55mm
    constraint_height = 0.1              # 100mm (10cm height)
    
    # Human dynamics model parameters (aligned with paper Eq. 6)
    human_stiffness = 201.0
    human_damping = 21.0
    interaction_force_threshold = 0.1
    
    # Human workspace parameters
    human_workspace_radius = 0.2         # 20cm radius around origin
    human_max_velocity = 0.3             # 30cm/s maximum human velocity
    human_intention_weight = 0.3         # Weight for human intention in action fusion
    
    # Human-robot collaboration parameters (aligned with paper)
    robot_action_weight = 0.7            # 70% robot weight in action fusion
    human_action_weight = 0.3            # 30% human weight in action fusion
    collaboration_adaptation_rate = 0.05 # Rate of adaptation for collaboration weights
    
    # Trajectory parameters
    trajectory_update_frequency = 10     # Update trajectory target every 10 steps
    trajectory_completion_threshold = 0.005  # 5mm threshold for trajectory point completion
    spiral_trajectory_turns = 2.0        # Number of turns in spiral trajectory
    spiral_radius_range = (0.01, 0.04)   # 1-4cm radius range for spiral
    spiral_height_range = (0.002, 0.08)  # 2-80mm height range for spiral
    trajectory_points_count = 200        # Number of points in trajectory
    
    # Safety parameters
    max_constraint_distance = 0.02       # 20mm maximum allowed distance from constraint
    emergency_stop_distance = 0.001      # 1mm emergency stop distance
    collision_recovery_force = 1.0       # 1N force for collision recovery
    sphere_radius = 0.002                # 2mm sphere radius for collision detection
    
    # Action scaling parameters
    action_velocity_scale = 0.1           # Scale actions to 10cm/s max velocity
    action_force_scale = 1.0             # Direct force scaling
    max_action_norm = 0.1                # Maximum action norm (10cm/s)
    
    # Observation normalization parameters
    position_normalization_range = 0.3   # ±30cm normalization range
    velocity_normalization_range = 0.5   # ±50cm/s normalization range
    force_normalization_range = 5.0      # ±5N normalization range
    observation_clamp_range = 10.0       # Clamp observations to ±10
    
    # Environment reset parameters
    reset_position_noise = 0.005         # ±5mm position noise on reset
    reset_velocity_noise = 0.002         # ±2mm/s velocity noise on reset
    reset_within_workspace = True        # Always reset within human workspace
    reset_workspace_radius_range = (0.02, 0.15)  # 2-15cm from workspace center
    
    # Physics simulation parameters
    physics_dt = 1/120                   # 120Hz physics simulation
    render_dt = 1/60                     # 60Hz rendering
    solver_iterations = 4                # Physics solver iterations
    solver_velocity_iterations = 1       # Velocity solver iterations
    use_gpu_physics = True               # Enable GPU physics if available