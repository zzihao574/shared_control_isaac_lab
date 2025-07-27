# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Paper-aligned surgical direct environment configuration - including Omni haptic device"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import AssetBaseCfg


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Simplified scene configuration containing only basic elements"""
    
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=5000.0),
    )


@configclass
class SurgicalDirectEnvCfg(DirectRLEnvCfg):
    """Paper-aligned surgical direct environment configuration - using Omni haptic device"""
    
    # Environment settings
    episode_length_s = 8.0  
    decimation = 2  
    action_space = 3  # xyz forces in Cartesian space for robot
    observation_space = 12  # Simplified observation space: position(3) + velocity(3) + target position(3) + padding(3)
    state_space = 9  # Paper standard state: z = [x, ẋ, f]^T ∈ R^9
    
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
    scene: InteractiveSceneCfg = MySceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    
    # Omni haptic device
    phantom_omni = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=100,
                solver_velocity_iteration_count=100,
                sleep_threshold=0.01,
                stabilization_threshold=0.01,
                fix_root_link=True
            ),
            scale=(1, 1, 1),
            asset_path='/home/zzh/workspace/surgical_robot_project/assets/models/urdf/omni.urdf',
            usd_dir='/home/zzh/workspace/surgical_robot_project/assets/models/usd',
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness={
                        "waist": 1,    
                        "shoulder": 1,
                        "elbow": 1,
                        "yaw": 10000,      
                        "pitch": 10000,    
                        "roll": 10000,     
                    }, 
                    damping={
                        "waist": 0.1,
                        "shoulder": 0.1,
                        "elbow": 0.1,
                        "yaw": 1000,       
                        "pitch": 1000,     
                        "roll": 1000,      
                    },
                ),
                drive_type="acceleration",
                target_type="position"
            ),
            fix_base=True,
            root_link_name='base',
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(1, 0.0, 0.0, 0.0),
            joint_pos={
                "waist": 0.0,
                "shoulder": 0.0,
                "elbow": 0.0,
                "yaw": 4.0,
                "pitch": 1.2,
                "roll": 0.0,
            },
            joint_vel={
                "waist": 0.0,
                "shoulder": 0.0,
                "elbow": 0.0,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
            },
        ),
        actuators={
            "arm_joints": IdealPDActuatorCfg(
                joint_names_expr=[
                    "waist", 
                    "shoulder", 
                    "elbow", 
                    "yaw", 
                    "pitch", 
                    "roll"
                ],
                stiffness={
                    "waist": 1,    
                    "shoulder": 1,
                    "elbow": 1,
                    "yaw": 10000,
                    "pitch": 10000,
                    "roll": 10000,
                },
                damping={
                    "waist": 0.1,
                    "shoulder": 0.1,
                    "elbow": 0.1,
                    "yaw": 1000,
                    "pitch": 1000,
                    "roll": 1000,
                },
            ),
        },
    )

    # Constraint configuration
    constraint = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Constraint", 
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/zzh/workspace/surgical_robot_project/assets/models/usd/ConeConstraint.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            scale=(0.01, 0.01, 0.015),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.15, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # Force and control parameters
    max_robot_force = 3.3
    max_human_force = 3.3
    force_scale = 1.0  
    
    # End effector parameters
    end_effector_body_name = "stylus"
    end_effector_body_id = 6
    
    # Trajectory parameters - distance-based target switching
    target_reach_threshold = 0.01  # Target point reach threshold
    
    # Physics simulation parameters
    physics_dt = 1/120
    render_dt = 1/60
    solver_iterations = 4
    solver_velocity_iterations = 1
    use_gpu_physics = True