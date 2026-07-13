# surgical_direct_marl_env_cfg.py - Clean version after removing duplicate definitions
# PHASE 1: No changes needed - compatible with Phase 1 modifications

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Scene configuration for surgical robot training environment."""
    
    # Ground plane for physics simulation
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # Dome light for scene illumination
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class SurgicalDirectMARLEnvCfg(DirectMARLEnvCfg):
    """Configuration for Surgical Direct MARL Environment."""
    
    # Episode configuration
    episode_length_s = 20  # Episode length in seconds
    decimation = 2         # Control decimation factor
    
    # Multi-agent configuration (single source of truth)
    possible_agents = ["human", "robot"]
    
    action_spaces = {
        "human": 3,  # 3D force control
        "robot": 3,  # 3D force control
    }
    
    observation_spaces = {
        "human": 6, # Corrected to actual dimensions
        "robot": 6, # Corrected to actual dimensions
    }
    
    state_space = 38

    # Physics simulation configuration
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,  # 120 Hz simulation frequency
        render_interval=decimation,
        gravity=(0.0, 0.0, 0.0),  # Zero gravity for surgical environment
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply", 
            static_friction=0.001,   # Low friction for smooth motion
            dynamic_friction=0.001,  # Low friction for smooth motion
            restitution=0.1,         # Low restitution for stable contact
        ),
    )
    
    # Scene configuration with multiple environments (count can be overridden from command line)
    scene: InteractiveSceneCfg = MySceneCfg(
        num_envs=512,      # Default value, can be overridden by --num_envs in train_maddpg.py
        env_spacing=1.0,   # Spacing between environments
        replicate_physics=True
    )
    
    # Phantom Omni robot configuration
    phantom_omni = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/zzh/workspace/shared_control_isaac_sim/assets/models/usd/omni.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_linear_velocity=0.06,   # Limited for safety
                max_angular_velocity=50.0,  # Limited for safety
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,  # High precision
                solver_velocity_iteration_count=8,   # Stable dynamics
                sleep_threshold=0.0,
                stabilization_threshold=0.0,
                fix_root_link=True  # Fixed base robot
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(1, 0.0, 0.0, 0.0),
            # Initial joint positions for stable starting pose
            joint_pos={
                "waist": -0.96,
                "shoulder": 0.0,
                "elbow": 1.0,
                "yaw": 0.0,
                "pitch": 2.0944,  # ~120 degrees
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
        # Individual actuator configuration for each joint
        # Note: effort_limit here is different from YAML force constraints
        # effort_limit is joint torque limit, max_robot_force is end-effector force limit
        actuators={
            "waist_actuator": ImplicitActuatorCfg(
                joint_names_expr=["waist"],
                effort_limit=5.0,   # Joint torque limit (Nm)
                stiffness=0.0,      # Zero stiffness for force control
                damping=0.0,        # Zero damping for free motion
            ),
            "shoulder_actuator": ImplicitActuatorCfg(
                joint_names_expr=["shoulder"],
                effort_limit=5.0,
                stiffness=0.0,
                damping=0.0,
            ),
            "elbow_actuator": ImplicitActuatorCfg(
                joint_names_expr=["elbow"],
                effort_limit=5.0,
                stiffness=0.0,
                damping=0.0,
            ),
            "yaw_actuator": ImplicitActuatorCfg(
                joint_names_expr=["yaw"],
                effort_limit=5.0,
                stiffness=0.0,
                damping=0.0,
            ),
            "pitch_actuator": ImplicitActuatorCfg(
                joint_names_expr=["pitch"],
                effort_limit=5.0,
                stiffness=0.0,
                damping=0.0,
            ),
            "roll_actuator": ImplicitActuatorCfg(
                joint_names_expr=["roll"],
                effort_limit=5.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
    
    # Constraint object for collision detection and trajectory guidance
    constraint = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Constraint", 
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/zzh/workspace/shared_control_isaac_sim/assets/models/usd/sphere.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,   # Kinematic object (doesn't respond to forces)
                disable_gravity=True,     # No gravity effect
            ),
            scale=(1, 1, 1),   # Small scale for precise constraints
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.14, 0.0, 0.0),         # Position relative to robot base
            rot=(1.0, 0.0, 0.0, 0.0),     # Default orientation
        ),
    )