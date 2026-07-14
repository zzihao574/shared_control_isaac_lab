# surgical_epigraph_env_cfg.py
# Configuration for Epigraph environment - identical to direct_marl config
# This ensures environment physics and structure remain unchanged

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


ASSET_USD_DIR = Path(__file__).resolve().parents[4] / "assets" / "models" / "usd"


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
class SurgicalEpigraphEnvCfg(DirectMARLEnvCfg):
    """Configuration for Surgical Epigraph Environment - identical to Direct MARL config."""
    
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
        "human": 6,  # Position + velocity
        "robot": 6,  # Position + velocity
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
    
    # Scene configuration with multiple environments
    scene: InteractiveSceneCfg = MySceneCfg(
        num_envs=512,      # Default value, can be overridden
        env_spacing=1.0,   # Spacing between environments
        replicate_physics=True
    )
    
    # Phantom Omni robot configuration
    phantom_omni = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSET_USD_DIR / "omni.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_linear_velocity=0.06,
                max_angular_velocity=50.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
                sleep_threshold=0.0,
                stabilization_threshold=0.0,
                fix_root_link=True
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(1, 0.0, 0.0, 0.0),
            joint_pos={
                "waist": -0.96,
                "shoulder": 0.0,
                "elbow": 1.0,
                "yaw": 0.0,
                "pitch": 2.0944,
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
            "waist_actuator": ImplicitActuatorCfg(
                joint_names_expr=["waist"],
                effort_limit=5.0,
                stiffness=0.0,
                damping=0.0,
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
            usd_path=str(ASSET_USD_DIR / "cube.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            scale=(1, 1, 1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.14, -0.02, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
