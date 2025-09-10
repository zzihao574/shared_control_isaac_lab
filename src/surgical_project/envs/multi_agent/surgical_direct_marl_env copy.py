# surgical_direct_marl_env.py - Modified for unified progress management and MetricsHub
# NEW: Four-zone reward system with A/B/C/D regions (preserving Isaac Lab method names)

from __future__ import annotations

import torch
import numpy as np
import yaml
import os
import gymnasium as gym  
from typing import Any, Dict, List, Optional
from collections import defaultdict

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.math import sample_uniform, quat_rotate_inverse

from .surgical_direct_marl_env_cfg import SurgicalDirectMARLEnvCfg
from .utils import CompleteConstraintChecker, TrajectoryManager, RewardLogger


class SurgicalDirectMARLEnv(DirectMARLEnv):
    """
    Human-robot collaborative surgical MARL environment.
    MODIFIED: NEW Four-zone reward system (A/B/C/D) with robot/human symmetric rewards
    
    Features:
    - Multi-agent force control for surgical tasks
    - Physics-based constraint checking and collision detection
    - Four-zone reward system: Track/Surface/Danger/Rejoin
    - Comprehensive reward system with safety considerations
    - Performance evaluation and milestone tracking
    - Console logging via utils (configured by YAML)
    """
    
    cfg: SurgicalDirectMARLEnvCfg
    
    def __init__(self, cfg: SurgicalDirectMARLEnvCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize surgical MARL environment.
        
        Args:
            cfg: Environment configuration
            render_mode: Rendering mode (optional)
            **kwargs: Additional arguments
        """
        super().__init__(cfg, render_mode, **kwargs)
        
        # Progress manager will be injected by trainer
        self.progress_manager = None
        
        # Load configuration and initialize core parameters
        self._setup_core_configuration()
        
        # Initialize state variables and caches
        self._initialize_state_variables()
        
        # Initialize utility managers and components
        self._initialize_components()
        
        # Restore gymnasium spaces setup - both Isaac Lab and Gymnasium need this
        self._setup_gymnasium_spaces()
        
    def set_progress_manager(self, pm):
        """Inject progress manager reference."""
        self.progress_manager = pm
        
    def _setup_core_configuration(self) -> None:
        """Load and setup core configuration parameters."""
        # If trainer already injected self.params, use it directly; otherwise fallback to local YAML
        if not hasattr(self, "params") or not isinstance(getattr(self, "params", None), dict):
            self.params = self._load_training_params()
            
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
        # Load and cache constraint parameters
        self._load_constraint_parameters()
        
        # Load safety and CBF parameters  
        self._load_safety_parameters()
        
        # Load termination conditions
        self._load_termination_parameters()
        
        print(f"[INFO] Episode length controlled by cfg.episode_length_s: {self.cfg.episode_length_s}s")
    
    def _load_training_params(self) -> dict:
        """Load training parameters from YAML configuration."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "agents", "training_params.yaml")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_constraint_parameters(self) -> None:
        """Load and cache constraint-related parameters."""
        constraints = self.params['constraints']
        
        # Position and force limits
        self.min_z_pos = constraints['min_z_position']
        self.max_robot_force = constraints['max_robot_force']
        self.max_human_force = constraints['max_human_force']
        
        # Joint limits as tensors for efficient operations
        joint_limits = constraints['joint_limits']
        self.joint_lower_limits = torch.tensor([
            joint_limits['waist'][0], joint_limits['shoulder'][0], joint_limits['elbow'][0],
            joint_limits['yaw'][0], joint_limits['pitch'][0], joint_limits['roll'][0]
        ], device=self.device, dtype=torch.float32)
        
        self.joint_upper_limits = torch.tensor([
            joint_limits['waist'][1], joint_limits['shoulder'][1], joint_limits['elbow'][1],
            joint_limits['yaw'][1], joint_limits['pitch'][1], joint_limits['roll'][1]
        ], device=self.device, dtype=torch.float32)
        
        # Fixed end joints configuration
        self.fixed_end_joints = torch.tensor([
            self.params['initial_conditions']['joint_positions']['yaw'],
            self.params['initial_conditions']['joint_positions']['pitch'],
            self.params['initial_conditions']['joint_positions']['roll']
        ], device=self.device, dtype=torch.float32)
    
    def _load_safety_parameters(self) -> None:
        """Load safety-related parameters."""        
        self.collision_threshold = self.params['constraint_geometry']['collision_threshold']
    
    def _load_termination_parameters(self) -> None:
        """Load episode termination condition parameters."""
        term_config = self.params.get('termination_conditions', {})
        self.enable_z_termination = term_config.get('z_below_zero', True)
        self.enable_edge_termination = term_config.get('edge_collision', True)
        self.safety_distance_threshold = term_config.get('safety_distance_threshold', 0.001)
    
    def _initialize_state_variables(self) -> None:
        """Initialize all state variables and caches."""
        # Environment base positions
        self.env_base_positions = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Agent actions storage
        self.agent_actions = {
            agent: torch.zeros(self.num_envs, 3, device=self.device)
            for agent in self.cfg.possible_agents
        }
        
        # Physics interaction state (forces applied at time t)
        self._initialize_physics_state()
        
        # State cache (will be updated in _get_observations)
        self._initialize_observation_cache()
        
        # NEW: Four-zone reward system state caches
        self._initialize_reward_state_caches()
        
        # Reward components cache
        self.reward_components = {}
        
        # Body index for stylus/end-effector (will be set during scene setup)
        self.stylus_body_idx = None
    
    def _initialize_physics_state(self) -> None:
        """Initialize physics interaction state variables (forces at time t)."""
        self.human_forces_t = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_forces_t = torch.zeros(self.num_envs, 3, device=self.device)
    
    def _initialize_observation_cache(self) -> None:
        """Initialize state caching variables (updated in _get_observations)."""
        self.stylus_pos_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.stylus_vel_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.safety_distances_t1 = torch.ones(self.num_envs, device=self.device) * 0.01
        self.is_violating_t1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.normal_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.constraint_results_t1 = None

    def _initialize_reward_state_caches(self) -> None:
        """NEW: Initialize four-zone reward system state caches."""
        # Progress/distance caches for delta calculations
        self.prev_safety_distances = torch.ones(self.num_envs, device=self.device) * 0.02
        self.prev_progress = torch.zeros(self.num_envs, device=self.device)
        self.best_progress = torch.zeros(self.num_envs, device=self.device)
        
        # Rejoin zone "10-step stability gate"
        self.rejoin_streak = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        
        # Goal point for rejoin zone calculations
        self._goal_point = None
    
    def _initialize_components(self) -> None:
        """Initialize utility managers and components."""
        # Trajectory manager for path following
        self.trajectory_manager = TrajectoryManager(
            device=self.device,
            params=self.params,
            num_envs=self.num_envs,
            env_base_positions=self.env_base_positions
        )
        
        # NEW: Initialize goal point after trajectory manager is ready
        t = self.trajectory_manager.line_direction
        if hasattr(self.trajectory_manager, "end_pos_local"):
            self._goal_point = self.trajectory_manager.end_pos_local.to(self.device)
        elif 'trajectory' in self.params and 'end_point' in self.params['trajectory']:
            self._goal_point = torch.tensor(self.params['trajectory']['end_point'], device=self.device, dtype=t.dtype)
        else:
            # Fallback: use trajectory direction
            self._goal_point = self.trajectory_manager.start_pos_local + t * self.trajectory_manager.total_distance
            
        print(f"[INFO] Goal point initialized: {self._goal_point}")
        
        # Reward logger with milestone tracking
        self.reward_logger = RewardLogger(self.num_envs, self.device)
        
        # Constraint checker for collision detection
        self.constraint_checker = CompleteConstraintChecker(
            device=self.device, 
            collision_threshold=self.collision_threshold
        )
    
    def _setup_gymnasium_spaces(self) -> None:
        """Setup Gymnasium compatibility spaces with unlimited bounds."""
        # Use unlimited bounds, actual limits handled by environment logic
        self.action_space = gym.spaces.Dict({
            agent: gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.cfg.action_spaces[agent],), 
                dtype=np.float32
            ) for agent in self.cfg.possible_agents
        })
        
        self.observation_space = gym.spaces.Dict({
            agent: gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.cfg.observation_spaces[agent],), 
                dtype=np.float32
            ) for agent in self.cfg.possible_agents
        })
        
    def _setup_scene(self):
        """Setup the simulation scene with robot and constraints."""
        # Initialize robot articulation
        self._omni_robot = Articulation(self.cfg.phantom_omni)
        self.scene.articulations["phantom_omni"] = self._omni_robot
        
        # Initialize constraint object
        self._constraint = RigidObject(self.cfg.constraint)
        self.scene.rigid_objects["constraint"] = self._constraint
        
        # Clone environments for parallel simulation
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    def _setup_post_scene_creation(self):
        """Post-scene creation setup including physics configuration."""
        super()._setup_post_scene_creation()
        
        # Initialize robot-specific configurations
        self._initialize_body_indices()

        # Update managers with robot data
        self._update_managers_with_robot_data()

    def _update_managers_with_robot_data(self) -> None:
        """Update managers with robot position data."""
        if hasattr(self, '_omni_robot'):
            # Update environment base positions from robot data
            self.env_base_positions = self._omni_robot.data.root_link_pos_w.clone()
            self.trajectory_manager.env_base_positions = self.env_base_positions
            
    def _initialize_body_indices(self):
        """Initialize body indices for end-effector/stylus identification."""
        if self.stylus_body_idx is not None:
            return  # Already cached
            
        if not hasattr(self._omni_robot, 'body_names'):
            return
        
        # Hard-coded stylus body name for performance
        target_name = "stylus"
        
        for i, name in enumerate(self._omni_robot.body_names):
            if target_name in name.lower():
                self.stylus_body_idx = i
                print(f"[INFO] Found stylus body: {name} (index {i})")
                return
        
        # Fallback to last body if stylus not found
        if len(self._omni_robot.body_names) > 0:
            self.stylus_body_idx = len(self._omni_robot.body_names) - 1
            print(f"[INFO] Using fallback stylus body index: {self.stylus_body_idx}")
        
    def _pre_physics_step(self, actions: Dict[str, torch.Tensor]) -> None:
        """Pre-physics step processing including action validation and force application."""
        # Process and validate actions - expect (num_envs, 3) from trainer
        for agent, action in actions.items():
            if agent in self.cfg.possible_agents:
                # Assert correct shape - no dimension handling here
                assert action.shape == (self.num_envs, 3), f"Action shape mismatch for {agent}: expected ({self.num_envs}, 3), got {action.shape}"
                
                # Apply force constraints
                max_force = self.max_robot_force if agent == "robot" else self.max_human_force
                self.agent_actions[agent] = torch.clamp(action, -max_force, max_force)
        
        # Cache forces for reward calculation (these are at time t)
        self.robot_forces_t = self.agent_actions["robot"]
        self.human_forces_t = self.agent_actions["human"]
        
        # Apply external forces to end-effector
        self._apply_external_forces()
        
        # Enforce joint constraints and fix end joints
        self._enforce_joint_constraints()
        
    def _apply_external_forces(self) -> None:
        """Apply external forces to the robot end-effector."""
        total_forces = self.robot_forces_t + self.human_forces_t
        stylus_quat = self._omni_robot.data.body_link_quat_w[:, self.stylus_body_idx, :]
        
        # Transform forces to local frame
        forces_local = quat_rotate_inverse(stylus_quat, total_forces)
        
        # Reshape for Isaac Lab API
        forces_with_body_dim = forces_local.unsqueeze(1)
        torques_with_body_dim = torch.zeros_like(forces_with_body_dim)
        
        self._omni_robot.set_external_force_and_torque(
            forces_with_body_dim, 
            torques_with_body_dim,
            body_ids=[self.stylus_body_idx]
        )
        
    def _enforce_joint_constraints(self) -> None:
        """Enforce joint limits and fix end joints."""
        joint_pos = self._omni_robot.data.joint_pos.clone()
        joint_vel = self._omni_robot.data.joint_vel.clone()
        
        # Apply joint limits
        joint_pos = torch.clamp(joint_pos, self.joint_lower_limits, self.joint_upper_limits)
        
        # Fix end joints (wrist orientation)
        joint_pos[:, 3:6] = self.fixed_end_joints.unsqueeze(0).expand(self.num_envs, -1)
        joint_vel[:, 3:6] = 0.0
        
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel)
        
    def _apply_action(self) -> None:
        """Apply actions to simulation."""
        self._omni_robot.write_data_to_sim()

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Compute observations for all agents and update state cache."""
        # Update all state cache (t+1 state after physics)
        self.stylus_pos_t1 = self._get_stylus_position()
        self.stylus_vel_t1 = self._get_stylus_velocity()
        
        # Update constraint state (t+1 state after physics)
        current_base_positions = self._omni_robot.data.root_link_pos_w.clone()
        self.constraint_results_t1 = self.constraint_checker.analyze_constraint_state_batch(
            self.stylus_pos_t1, current_base_positions
        )
        self.safety_distances_t1 = self.constraint_results_t1['distances_constraint']
        self.is_violating_t1 = self.constraint_results_t1['is_overlapping']
        self.normal_t1 = self.constraint_results_t1['normal_vectors']

        # Ensure constraint distances have correct shape for concatenation
        constraint_distances = self.safety_distances_t1.unsqueeze(-1)  # [num_envs] -> [num_envs, 1]

        # Concatenate observation components (7 dimensions total)
        obs = torch.cat([
            self.stylus_pos_t1,       # End-effector position (3)
            self.stylus_vel_t1,       # End-effector velocity (3) 
            constraint_distances,     # Distance measurements (1)
        ], dim=-1)                    # Total: 7 dimensions
        
        # NEW: Update reward state caches at end of observation update
        s_t = self.trajectory_manager.get_progress(self.stylus_pos_t1)
        self.prev_safety_distances = self.safety_distances_t1.detach().clone()
        self.prev_progress = s_t.detach().clone()
        
        # Create observation dictionary for each agent
        observations = {}
        for agent in self.cfg.possible_agents:
            observations[agent] = obs
            
        return observations

    # =========================================================================
    # NEW: FOUR-ZONE REWARD SYSTEM
    # =========================================================================

    def _build_zone_masks(self):
        """NEW: Build zone masks and alpha mixing coefficient (mutually exclusive with stability gate)."""
        D, O = 0.0075, 0.015  # Danger threshold, Obstacle boundary
        d = self.safety_distances_t1
        
        # Basic zone masks
        surface = (d > D) & (d < O)
        outside = (d >= O) 
        danger = (d <= D)
        
        # Rejoin zone geometric conditions
        t = self.trajectory_manager.line_direction
        if self._goal_point is not None:
            goal_vec = self._goal_point.unsqueeze(0) - self.stylus_pos_t1  # [num_envs, 3]
            g = goal_vec / torch.norm(goal_vec, dim=-1, keepdim=True).clamp(min=1e-8)
        else:
            g = t.unsqueeze(0).expand_as(self.stylus_pos_t1)
        
        c1, c2 = 0.90, 0.60
        align_goal = (g * t.unsqueeze(0)).sum(dim=-1) >= c1      # Face towards goal
        oppose_norm = (self.normal_t1 * t.unsqueeze(0)).sum(dim=-1) <= -c2  # Normal opposes trajectory
        rejoin_geom = surface & align_goal & oppose_norm
        
        # 10-step stability gate for rejoin zone
        self.rejoin_streak = torch.where(
            rejoin_geom, 
            self.rejoin_streak + 1, 
            torch.zeros_like(self.rejoin_streak)
        )
        rejoin = self.rejoin_streak >= 10
        
        # Alpha mixing coefficient for zone D
        alpha = ((d - D) / (O - D)).clamp(0.0, 1.0)
        
        return outside, surface, danger, rejoin, alpha

    def _zweight(self, W: dict, zone: str, default: float = 1.0) -> float:
        """Get zone total weight with fallback."""
        z = W.get('zones', {}).get(zone, default)
        if isinstance(z, dict):
            return float(z.get('weight', default))
        return float(z)

    def _zcw(self, W: dict, zone: str, comp_key: str, default: float = 1.0) -> float:
        """Get zone-component weight with fallback to global components."""
        z = W.get('zones', {}).get(zone, {})
        if isinstance(z, dict):
            cz = z.get('components', {})
            if comp_key in cz:
                return float(cz[comp_key])
        return float(W.get('components', {}).get(comp_key, default))

    # =========================================================================
    # Component reward functions (raw scores with physical coefficients)
    # =========================================================================

    def _comp_progress(self):
        """Progress component: effective progress delta."""
        s_t = self.trajectory_manager.get_progress(self.stylus_pos_t1)
        baseline = torch.maximum(self.best_progress, self.prev_progress)
        delta = (s_t - baseline).clamp(min=0.0)
        kappa = 600.0
        raw = kappa * delta
        
        # Update best progress
        advanced = delta > 0
        if torch.any(advanced):
            self.best_progress[advanced] = torch.maximum(self.best_progress[advanced], s_t[advanced])
        return raw

    def _comp_deviation(self):
        """Deviation component: trajectory precision."""
        eta = 33.3
        deviations, _ = self.trajectory_manager.get_deviation(self.stylus_pos_t1)
        return eta * torch.clamp(0.025 - deviations, min=-0.05, max=0.015)

    def _comp_rejoin_speed(self):
        """Rejoin speed component: velocity along trajectory."""
        mu = 8.0
        t = self.trajectory_manager.line_direction
        v_t = (self.stylus_vel_t1 * t.unsqueeze(0)).sum(dim=-1)
        return mu * v_t

    def _comp_surface_gap(self):
        """Surface gap component: maintain optimal distance."""
        beta = 8.0e4
        d = self.safety_distances_t1
        return -beta * (d - 0.010) ** 2

    def _comp_surface_tangent(self):
        """Surface tangent component: move along surface."""
        gamma = 10.0
        t = self.trajectory_manager.line_direction
        n = self.normal_t1
        
        # Calculate surface tangent direction
        t_dot_n = (t.unsqueeze(0) * n).sum(dim=-1, keepdim=True)
        t_surf = t.unsqueeze(0) - t_dot_n * n
        t_surf = t_surf / torch.norm(t_surf, dim=-1, keepdim=True).clamp(min=1e-8)
        
        # Velocity along surface tangent
        v_surf = (self.stylus_vel_t1 * t_surf).sum(dim=-1)
        return gamma * v_surf

    def _comp_inward_penalty(self):
        """Inward penalty component: penalize moving into obstacle."""
        lam = 30.0
        v_in = (-(self.stylus_vel_t1 * self.normal_t1).sum(dim=-1)).clamp(min=0.0)
        return -lam * v_in  # Returns negative values

    # =========================================================================
    # Zone reward functions (mutually exclusive, only active in respective masks)
    # =========================================================================

    def _zone_track_reward(self, masks, W):
        """Zone A (Track): Outside obstacle boundary (d >= 1.5cm)."""
        outside, surface, danger, rejoin, alpha = masks
        
        prog = self._comp_progress()
        dev = self._comp_deviation()
        
        w_prog = self._zcw(W, 'track', 'progress')
        w_dev = self._zcw(W, 'track', 'deviation')
        
        raw = w_prog * prog + w_dev * dev
        out = torch.zeros_like(prog)
        out[outside] = raw[outside]
        
        zw = self._zweight(W, 'track')
        return zw * out, {
            'A_progress_w': w_prog * prog,
            'A_deviation_w': w_dev * dev
        }

    def _zone_surface_reward(self, masks, W):
        """Zone B (Surface): Near obstacle (0.75cm < d < 1.5cm), not in rejoin."""
        outside, surface, danger, rejoin, alpha = masks
        
        gap = self._comp_surface_gap()
        surf = self._comp_surface_tangent()
        inwd = self._comp_inward_penalty()
        
        w_gap = self._zcw(W, 'surface', 'gap')
        w_surf = self._zcw(W, 'surface', 'surf_tangent')
        w_in = self._zcw(W, 'surface', 'inward_penalty')
        
        raw = w_gap * gap + w_surf * surf + w_in * inwd
        out = torch.zeros_like(gap)
        surface_only = surface & (~rejoin)
        out[surface_only] = raw[surface_only]
        
        zw = self._zweight(W, 'surface')
        return zw * out, {
            'B_gap_w': w_gap * gap,
            'B_surf_tangent_w': w_surf * surf,
            'B_inward_w': w_in * inwd
        }

    def _zone_danger_reward(self, masks, W):
        """Zone C (Danger): Inside obstacle (d <= 0.75cm)."""
        outside, surface, danger, rejoin, alpha = masks
        
        R_off = -0.6
        inwd = self._comp_inward_penalty()
        
        w_off = self._zcw(W, 'danger', 'off_penalty')
        w_in = self._zcw(W, 'danger', 'inward_penalty')
        
        raw = w_off * (R_off * torch.ones_like(inwd)) + w_in * inwd
        out = torch.zeros_like(inwd)
        out[danger] = raw[danger]
        
        zw = self._zweight(W, 'danger')
        return zw * out, {
            'C_off_w': w_off * (R_off * torch.ones_like(inwd)),
            'C_inward_w': w_in * inwd
        }

    def _zone_rejoin_reward(self, masks, W):
        """Zone D (Rejoin): Transition zone with geometric conditions and 10-step gate."""
        outside, surface, danger, rejoin, alpha = masks
        
        prog = self._comp_progress()
        rspd = self._comp_rejoin_speed()
        dev = self._comp_deviation()
        gap = self._comp_surface_gap()
        inwd = self._comp_inward_penalty()
        
        # Alpha mixing
        dev_mix = alpha * dev
        gap_mix = (1.0 - alpha) * gap
        
        w_prog = self._zcw(W, 'rejoin', 'progress')
        w_rspd = self._zcw(W, 'rejoin', 'rejoin_speed')
        w_dev = self._zcw(W, 'rejoin', 'deviation')
        w_gap = self._zcw(W, 'rejoin', 'gap')
        w_in = self._zcw(W, 'rejoin', 'inward_penalty')
        
        raw = w_prog * prog + w_rspd * rspd + w_dev * dev_mix + w_gap * gap_mix + w_in * inwd
        out = torch.zeros_like(prog)
        out[rejoin] = raw[rejoin]
        
        zw = self._zweight(W, 'rejoin')
        return zw * out, {
            'D_progress_w': w_prog * prog,
            'D_rejoin_speed_w': w_rspd * rspd,
            'D_deviation_w': w_dev * dev_mix,
            'D_gap_w': w_gap * gap_mix,
            'D_inward_w': w_in * inwd,
        }

    # =========================================================================
    # Isaac Lab required method name: _get_rewards (CANNOT be changed)
    # =========================================================================

    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        """NEW: Four-zone reward system with robot/human symmetric rewards (Isaac Lab required method)."""
        # 1) Get reward weights
        robot_w = self.params['reward_parameters']['robot_weights']
        human_w = self.params['reward_parameters']['human_weights']

        # 2) Build zone masks and alpha
        masks = self._build_zone_masks()

        # 3) Calculate four-zone rewards for both agents
        A_r, A_parts_r = self._zone_track_reward(masks, robot_w)
        B_r, B_parts_r = self._zone_surface_reward(masks, robot_w)
        C_r, C_parts_r = self._zone_danger_reward(masks, robot_w)
        D_r, D_parts_r = self._zone_rejoin_reward(masks, robot_w)

        A_h, A_parts_h = self._zone_track_reward(masks, human_w)
        B_h, B_parts_h = self._zone_surface_reward(masks, human_w)
        C_h, C_parts_h = self._zone_danger_reward(masks, human_w)
        D_h, D_parts_h = self._zone_rejoin_reward(masks, human_w)

        # 4) Calculate global rewards (preserved from original system)
        force_pen = self._calculate_force_penalties()   # {'robot': tensor, 'human': tensor}
        z_pen = self._calculate_z_penalty()             # tensor[num_envs]
        completion = self._calculate_completion_reward()  # tensor[num_envs]
        time_eff = self._calculate_time_efficiency_reward()  # tensor[num_envs]

        # 5) Aggregate final rewards for each agent
        rewards = {}
        rewards["robot"] = (
            A_r + B_r + C_r + D_r
            + force_pen['robot'] * robot_w.get('force_efficiency', 0.0)
            + force_pen['human'] * robot_w.get('human_awareness', 0.0)
            + z_pen * robot_w.get('z_penalty', 0.0)
            + completion * robot_w.get('completion_reward', 0.0)
            + time_eff * robot_w.get('time_efficiency', 0.0)
        )
        rewards["human"] = (
            A_h + B_h + C_h + D_h
            + force_pen['human'] * human_w.get('force_efficiency', 0.0)
            + force_pen['robot'] * human_w.get('robot_awareness', 0.0)  # Note: robot_awareness for human
            + z_pen * human_w.get('z_penalty', 0.0)
            + completion * human_w.get('completion_reward', 0.0)
            + time_eff * human_w.get('time_efficiency', 0.0)
        )

        # 6) Calculate observables (required by update_step_metrics_batch)
        deviations, _ = self.trajectory_manager.get_deviation(self.stylus_pos_t1)
        progress_ratio = self.trajectory_manager.get_progress(self.stylus_pos_t1)
        distance_to_final = torch.norm(
            self.stylus_pos_t1 - self.trajectory_manager.end_pos_local.unsqueeze(0), dim=-1
        )

        # 7) Build comprehensive reward_components for logging (robot & human dual-copy)
        self.reward_components = {
            # ---- Zone totals (robot) ----
            'robot_A_track_total': A_r,
            'robot_B_surface_total': B_r,
            'robot_C_danger_total': C_r,
            'robot_D_rejoin_total': D_r,
            # ---- Zone totals (human) ----
            'human_A_track_total': A_h,
            'human_B_surface_total': B_h,
            'human_C_danger_total': C_h,
            'human_D_rejoin_total': D_h,

            # ---- Robot component breakdowns ----
            **{k.replace('_w', '_robot_w'): v for k, v in {**A_parts_r, **B_parts_r, **C_parts_r, **D_parts_r}.items()},
            # ---- Human component breakdowns ----
            **{k.replace('_w', '_human_w'): v for k, v in {**A_parts_h, **B_parts_h, **C_parts_h, **D_parts_h}.items()},

            # ---- Global components (raw values) ----
            'robot_force_penalty': force_pen['robot'],
            'human_force_penalty': force_pen['human'],
            'z_penalty': z_pen,
            'completion_reward': completion,
            'time_efficiency_reward': time_eff,

            # ---- Observables (required by update_step_metrics_batch) ----
            'deviation': deviations,
            'progress_ratio': progress_ratio,
            'distance_to_final': distance_to_final,
        }

        # 8) Logging and batch updates (unchanged interface)
        self.reward_logger.log_console_if_enabled(self, rewards, robot_w, human_w)
        self.reward_logger.update_step_metrics_batch(self.reward_components, self.safety_distances_t1, rewards)
        
        return rewards

    # =========================================================================
    # PRESERVED GLOBAL REWARD FUNCTIONS (unchanged)
    # =========================================================================

    def _calculate_force_penalties(self) -> Dict[str, torch.Tensor]:
        """Calculate force efficiency penalties - shared across all phases."""
        return {
            'robot': -50.0 * torch.norm(self.robot_forces_t, dim=-1),
            'human': -50.0 * torch.norm(self.human_forces_t, dim=-1)
        }

    def _calculate_z_penalty(self) -> torch.Tensor:
        """Calculate Z-axis constraint penalty - shared across all phases."""
        return torch.where(
            self.stylus_pos_t1[:, 2] < 0.0,
            -500.0 * torch.abs(self.stylus_pos_t1[:, 2]),
            torch.zeros_like(self.stylus_pos_t1[:, 2])
        )

    def _calculate_completion_reward(self) -> torch.Tensor:
        """Calculate task completion reward - shared across all phases."""
        is_final_reached = self.trajectory_manager.is_final_setpoint_reached(self.stylus_pos_t1)
        return torch.where(
            is_final_reached,
            torch.full_like(self.safety_distances_t1, 100.0),
            torch.zeros_like(self.safety_distances_t1)
        )

    def _calculate_time_efficiency_reward(self) -> torch.Tensor:
        """
        Calculate time efficiency reward using progress manager's step tracking.
        Encourages faster completion.
        """
        max_steps = 1200  # 20s * 60fps = 1200 steps per episode
        
        # Get step count from progress manager (float tensor)
        current_steps = self.progress_manager.step_counts.to(self.device).float()
        
        # Time efficiency: higher reward for completing tasks faster
        time_efficiency = (max_steps - current_steps) / max_steps
        time_efficiency = torch.clamp(time_efficiency, min=0.0, max=1.0)

        return time_efficiency * 3.0  # Scale time efficiency reward

    # =========================================================================
    # UNCHANGED METHODS (preserved from original)
    # =========================================================================

    def _get_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Determine termination and truncation conditions."""
        
        # Z-axis termination (safety)
        z_below_zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.enable_z_termination:
            z_below_zero = self.stylus_pos_t1[:, 2] < self.min_z_pos
        
        # Edge collision termination (safety) - MODIFIED for direct termination
        edge_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.enable_edge_termination:
            edge_collision = self.safety_distances_t1 <= self.safety_distance_threshold
        
        # Task completion
        final_reached = self.trajectory_manager.is_final_setpoint_reached(self.stylus_pos_t1)
        
        # Combine termination conditions
        terminated_condition = z_below_zero | edge_collision | final_reached

        # Time truncation - Isaac Lab manages independent episode_length_buf per environment
        truncated_condition = self.episode_length_buf >= self.max_episode_length - 1
        
        terminated = {agent: terminated_condition for agent in self.cfg.possible_agents}
        truncated = {agent: truncated_condition for agent in self.cfg.possible_agents}
        
        return terminated, truncated
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """MODIFIED: Reset specified environments and four-zone state caches."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Unified filtering: only episodes that actually ran can be settled
        valid_env_ids = self.progress_manager.filter_valid_for_episode_end(env_ids, min_steps=1)

        if valid_env_ids:
            # Fixed order: PM settlement first, then RewardLogger aggregation
            self.progress_manager.on_episode_end(valid_env_ids, reason="env_reset")
            self.reward_logger.on_episode_end(valid_env_ids)

        # Parent class reset + other initialization
        super()._reset_idx(env_ids)
        
        if self.stylus_body_idx is None:
            self._initialize_body_indices()
        
        num_resets = len(env_ids)
        
        # Use stable initial configuration
        joint_pos = torch.zeros((num_resets, 6), device=self.device)
        joint_pos[:, 0] = -0.96    # waist
        joint_pos[:, 1] = 0.0      # shoulder
        joint_pos[:, 2] = 1.0      # elbow
        joint_pos[:, 3] = 0.0      # yaw
        joint_pos[:, 4] = 2.0944   # pitch (~120 degrees)
        joint_pos[:, 5] = 0.0      # roll
        
        joint_vel = torch.zeros((num_resets, 6), device=self.device)
        
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # Reset state variables
        for agent in self.cfg.possible_agents:
            self.agent_actions[agent][env_ids] = 0.0
        
        self.human_forces_t[env_ids] = 0.0
        self.robot_forces_t[env_ids] = 0.0
        self.safety_distances_t1[env_ids] = 0.01
        self.is_violating_t1[env_ids] = False
        self.normal_t1[env_ids] = torch.zeros((num_resets, 3), device=self.device)

        # NEW: Reset four-zone reward state caches
        self.prev_safety_distances[env_ids] = 0.02
        self.prev_progress[env_ids] = 0.0
        self.best_progress[env_ids] = 0.0
        self.rejoin_streak[env_ids] = 0

    def _get_stylus_position(self) -> torch.Tensor:
        """Get stylus position relative to robot base."""
        if self.stylus_body_idx is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        
        base_pos = self._omni_robot.data.root_link_pos_w
        ee_pos = self._omni_robot.data.body_link_pos_w[:, self.stylus_body_idx, :]
        return ee_pos - base_pos
    
    def _get_stylus_velocity(self) -> torch.Tensor:
        """Get stylus velocity in world frame."""
        if self.stylus_body_idx is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        
        return self._omni_robot.data.body_link_lin_vel_w[:, self.stylus_body_idx, :]
    
    def __del__(self):
        """Destructor to clean up resources."""
        if hasattr(self, 'reward_logger'):
            self.reward_logger.close_all_files()
    
    def get_constraint_state(self, env_ids: Optional[List[int]] = None) -> Dict:
        """Get constraint state information."""
        if self.constraint_results_t1 is None:
            return {}
        
        if env_ids is None:
            return self.constraint_results_t1
        
        return {
            'distances_constraint': self.constraint_results_t1['distances_constraint'][env_ids],
            'closest_points': self.constraint_results_t1['closest_points'][env_ids],
            'normal_vectors': self.constraint_results_t1['normal_vectors'][env_ids],
            'is_overlapping': self.constraint_results_t1['is_overlapping'][env_ids],
            'is_inside': self.constraint_results_t1['is_inside'][env_ids]
        }
    
    def get_reward_details(self, env_ids: Optional[List[int]] = None) -> Dict:
        """Get detailed reward component information."""
        if not self.reward_components:
            return {}
        
        if env_ids is None:
            return self.reward_components
        
        return {
            key: value[env_ids] if torch.is_tensor(value) else value
            for key, value in self.reward_components.items()
        }