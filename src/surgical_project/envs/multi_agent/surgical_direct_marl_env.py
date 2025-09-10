# surgical_direct_marl_env.py - Simplified Zone System with Binary Progress Penalty
# MODIFIED: Simplified progress penalty: -2 if no improvement over historical best, otherwise proportional reward
# REMOVED: All stall/back accumulation logic, p_tm1 calculations, stylus_pos_t0 tracking

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
    MODIFIED: Simplified four-zone reward system with binary progress penalty
    MODIFIED: Removed all stall/back accumulation logic for cleaner implementation
    
    Features:
    - Multi-agent force control for surgical tasks
    - Physics-based constraint checking and collision detection
    - Four-zone reward system: Track/Surface/Danger/Rejoin
    - Simplified progress penalty: -2 for no improvement, proportional for progress
    - Performance evaluation and milestone tracking
    - Console logging via utils (configured by YAML)
    - Actor network debug information display
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
        
        # Simplified reward system state caches
        self._initialize_reward_state_caches()
        
        # Reward components cache
        self.reward_components = {}
        
        # Body index for stylus/end-effector (will be set during scene setup)
        self.stylus_body_idx = None
    
    def _initialize_physics_state(self) -> None:
        """Initialize physics interaction state variables (forces at time t)."""
        self.human_forces_t = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_forces_t = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Cache for actor network outputs (for debugging display)
        self.actor_mean_forces = {
            agent: torch.zeros(self.num_envs, 3, device=self.device)
            for agent in self.cfg.possible_agents
        }
        self.actor_noise_forces = {
            agent: torch.zeros(self.num_envs, 3, device=self.device)  
            for agent in self.cfg.possible_agents
        }
    
    def _initialize_observation_cache(self) -> None:
        """Initialize state caching variables (updated in _get_observations)."""
        self.stylus_pos_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.stylus_vel_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.safety_distances_t1 = torch.ones(self.num_envs, device=self.device) * 0.01
        self.is_violating_t1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.normal_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.constraint_results_t1 = None

    def _initialize_reward_state_caches(self) -> None:
        """Initialize simplified reward system state caches."""
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
        
        # Initialize goal point after trajectory manager is ready
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
    
    # =========================================================================
    # Weight accessor methods
    # =========================================================================
    
    def _w(self, key: str, default: float = 0.0) -> float:
        """Get weight from flat YAML structure with fallback."""
        try:
            return float(self.params['reward_parameters']['weights'].get(key, default))
        except Exception:
            return float(default)

    def _zone_w(self, zone_letter: str, agent: str) -> float:
        """Get zone weight (e.g., zoneA_weight_robot)."""
        return self._w(f'zone{zone_letter}_weight_{agent}', 0.0)

    def _comp_w(self, zone_letter: str, comp: str, agent: str) -> float:
        """Get component weight (e.g., zoneA_progress_robot)."""
        return self._w(f'zone{zone_letter}_{comp}_{agent}', 0.0)
        
    def _pre_physics_step(self, actions: Dict[str, torch.Tensor]) -> None:
        """Pre-physics step processing including action validation and force application."""
        # Extract actor debug info if available (passed from MADDPG)
        if hasattr(self, '_debug_actor_info') and self._debug_actor_info is not None:
            for agent in self.cfg.possible_agents:
                if agent in self._debug_actor_info['mean_actions']:
                    self.actor_mean_forces[agent] = self._debug_actor_info['mean_actions'][agent].clone()
                if agent in self._debug_actor_info['noise_actions']:
                    self.actor_noise_forces[agent] = self._debug_actor_info['noise_actions'][agent].clone()
        
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
    
    def set_debug_actor_info(self, debug_info: Dict[str, Any]) -> None:
        """Set debug information from MADDPG actor outputs."""
        self._debug_actor_info = debug_info
        
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
        
        # Update reward state caches at end of observation update
        s_t = self.trajectory_manager.get_progress(self.stylus_pos_t1)
        self.prev_safety_distances = self.safety_distances_t1.detach().clone()
        self.prev_progress = s_t.detach().clone()
        
        # Create observation dictionary for each agent
        observations = {}
        for agent in self.cfg.possible_agents:
            observations[agent] = obs
            
        return observations

    # =========================================================================
    # FOUR-ZONE REWARD SYSTEM
    # =========================================================================

    def _build_zone_masks(self):
        """Build zone masks and alpha mixing coefficient (mutually exclusive with stability gate)."""
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
        
        # Alpha mixing coefficient for zone D (not used anymore but kept for compatibility)
        alpha = ((d - D) / (O - D)).clamp(0.0, 1.0)
        
        return outside, surface, danger, rejoin, alpha

    # =========================================================================
    # SIMPLIFIED ZONE REWARD FUNCTIONS (inline calculations with unified inward)
    # =========================================================================

    def _zone_A_reward(self, masks, agent: str):
        """Zone A (Track): Outside obstacle boundary (d >= 1.5cm) - Inline calculations."""
        outside, surface, danger, rejoin, alpha = masks
        
        # === SIMPLIFIED PROGRESS: Binary penalty system ===
        T = 1200.0  # Total steps in episode
        
        # Get current progress ratio (clamped to valid range)
        p_t = self.trajectory_manager.get_progress(self.stylus_pos_t1).clamp(0.0, 1.0)
        best_p_tm1 = self.best_progress.clone()
        
        # Only reward progress that beats historical best
        delta_p = p_t - best_p_tm1
        reward_forward = torch.clamp(delta_p * T, min=0.0, max=4.0)
        
        # Binary penalty: -2 if no improvement over historical best, otherwise use reward_forward
        prog_raw = torch.where(reward_forward > 0.0, reward_forward, torch.full_like(reward_forward, -2.0))
        
        # Update historical best progress
        self.best_progress = torch.maximum(best_p_tm1, p_t)
        
        # === INLINE LINEAR DEVIATION PENALTY ===
        # Linear penalty: 1cm = -0.5, 2cm = -1, max penalty = -4
        deviations, _ = self.trajectory_manager.get_deviation(self.stylus_pos_t1)
        dev_raw = -torch.clamp(deviations * 50.0, min=0.0, max=4.0)  # 0.01m * 50 = 0.5
        
        # === ZONE A COMBINATION ===
        wp = self._comp_w('A', 'progress', agent)
        wd = self._comp_w('A', 'deviation', agent)
        zw = self._zone_w('A', agent)

        prog_contrib = wp * prog_raw
        dev_contrib = wd * dev_raw
        zone_total = zw * (prog_contrib + dev_contrib)

        out = torch.zeros_like(prog_raw)
        out[outside] = zone_total[outside]

        # Store for console display
        RC, device = self.reward_components, prog_raw.device
        RC[f'zoneA_weight_{agent}'] = torch.tensor(zw, device=device)
        RC[f'zoneA_total_{agent}'] = zone_total
        RC[f'zoneA_progress_{agent}_raw'] = prog_raw
        RC[f'zoneA_progress_{agent}_weight'] = torch.tensor(wp, device=device)
        RC[f'zoneA_progress_{agent}_contrib'] = prog_contrib
        RC[f'zoneA_deviation_{agent}_raw'] = dev_raw
        RC[f'zoneA_deviation_{agent}_weight'] = torch.tensor(wd, device=device)
        RC[f'zoneA_deviation_{agent}_contrib'] = dev_contrib

        return out

    def _zone_B_reward(self, masks, agent: str):
        """Zone B (Surface): Near obstacle (0.75cm < d < 1.5cm), not in rejoin - Inline calculations."""
        outside, surface, danger, rejoin, alpha = masks
        
        # === INLINE GAP CALCULATION ===
        beta = 8.0e4
        d = self.safety_distances_t1
        gap_raw = -beta * (d - 0.010) ** 2
        
        # === INLINE SURFACE TANGENT CALCULATION ===
        gamma = 10.0
        t = self.trajectory_manager.line_direction
        n = self.normal_t1
        
        # Calculate surface tangent direction
        t_dot_n = (t.unsqueeze(0) * n).sum(dim=-1, keepdim=True)
        t_surf = t.unsqueeze(0) - t_dot_n * n
        t_surf = t_surf / torch.norm(t_surf, dim=-1, keepdim=True).clamp(min=1e-8)
        
        # Velocity along surface tangent
        v_surf = (self.stylus_vel_t1 * t_surf).sum(dim=-1)
        surf_raw = gamma * v_surf
        
        # === UNIFIED INWARD CALCULATION (B/C zones: penalize moving toward obstacle) ===
        lam = 30.0
        v_dot_n = (self.stylus_vel_t1 * self.normal_t1).sum(dim=-1)
        inward_raw = -lam * torch.clamp(v_dot_n, min=0.0)  # Penalize moving toward obstacle
        
        # === ZONE B COMBINATION ===
        wg = self._comp_w('B', 'gap', agent)
        ws = self._comp_w('B', 'surftangent', agent)
        wi = self._comp_w('B', 'inward', agent)
        zw = self._zone_w('B', agent)
        
        gap_contrib = wg * gap_raw
        surf_contrib = ws * surf_raw
        inward_contrib = wi * inward_raw
        zone_total = zw * (gap_contrib + surf_contrib + inward_contrib)
        
        out = torch.zeros_like(gap_raw)
        surface_only = surface & (~rejoin)
        out[surface_only] = zone_total[surface_only]
        
        # Store for console display
        RC, device = self.reward_components, gap_raw.device
        RC[f'zoneB_weight_{agent}'] = torch.tensor(zw, device=device)
        RC[f'zoneB_total_{agent}'] = zone_total
        RC[f'zoneB_gap_{agent}_raw'] = gap_raw
        RC[f'zoneB_gap_{agent}_weight'] = torch.tensor(wg, device=device)
        RC[f'zoneB_gap_{agent}_contrib'] = gap_contrib
        RC[f'zoneB_surftangent_{agent}_raw'] = surf_raw
        RC[f'zoneB_surftangent_{agent}_weight'] = torch.tensor(ws, device=device)
        RC[f'zoneB_surftangent_{agent}_contrib'] = surf_contrib
        RC[f'zoneB_inward_{agent}_raw'] = inward_raw
        RC[f'zoneB_inward_{agent}_weight'] = torch.tensor(wi, device=device)
        RC[f'zoneB_inward_{agent}_contrib'] = inward_contrib

        return out

    def _zone_C_reward(self, masks, agent: str):
        """Zone C (Danger): Inside obstacle (d <= 0.75cm) - Inline calculations."""
        outside, surface, danger, rejoin, alpha = masks
        
        # === INLINE OFF-PENALTY CALCULATION ===
        R_off = -0.6
        offpen_raw = R_off * torch.ones_like(self.safety_distances_t1)
        
        # === UNIFIED INWARD CALCULATION (B/C zones: penalize moving toward obstacle) ===
        lam = 30.0
        v_dot_n = (self.stylus_vel_t1 * self.normal_t1).sum(dim=-1)
        inward_raw = -lam * torch.clamp(v_dot_n, min=0.0)  # Penalize moving toward obstacle
        
        # === ZONE C COMBINATION ===
        wo = self._comp_w('C', 'offpen', agent)
        wi = self._comp_w('C', 'inward', agent)
        zw = self._zone_w('C', agent)
        
        offpen_contrib = wo * offpen_raw
        inward_contrib = wi * inward_raw
        zone_total = zw * (offpen_contrib + inward_contrib)
        
        out = torch.zeros_like(offpen_raw)
        out[danger] = zone_total[danger]
        
        # Store for console display
        RC, device = self.reward_components, offpen_raw.device
        RC[f'zoneC_weight_{agent}'] = torch.tensor(zw, device=device)
        RC[f'zoneC_total_{agent}'] = zone_total
        RC[f'zoneC_offpen_{agent}_raw'] = offpen_raw
        RC[f'zoneC_offpen_{agent}_weight'] = torch.tensor(wo, device=device)
        RC[f'zoneC_offpen_{agent}_contrib'] = offpen_contrib
        RC[f'zoneC_inward_{agent}_raw'] = inward_raw
        RC[f'zoneC_inward_{agent}_weight'] = torch.tensor(wi, device=device)
        RC[f'zoneC_inward_{agent}_contrib'] = inward_contrib

        return out

    def _zone_D_reward(self, masks, agent: str):
        """Zone D (Rejoin): SIMPLIFIED - Progress + Deviation + Inward only."""
        outside, surface, danger, rejoin, alpha = masks  # alpha not used anymore

        # ===== SIMPLIFIED PROGRESS: Binary penalty system =====
        T = 1200.0
        p_t = self.trajectory_manager.get_progress(self.stylus_pos_t1).clamp(0.0, 1.0)
        best_p_tm1 = self.best_progress.clone()

        delta_p = p_t - best_p_tm1
        reward_forward = torch.clamp(delta_p * T, min=0.0, max=4.0)

        # Binary penalty: -2 if no improvement over historical best, otherwise use reward_forward
        prog_raw = torch.where(reward_forward > 0.0, reward_forward, torch.full_like(reward_forward, -2.0))
        self.best_progress = torch.maximum(best_p_tm1, p_t)

        # ===== DEVIATION (same as Zone A) =====
        deviations, _ = self.trajectory_manager.get_deviation(self.stylus_pos_t1)
        dev_raw = -torch.clamp(deviations * 50.0, min=0.0, max=4.0)

        # ===== INWARD (Zone D: reward moving away, penalize moving toward) =====
        lam = 30.0
        v_dot_n = (self.stylus_vel_t1 * self.normal_t1).sum(dim=-1)
        inward_raw = torch.where(
            v_dot_n >= 0,        # Moving toward obstacle
            -lam * v_dot_n,      # Penalize
            lam * (-v_dot_n)     # Moving away from obstacle, reward
        )

        # ===== Weights + combine =====
        wp = self._comp_w('D', 'progress', agent)
        wd = self._comp_w('D', 'deviation', agent)
        wi = self._comp_w('D', 'inward', agent)
        zw = self._zone_w('D', agent)

        prog_contrib = wp * prog_raw
        dev_contrib  = wd * dev_raw
        inw_contrib  = wi * inward_raw
        zone_total = zw * (prog_contrib + dev_contrib + inw_contrib)

        out = torch.zeros_like(prog_raw)
        out[rejoin] = zone_total[rejoin]

        # ===== Console detail =====
        RC, device = self.reward_components, prog_raw.device
        RC[f'zoneD_weight_{agent}'] = torch.tensor(zw, device=device)
        RC[f'zoneD_total_{agent}'] = zone_total
        RC[f'zoneD_progress_{agent}_raw'] = prog_raw
        RC[f'zoneD_progress_{agent}_weight'] = torch.tensor(wp, device=device)
        RC[f'zoneD_progress_{agent}_contrib'] = prog_contrib
        RC[f'zoneD_deviation_{agent}_raw'] = dev_raw
        RC[f'zoneD_deviation_{agent}_weight'] = torch.tensor(wd, device=device)
        RC[f'zoneD_deviation_{agent}_contrib'] = dev_contrib
        RC[f'zoneD_inward_{agent}_raw'] = inward_raw
        RC[f'zoneD_inward_{agent}_weight'] = torch.tensor(wi, device=device)
        RC[f'zoneD_inward_{agent}_contrib'] = inw_contrib

        return out

    # =========================================================================
    # Global reward functions with three-piece structure
    # =========================================================================

    def _globals_for_agent(self, agent: str):
        """Calculate global rewards for agent with three-piece structure."""
        RC, dev = self.reward_components, self.device

        # Z penalty
        z_raw = self._calculate_z_penalty()
        zw = self._w(f'zpenalty_{agent}', 0.0)
        RC[f'global_zpenalty_{agent}_raw'] = z_raw
        RC[f'global_zpenalty_{agent}_weight'] = torch.tensor(zw, device=dev)
        RC[f'global_zpenalty_{agent}_contrib'] = z_raw * zw

        # completion
        c_raw = self._calculate_completion_reward()
        cw = self._w(f'completion_{agent}', 0.0)
        RC[f'global_completion_{agent}_raw'] = c_raw
        RC[f'global_completion_{agent}_weight'] = torch.tensor(cw, device=dev)
        RC[f'global_completion_{agent}_contrib'] = c_raw * cw

        # time efficiency
        t_raw = self._calculate_time_efficiency_reward()
        tw = self._w(f'timeeff_{agent}', 0.0)
        RC[f'global_timeeff_{agent}_raw'] = t_raw
        RC[f'global_timeeff_{agent}_weight'] = torch.tensor(tw, device=dev)
        RC[f'global_timeeff_{agent}_contrib'] = t_raw * tw

        # own force
        f_raw = self._calculate_force_penalties()[agent]
        fw = self._w(f'forceeff_{agent}', 0.0)
        RC[f'{agent}force_raw'] = f_raw
        RC[f'{agent}force_weight'] = torch.tensor(fw, device=dev)
        RC[f'{agent}force_contrib'] = f_raw * fw

        # awareness (other agent)
        other = 'human' if agent == 'robot' else 'robot'
        aw_key = 'humanaware_robot' if agent == 'robot' else 'robotaware_human'
        aw_raw = self._calculate_force_penalties()[other]
        aw_w = self._w(aw_key, 0.0)
        label = 'humanaware' if agent == 'robot' else 'robotaware'
        RC[f'{label}_raw'] = aw_raw
        RC[f'{label}_weight'] = torch.tensor(aw_w, device=dev)
        RC[f'{label}_contrib'] = aw_raw * aw_w

    # =========================================================================
    # Isaac Lab required method name: _get_rewards (CANNOT be changed)
    # =========================================================================

    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        """Four-zone reward system with robot/human symmetric rewards (Isaac Lab required method)."""
        self.reward_components = {}  # Reset each step
        masks = self._build_zone_masks()

        def _agent_zone_sum(agent: str):
            ZA = self._zone_A_reward(masks, agent)
            ZB = self._zone_B_reward(masks, agent)
            ZC = self._zone_C_reward(masks, agent)
            ZD = self._zone_D_reward(masks, agent)
            return ZA + ZB + ZC + ZD

        robot_zones = _agent_zone_sum('robot')
        human_zones = _agent_zone_sum('human')

        self._globals_for_agent('robot')
        self._globals_for_agent('human')

        def _agent_total(agent: str):
            RC = self.reward_components
            globals_sum = (
                RC[f'global_zpenalty_{agent}_contrib'] +
                RC[f'global_completion_{agent}_contrib'] +
                RC[f'global_timeeff_{agent}_contrib'] +
                RC[f'{agent}force_contrib'] +
                (RC['humanaware_contrib'] if agent=='robot' else RC['robotaware_contrib'])
            )
            return (robot_zones if agent=='robot' else human_zones) + globals_sum

        rewards = {'robot': _agent_total('robot'), 'human': _agent_total('human')}

        # Public indicators for RewardLogger
        deviations, _ = self.trajectory_manager.get_deviation(self.stylus_pos_t1)
        progress_ratio = self.trajectory_manager.get_progress(self.stylus_pos_t1)
        distance_to_final = torch.norm(
            self.stylus_pos_t1 - self.trajectory_manager.end_pos_local.unsqueeze(0), dim=-1
        )
        self.reward_components['deviation'] = deviations
        self.reward_components['progress_ratio'] = progress_ratio
        self.reward_components['distance_to_final'] = distance_to_final

        # Call utils with new signature (no robot_weights/human_weights parameters)
        current_step = getattr(self, 'step_counter', getattr(self, '_sim_step_counter', 0))
        active_envs = getattr(self, 'active_env_ids', None)
        self.reward_logger.log_console_if_enabled(self, rewards, current_step, active_envs)
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
        """Reset specified environments and simplified reward state caches."""
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

        # Reset simplified reward state caches
        self.prev_safety_distances[env_ids] = 0.02
        self.prev_progress[env_ids] = 0.0
        self.best_progress[env_ids] = 0.0
        self.rejoin_streak[env_ids] = 0

        # Reset actor debug info caches
        for agent in self.cfg.possible_agents:
            self.actor_mean_forces[agent][env_ids] = 0.0
            self.actor_noise_forces[agent][env_ids] = 0.0

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