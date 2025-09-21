# surgical_direct_marl_env.py - Cleaned version with single evaluation chain
"""
Human-robot collaborative surgical MARL environment for shared networks.

Features:
- Multi-agent force control for surgical tasks
- Physics-based constraint checking and collision detection
- Four-zone reward system: Track/Surface/Danger/Rejoin
- Unified global_step integration with trainer
- Optional console logging via injected StepTracer
- Configuration from YAML file via trainer injection
- ADDED: Potential reward system for both robot and human agents
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Any, Dict, List, Optional
from collections import defaultdict
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.math import sample_uniform, quat_rotate_inverse

from .surgical_direct_marl_env_cfg import SurgicalDirectMARLEnvCfg
from .utils import CompleteConstraintChecker, TrajectoryManager


class SurgicalDirectMARLEnv(DirectMARLEnv):
    """
    Human-robot collaborative surgical MARL environment for shared networks.
    
    Features:
    - Multi-agent force control for surgical tasks
    - Physics-based constraint checking and collision detection
    - Four-zone reward system: Track/Surface/Danger/Rejoin
    - Unified global_step integration with trainer
    - Optional console logging via injected StepTracer
    - Configuration from YAML file via trainer injection
    - ADDED: Potential reward system for smooth trajectory guidance
    """
    
    cfg: SurgicalDirectMARLEnvCfg
    
    def __init__(self, cfg: SurgicalDirectMARLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize surgical MARL environment for shared networks."""
        super().__init__(cfg, render_mode, **kwargs)
        
        self._setup_core_configuration()
        self._initialize_state_variables()
        self._initialize_components()
        self._setup_gymnasium_spaces()
        
        # Unified global_step integration with trainer
        self._trainer_global_step = None
    
    def _setup_core_configuration(self) -> None:
        """Setup core configuration from trainer injection or direct YAML reading."""
        # Always try trainer injection first (standard flow)
        if hasattr(self, "params") and isinstance(getattr(self, "params", None), dict):
            print("[ENV] Using configuration injected by trainer")
        else:
            # Fallback: direct YAML reading (for standalone testing)
            print("[ENV] Configuration not injected by trainer, loading directly from YAML")
            try:
                import yaml
                import os
                # Get absolute path from current file location
                current_dir = os.path.dirname(os.path.abspath(__file__))
                yaml_file = os.path.join(current_dir, "agents", "training_params.yaml")
                
                if not os.path.exists(yaml_file):
                    raise FileNotFoundError(f"YAML config not found at: {yaml_file}")
                
                with open(yaml_file, 'r') as f:
                    self.params = yaml.safe_load(f)
                print(f"[ENV] Loaded configuration from: {yaml_file}")
                
            except Exception as e:
                print(f"[ERROR] Failed to load YAML configuration: {e}")
                raise RuntimeError(f"Could not load configuration: {e}")
            
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
        self._load_constraint_parameters()
        self._load_safety_parameters()
        self._load_termination_parameters()
        
        print(f"[ENV] Episode length: {self.cfg.episode_length_s}s")
        print(f"[ENV] Environment configured for physics + rewards only")
    
    def _load_constraint_parameters(self) -> None:
        """Load constraint-related parameters from configuration."""
        constraints = self.params['constraints']
        
        # Position and force limits
        self.min_z_pos = constraints['min_z_position']
        
        # Joint limits as tensors for efficient constraint enforcement
        joint_limits = constraints['joint_limits']
        self.joint_lower_limits = torch.tensor([
            joint_limits['waist'][0], joint_limits['shoulder'][0], joint_limits['elbow'][0],
            joint_limits['yaw'][0], joint_limits['pitch'][0], joint_limits['roll'][0]
        ], device=self.device, dtype=torch.float32)
        
        self.joint_upper_limits = torch.tensor([
            joint_limits['waist'][1], joint_limits['shoulder'][1], joint_limits['elbow'][1],
            joint_limits['yaw'][1], joint_limits['pitch'][1], joint_limits['roll'][1]
        ], device=self.device, dtype=torch.float32)
        
        # Fixed end joints configuration for stable wrist orientation
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
        self.enable_z_termination = term_config.get('z_below_zero', False)
        self.enable_edge_termination = term_config.get('edge_collision', True)
        self.safety_distance_threshold = term_config.get('safety_distance_threshold', 0.0)
    
    def _initialize_state_variables(self) -> None:
        """Initialize all state variables and caches."""
        # Environment base positions for coordinate transformations
        self.env_base_positions = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Agent actions storage for force application
        self.agent_actions = {
            agent: torch.zeros(self.num_envs, 3, device=self.device)
            for agent in self.cfg.possible_agents
        }
        
        self._initialize_physics_state()
        self._initialize_observation_cache()
        self._initialize_reward_state_caches()
        
        # Reward components cache for console logging
        self.reward_components = {}
        
        # Body index for stylus/end-effector identification
        self.stylus_body_idx = None
    
    def _initialize_physics_state(self) -> None:
        """Initialize physics interaction state variables."""
        self.human_forces_t = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_forces_t = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Cache for actor network outputs (for debugging console display)
        self.actor_mean_forces = {
            agent: torch.zeros(self.num_envs, 3, device=self.device)
            for agent in self.cfg.possible_agents
        }
        self.actor_noise_forces = {
            agent: torch.zeros(self.num_envs, 3, device=self.device)  
            for agent in self.cfg.possible_agents
        }
    
    def _initialize_observation_cache(self) -> None:
        """Initialize observation caching variables for efficient updates."""
        self.stylus_pos_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.stylus_vel_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.safety_distances_t1 = torch.ones(self.num_envs, device=self.device) * 0.01
        self.is_violating_t1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.normal_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.constraint_results_t1 = None

    def _initialize_reward_state_caches(self) -> None:
        """Initialize reward system state caches for delta calculations."""
        # Rejoin zone stability gate counter
        self.rejoin_streak = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        
        # Goal point for rejoin zone calculations
        self._goal_point = None
    
    def _initialize_components(self) -> None:
        """Initialize utility managers and components."""
        # Trajectory manager for path following and progress tracking
        self.trajectory_manager = TrajectoryManager(
            device=self.device,
            params=self.params,
            num_envs=self.num_envs,
            env_base_positions=self.env_base_positions
        )
        
        # Initialize goal point for rejoin zone calculations
        self._goal_point = self.trajectory_manager.end_pos_local.to(self.device)
        print(f"[ENV] Goal point: {self._goal_point}")
        
        # Constraint checker for collision detection and safety analysis
        self.constraint_checker = CompleteConstraintChecker(
            device=self.device, 
            collision_threshold=self.collision_threshold
        )
    
    def _setup_gymnasium_spaces(self) -> None:
        """Setup Gymnasium compatibility spaces for multi-agent interface."""
        import gymnasium as gym
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
    
    def set_trainer_global_step(self, global_step: int) -> None:
        """Set the current global step from trainer for unified logging."""
        self._trainer_global_step = global_step
        
    def _setup_scene(self):
        """Setup the simulation scene with robot and constraint objects."""
        # Initialize robot articulation
        self._omni_robot = Articulation(self.cfg.phantom_omni)
        self.scene.articulations["phantom_omni"] = self._omni_robot
        
        # Initialize constraint object for collision detection
        self._constraint = RigidObject(self.cfg.constraint)
        self.scene.rigid_objects["constraint"] = self._constraint
        
        # Clone environments for parallel simulation
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    def _setup_post_scene_creation(self):
        """Post-scene creation setup and manager updates."""
        super()._setup_post_scene_creation()
        
        self._initialize_body_indices()
        self._update_managers_with_robot_data()

    def _update_managers_with_robot_data(self) -> None:
        """Update managers with robot position data after scene creation."""
        if hasattr(self, '_omni_robot'):
            # Update environment base positions for coordinate transformations
            self.env_base_positions = self._omni_robot.data.root_link_pos_w.clone()
            self.trajectory_manager.env_base_positions = self.env_base_positions
            
    def _initialize_body_indices(self):
        """Initialize body indices for end-effector/stylus identification."""
        if self.stylus_body_idx is not None:
            return  # Already cached
            
        if not hasattr(self._omni_robot, 'body_names'):
            return
        
        # Find stylus body in robot hierarchy
        target_name = "stylus"
        
        for i, name in enumerate(self._omni_robot.body_names):
            if target_name in name.lower():
                self.stylus_body_idx = i
                print(f"[ENV] Found stylus body: {name} (index {i})")
                return
        
        # Fail fast if stylus not found
        raise RuntimeError(f"stylus link not found in robot body names: {self._omni_robot.body_names}")
    
    # =========================================================================
    # Weight accessor methods - simplified configuration access
    # =========================================================================
    
    def _get_weight(self, key: str, default: float = 0.0) -> float:
        """Get weight from flat YAML structure with fallback."""
        return float(
            self.params.get('reward_parameters', {})
                       .get('weights', {})
                       .get(key, default)
        )

    def _get_zone_weight(self, zone_letter: str, agent: str) -> float:
        """Get zone weight for specific agent."""
        return self._get_weight(f'zone{zone_letter}_weight_{agent}', 0.0)

    def _get_component_weight(self, zone_letter: str, comp: str, agent: str) -> float:
        """Get component weight for specific zone and agent."""
        return self._get_weight(f'zone{zone_letter}_{comp}_{agent}', 0.0)

    # =========================================================================
    # NEW: Non-incremental progress calculation with velocity-based direction
    # =========================================================================
    
    def _progress_raw_signed_by_velocity(self) -> torch.Tensor:
        # 轨迹方向
        t = self.trajectory_manager.line_direction
        # 当前速度
        v = self.stylus_vel_t1
        v_along = (v * t.unsqueeze(0)).sum(dim=-1)  # [N]

        # 当前绝对进度 p ∈ [0,1]
        p = self.trajectory_manager.get_progress(self.stylus_pos_t1).clamp(0.0, 1.0)  # [N]

        # 奖励参数
        MIN_REWARD = 0.1        # 起点处的最小奖励幅度
        MAX_REWARD = 0.2        # 终点处的最大奖励幅度
        EPS_V = 1e-4           # 速度死区阈值

        # 计算奖励幅度，从0.1（起点）到0.2（终点）线性变化
        reward_magnitude = MIN_REWARD + (MAX_REWARD - MIN_REWARD) * p  # [N] ∈ [0.1, 0.2]

        # 方向判断
        pos = (v_along >  EPS_V).float()   # 正方向
        neg = (v_along < -EPS_V).float()   # 负方向
        
        # 最终奖励，正方向为正，负方向为负
        return (pos - neg) * reward_magnitude  # [N] ∈ [-0.2, 0.2]
        
    def _pre_physics_step(self, actions: Dict[str, torch.Tensor]) -> None:
        """Pre-physics step processing with action validation and force application."""
        # FIXED: Always extract and update actor detail info, even in eval mode
        if hasattr(self, '_detail_actor_info') and self._detail_actor_info is not None:
            for agent in self.cfg.possible_agents:
                if agent in self._detail_actor_info['mean_actions']:
                    self.actor_mean_forces[agent] = self._detail_actor_info['mean_actions'][agent].clone()
                if agent in self._detail_actor_info['noise_actions']:
                    self.actor_noise_forces[agent] = self._detail_actor_info['noise_actions'][agent].clone()
        else:
            # If no detail info available (shouldn't happen), show current actions as mean forces with zero noise
            for agent in self.cfg.possible_agents:
                if agent in actions:
                    self.actor_mean_forces[agent] = actions[agent].clone()
                    self.actor_noise_forces[agent] = torch.zeros_like(actions[agent])
        
        # Process actions without additional force constraints (已在select_actions中限制)
        for agent, action in actions.items():
            assert agent in self.cfg.possible_agents, f"Unknown agent: {agent}"
            assert action.shape == (self.num_envs, 3), f"Action shape mismatch for {agent}: expected ({self.num_envs}, 3), got {action.shape}"
            
            # 直接使用动作，不再额外限制
            self.agent_actions[agent] = action
        
        # Cache forces for reward calculation
        self.robot_forces_t = self.agent_actions["robot"]
        self.human_forces_t = self.agent_actions["human"]
        
        self._apply_external_forces()
        self._enforce_joint_constraints()

    
    def set_detail_actor_info(self, detail_info: Dict[str, Any]) -> None:
        """Set detail information from MADDPG actor outputs for console logging."""
        self._detail_actor_info = detail_info
        
    def _apply_external_forces(self) -> None:
        """Apply external forces to the robot end-effector in world coordinates."""
        total_forces = self.robot_forces_t + self.human_forces_t
        stylus_quat = self._omni_robot.data.body_link_quat_w[:, self.stylus_body_idx, :]
        
        # Transform forces to local frame for proper force application
        forces_local = quat_rotate_inverse(stylus_quat, total_forces)
        
        # Reshape for Isaac Lab API requirements
        forces_with_body_dim = forces_local.unsqueeze(1)
        torques_with_body_dim = torch.zeros_like(forces_with_body_dim)
        
        self._omni_robot.set_external_force_and_torque(
            forces_with_body_dim, 
            torques_with_body_dim,
            body_ids=[self.stylus_body_idx]
        )
        
    def _enforce_joint_constraints(self) -> None:
        """Enforce joint limits and fix end joints for stable operation."""
        joint_pos = self._omni_robot.data.joint_pos.clone()
        joint_vel = self._omni_robot.data.joint_vel.clone()
        
        # Apply joint limits to prevent mechanical damage
        joint_pos = torch.clamp(joint_pos, self.joint_lower_limits, self.joint_upper_limits)
        
        # Fix end joints (wrist orientation) for stable task execution
        joint_pos[:, 3:6] = self.fixed_end_joints.unsqueeze(0).expand(self.num_envs, -1)
        joint_vel[:, 3:6] = 0.0
        
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel)
        
    def _apply_action(self) -> None:
        """Apply actions to simulation - required by Isaac Lab interface."""
        self._omni_robot.write_data_to_sim()

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """
        Compute observations for all agents and update state cache.
        
        Returns:
            Dictionary mapping agent_id to observation tensor [num_envs, obs_dim]
        """
        # Update state cache with current physics state
        self.stylus_pos_t1 = self._get_stylus_position()
        self.stylus_vel_t1 = self._get_stylus_velocity()
        
        # Update constraint state for collision detection
        current_base_positions = self._omni_robot.data.root_link_pos_w.clone()
        self.constraint_results_t1 = self.constraint_checker.analyze_constraint_state_batch(
            self.stylus_pos_t1, current_base_positions
        )
        self.safety_distances_t1 = self.constraint_results_t1['distances_constraint']
        self.is_violating_t1 = self.constraint_results_t1['is_overlapping']
        self.normal_t1 = self.constraint_results_t1['normal_vectors']

        # Ensure constraint distances have correct shape for concatenation
        constraint_distances = self.safety_distances_t1.unsqueeze(-1)  # [num_envs, 1]

        # Concatenate observation components (7 dimensions total)
        obs = torch.cat([
            self.stylus_pos_t1,       # End-effector position (3)
            self.stylus_vel_t1,       # End-effector velocity (3) 
            constraint_distances,     # Distance measurements (1)
        ], dim=-1)                    # Total: 7 dimensions
        
        # Create observation dictionary for each agent (shared observations)
        observations = {}
        for agent in self.cfg.possible_agents:
            observations[agent] = obs
            
        return observations

    # =========================================================================
    # FOUR-ZONE REWARD SYSTEM - Core reward calculation logic
    # =========================================================================

    def _build_zone_masks(self):
        """
        Build zone masks and mixing coefficient for four-zone reward system.
        
        Returns:
            Tuple of (outside, surface, danger, rejoin) masks
        """
        D, O = 0.0075, 0.015  # Danger threshold, Obstacle boundary
        d = self.safety_distances_t1
        
        # Basic zone masks based on safety distance
        surface = (d > D) & (d < O)
        outside = (d >= O) 
        danger = (d <= D)
        
        # Rejoin zone geometric conditions for trajectory recovery
        t = self.trajectory_manager.line_direction
        if self._goal_point is not None:
            goal_vec = self._goal_point.unsqueeze(0) - self.stylus_pos_t1
            g = goal_vec / torch.norm(goal_vec, dim=-1, keepdim=True).clamp(min=1e-8)
        else:
            g = t.unsqueeze(0).expand_as(self.stylus_pos_t1)
        
        c1, c2 = 0.90, 0.60
        align_goal = (g * t.unsqueeze(0)).sum(dim=-1) >= c1
        oppose_norm = (self.normal_t1 * t.unsqueeze(0)).sum(dim=-1) <= -c2
        rejoin_geom = surface & align_goal & oppose_norm
        
        # 10-step stability gate for rejoin zone activation
        self.rejoin_streak = torch.where(
            rejoin_geom, 
            self.rejoin_streak + 1, 
            torch.zeros_like(self.rejoin_streak)
        )
        rejoin = self.rejoin_streak >= 10
        
        return outside, surface, danger, rejoin

    def _zone_A_reward(self, masks, agent: str):
        """Zone A (Track): Outside obstacle boundary - Progress and deviation rewards."""
        outside, surface, danger, rejoin = masks
        
        # NEW: Non-incremental progress based on velocity direction
        prog_raw = self._progress_raw_signed_by_velocity()
        
        # Linear deviation penalty from desired trajectory
        deviations, _ = self.trajectory_manager.get_deviation(self.stylus_pos_t1)
        dev_raw = -torch.clamp(deviations * 4, min=0.0, max=0.2)  # 1cm deviation gives -0.04 penalty, max -0.2
        
        # Zone A combination with configurable weights
        wp = self._get_component_weight('A', 'progress', agent)
        wd = self._get_component_weight('A', 'deviation', agent)
        zw = self._get_zone_weight('A', agent)

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
        """Zone B (Surface): Near obstacle - Gap optimization and surface following."""
        outside, surface, danger, rejoin = masks
        
        # Gap reward for maintaining optimal distance from constraint
        beta = 6.0e3
        d = self.safety_distances_t1
        gap_raw = -beta * (d - 0.010) ** 2  # 2.5mm deviation gives -0.0375 penalty, 5mm gives -0.15
        # Surface tangent reward for following constraint boundary
        gamma = 1.0
        t = self.trajectory_manager.line_direction
        n = self.normal_t1
        
        t_dot_n = (t.unsqueeze(0) * n).sum(dim=-1, keepdim=True)
        t_surf = t.unsqueeze(0) - t_dot_n * n
        t_surf = t_surf / torch.norm(t_surf, dim=-1, keepdim=True).clamp(min=1e-8)
        
        v_surf = (self.stylus_vel_t1 * t_surf).sum(dim=-1)
        surf_raw = gamma * v_surf  # Tangential velocity 0.06m/s gives 0.06 reward
        
        # Unified inward penalty (penalize moving toward obstacle)
        lam = 2.0
        v_dot_n = (self.stylus_vel_t1 * self.normal_t1).sum(dim=-1)
        inward_raw = -lam * torch.clamp(v_dot_n, min=0.0)  # Normal velocity 0.06m/s gives -0.12 penalty, weight 2, only inward movement penalized
        
        # Zone B combination with configurable weights
        wg = self._get_component_weight('B', 'gap', agent)
        ws = self._get_component_weight('B', 'surftangent', agent)
        wi = self._get_component_weight('B', 'inward', agent)
        zw = self._get_zone_weight('B', agent)
        
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
        """
        Zone C (Danger): Implements a binary penalty system using raw * weight model.
        - If overlapping: A single, large, constant penalty is applied.
        - If too close (but not overlapping): The original penalties apply.
        """
        outside, surface, danger, rejoin = masks
        
        # Mode A: Physical overlap penalty (raw * weight)
        R_overlap_raw = -0.3  # Fixed overlap penalty of 0.3
        w_overlap = self._get_component_weight('C', 'overlap', agent)
        overlap_contrib = w_overlap * R_overlap_raw

        # Mode B: Dangerous proximity penalty (original logic)
        R_off = -0.2  # Fixed proximity penalty of 0.2
        offpen_raw = R_off * torch.ones_like(self.safety_distances_t1)
        lam = 2.0
        v_dot_n = (self.stylus_vel_t1 * self.normal_t1).sum(dim=-1)
        inward_raw = -lam * torch.clamp(v_dot_n, min=0.0)  # Normal velocity 0.06m/s gives -0.12 penalty, weight 2, only inward movement penalized
        wo = self._get_component_weight('C', 'offpen', agent)
        wi = self._get_component_weight('C', 'inward', agent)
        too_close_contrib = (wo * offpen_raw) + (wi * inward_raw)

        # Use torch.where to select final contribution based on collision status
        zone_c_total_contrib = torch.where(
            self.is_violating_t1,
            torch.tensor(overlap_contrib, device=self.device),
            too_close_contrib
        )

        # Apply zone weight
        zw = self._get_zone_weight('C', agent)
        zone_total = zw * zone_c_total_contrib
        
        out = torch.zeros_like(zone_total)
        out[danger] = zone_total[danger]

        # --- Update logging components ---
        RC, device = self.reward_components, zone_total.device
        RC[f'zoneC_weight_{agent}'] = torch.tensor(zw, device=device)
        RC[f'zoneC_total_{agent}'] = zone_total

        # Only when not overlapping, offpen and inward contributions are non-zero
        RC[f'zoneC_offpen_{agent}_contrib'] = torch.where(self.is_violating_t1, torch.zeros_like(zone_total), wo * offpen_raw)
        RC[f'zoneC_inward_{agent}_contrib'] = torch.where(self.is_violating_t1, torch.zeros_like(zone_total), wi * inward_raw)
        
        # Only when overlapping, overlap contribution is non-zero
        RC[f'zoneC_overlap_{agent}_contrib'] = torch.where(self.is_violating_t1, torch.tensor(overlap_contrib, device=self.device), torch.zeros_like(zone_total))

        # Always record raw and weight for StepTracer access
        RC[f'zoneC_offpen_{agent}_raw'] = offpen_raw
        RC[f'zoneC_offpen_{agent}_weight'] = torch.tensor(wo, device=device)
        RC[f'zoneC_inward_{agent}_raw'] = inward_raw
        RC[f'zoneC_inward_{agent}_weight'] = torch.tensor(wi, device=device)
        RC[f'zoneC_overlap_{agent}_raw'] = torch.full_like(zone_total, R_overlap_raw)
        RC[f'zoneC_overlap_{agent}_weight'] = torch.tensor(w_overlap, device=device)
        
        return out
    
    def _zone_D_reward(self, masks, agent: str):
        """Zone D (Rejoin): Trajectory recovery - Progress + Deviation + Outward movement."""
        outside, surface, danger, rejoin = masks

        # NEW: Non-incremental progress based on velocity direction
        prog_raw = self._progress_raw_signed_by_velocity()

        # Enhanced deviation penalty for trajectory recovery
        deviations, _ = self.trajectory_manager.get_deviation(self.stylus_pos_t1)
        dev_raw = -torch.clamp(deviations * 4, min=0.0, max=0.2)  # 1cm deviation gives -0.04 penalty, max -0.2

        # Inward movement handling (reward moving away, penalize moving toward)
        lam = 2.0
        v_dot_n = (self.stylus_vel_t1 * self.normal_t1).sum(dim=-1)
        inward_raw = torch.where(
            v_dot_n >= 0,        # Moving toward obstacle
            -lam * v_dot_n,      # Penalize
            lam * (-v_dot_n)     # Moving away from obstacle, reward
        )  # Normal velocity inward 0.06m/s gives -0.12 penalty, outward gives +0.12 reward, weight 2

        wp = self._get_component_weight('D', 'progress', agent)
        wd = self._get_component_weight('D', 'deviation', agent)
        wi = self._get_component_weight('D', 'inward', agent)
        zw = self._get_zone_weight('D', agent)

        prog_contrib = wp * prog_raw
        dev_contrib  = wd * dev_raw
        inw_contrib  = wi * inward_raw
        zone_total = zw * (prog_contrib + dev_contrib + inw_contrib)

        out = torch.zeros_like(prog_raw)
        out[rejoin] = zone_total[rejoin]

        # Store for console display
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
    # Global reward functions - Task-level objectives
    # =========================================================================

    def _globals_for_agent(self, agent: str):
        """Calculate global rewards for agent with three-piece structure."""
        RC, dev = self.reward_components, self.device

        # Z position penalty for safety constraint
        z_raw = self._calculate_z_penalty()
        zw = self._get_weight(f'zpenalty_{agent}', 0.0)
        RC[f'global_zpenalty_{agent}_raw'] = z_raw
        RC[f'global_zpenalty_{agent}_weight'] = torch.tensor(zw, device=dev)
        RC[f'global_zpenalty_{agent}_contrib'] = z_raw * zw

        # Task completion reward
        c_raw = self._calculate_completion_reward()
        cw = self._get_weight(f'completion_{agent}', 0.0)
        RC[f'global_completion_{agent}_raw'] = c_raw
        RC[f'global_completion_{agent}_weight'] = torch.tensor(cw, device=dev)
        RC[f'global_completion_{agent}_contrib'] = c_raw * cw

        # Time efficiency reward
        t_raw = self._calculate_time_efficiency_reward()
        tw = self._get_weight(f'timeeff_{agent}', 0.0)
        RC[f'global_timeeff_{agent}_raw'] = t_raw
        RC[f'global_timeeff_{agent}_weight'] = torch.tensor(tw, device=dev)
        RC[f'global_timeeff_{agent}_contrib'] = t_raw * tw

        # Potential (distance-to-goal shaped reward)
        p_raw = self._calculate_potential_reward()
        pw = self._get_weight(f'potential_{agent}', 0.0)
        RC[f'global_potential_{agent}_raw'] = p_raw
        RC[f'global_potential_{agent}_weight'] = torch.tensor(pw, device=dev)
        RC[f'global_potential_{agent}_contrib'] = p_raw * pw

        # Own force efficiency penalty
        f_raw = self._calculate_force_penalties()[agent]
        fw = self._get_weight(f'forceeff_{agent}', 0.0)
        RC[f'{agent}force_raw'] = f_raw
        RC[f'{agent}force_weight'] = torch.tensor(fw, device=dev)
        RC[f'{agent}force_contrib'] = f_raw * fw

        # Cross-agent awareness reward
        other = 'human' if agent == 'robot' else 'robot'
        aw_key = 'humanaware_robot' if agent == 'robot' else 'robotaware_human'
        aw_raw = self._calculate_force_penalties()[other]
        aw_w = self._get_weight(aw_key, 0.0)
        label = 'humanaware' if agent == 'robot' else 'robotaware'
        RC[f'{label}_raw'] = aw_raw
        RC[f'{label}_weight'] = torch.tensor(aw_w, device=dev)
        RC[f'{label}_contrib'] = aw_raw * aw_w

    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        """
        Four-zone reward system with robot/human symmetric rewards.
        Isaac Lab required method - cannot be renamed.
        """
        self.reward_components = {}  # Reset each step
        masks = self._build_zone_masks()

        def _agent_zone_sum(agent: str):
            """Calculate total zone rewards for specific agent."""
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
            """Calculate total reward including zones and globals for agent."""
            RC = self.reward_components
            globals_sum = (
                RC[f'global_zpenalty_{agent}_contrib'] +
                RC[f'global_completion_{agent}_contrib'] +
                RC[f'global_timeeff_{agent}_contrib'] +
                RC[f'global_potential_{agent}_contrib'] +   # 新增势能项
                RC[f'{agent}force_contrib'] +
                (RC['humanaware_contrib'] if agent=='robot' else RC['robotaware_contrib'])
            )
            return (robot_zones if agent=='robot' else human_zones) + globals_sum

        rewards = {'robot': _agent_total('robot'), 'human': _agent_total('human')}

        # Add unified keys for console logging consistency
        deviations, _ = self.trajectory_manager.get_deviation(self.stylus_pos_t1)
        progress_ratio = self.trajectory_manager.get_progress(self.stylus_pos_t1)
        distance_to_final = torch.norm(
            self.stylus_pos_t1 - self.trajectory_manager.end_pos_local.unsqueeze(0), dim=-1
        )
        completion_reward = self._calculate_completion_reward()
        
        self.reward_components.update({
            'deviation': deviations,
            'progress_ratio': progress_ratio,
            'distance_to_final': distance_to_final,
            'completion_reward': completion_reward,
            'min_safety_distance': self.safety_distances_t1,
        })
        
        # Ensure all required keys exist with default values
        default_keys = ['deviation', 'progress_ratio', 'min_safety_distance', 'completion_reward']
        for key in default_keys:
            if key not in self.reward_components:
                self.reward_components[key] = torch.zeros(self.num_envs, device=self.device)

        # Optional console logging via injected StepTracer
        if hasattr(self, "step_tracer") and self.step_tracer is not None:
            current_step = self._trainer_global_step if self._trainer_global_step is not None else 0
            self.step_tracer.maybe_print_step(self, rewards, current_step)
        
        return rewards

    # =========================================================================
    # Global reward calculation functions
    # =========================================================================

    def _calculate_potential_reward(self) -> torch.Tensor:
        """
        Concave increasing potential based on distance to final setpoint:
        R = 1 - sqrt(clamp(d/d0, 0, 1))
        - 0 at start (d=d0), 1 at goal (d=0)
        
        NOTE: For parallel environments, we use local coordinates (stylus_pos_t1 is already relative to robot base)
        and trajectory_manager.end_pos_local is in local coordinates, so no additional offset needed.
        """
        # 当前到终点的距离 (都在局部坐标系中)
        d = torch.norm(
            self.stylus_pos_t1 - self.trajectory_manager.end_pos_local.unsqueeze(0),
            dim=-1
        )
        # 起点到终点的基准距离（避免除零）
        d0 = max(float(self.trajectory_manager.total_distance), 1e-8)
        u = torch.clamp(d / d0, min=0.0, max=1.0)
        return 1.0 - torch.sqrt(u)

    def _calculate_force_penalties(self) -> Dict[str, torch.Tensor]:
        """Calculate force efficiency penalties for both agents."""
        return {
            'robot': -1.0 * torch.norm(self.robot_forces_t, dim=-1),
            'human': -1.0 * torch.norm(self.human_forces_t, dim=-1)
        }

    def _calculate_z_penalty(self) -> torch.Tensor:
        """Calculate Z-axis constraint penalty for safety."""
        return torch.where(
            self.stylus_pos_t1[:, 2] < 0.0,
            -500.0 * torch.abs(self.stylus_pos_t1[:, 2]),
            torch.zeros_like(self.stylus_pos_t1[:, 2])
        )

    def _calculate_completion_reward(self) -> torch.Tensor:
        """Calculate task completion reward using unified threshold."""
        is_final_reached = self.trajectory_manager.is_final_setpoint_reached(self.stylus_pos_t1)
        return torch.where(
            is_final_reached,
            torch.full_like(self.safety_distances_t1, 1.0),  # Completion reward of 1.0
            torch.zeros_like(self.safety_distances_t1)
        )

    def _calculate_time_efficiency_reward(self) -> torch.Tensor:
        """Calculate time efficiency reward based on episode progress."""
        max_steps = 1200  # 20s * 60fps = 1200 steps per episode
        
        # Use environment's episode step counter
        current_steps = self.episode_length_buf.to(self.device).float()
        
        # Higher reward for completing tasks faster
        time_efficiency = (max_steps - current_steps) / max_steps
        time_efficiency = torch.clamp(time_efficiency, min=0.0, max=1.0)

        return time_efficiency * 3.0

    # =========================================================================
    # Episode management and termination
    # =========================================================================

    def _get_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Determine termination and truncation conditions for all agents."""
        
        # Z-axis termination for safety
        z_below_zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.enable_z_termination:
            z_below_zero = self.stylus_pos_t1[:, 2] < self.min_z_pos
        
        # Edge collision termination for safety
        edge_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.enable_edge_termination:
            edge_collision = self.safety_distances_t1 <= self.safety_distance_threshold
        
        # Task completion using unified threshold
        final_reached = self.trajectory_manager.is_final_setpoint_reached(self.stylus_pos_t1)
        
        # Combine termination conditions
        terminated_condition = z_below_zero | edge_collision | final_reached

        # Time truncation managed by Isaac Lab
        truncated_condition = self.episode_length_buf >= self.max_episode_length - 1
        
        terminated = {agent: terminated_condition for agent in self.cfg.possible_agents}
        truncated = {agent: truncated_condition for agent in self.cfg.possible_agents}
        
        return terminated, truncated
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments with stable initial configuration."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        super()._reset_idx(env_ids)
        
        if self.stylus_body_idx is None:
            self._initialize_body_indices()
        
        num_resets = len(env_ids)
        
        # Use stable initial joint configuration
        joint_pos = torch.zeros((num_resets, 6), device=self.device)
        joint_pos[:, 0] = -0.96    # waist
        joint_pos[:, 1] = 0.0      # shoulder
        joint_pos[:, 2] = 1.0      # elbow
        joint_pos[:, 3] = 0.0      # yaw
        joint_pos[:, 4] = 2.0944   # pitch (~120 degrees)
        joint_pos[:, 5] = 0.0      # roll
        
        joint_vel = torch.zeros((num_resets, 6), device=self.device)
        
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # Reset state variables for specified environments
        for agent in self.cfg.possible_agents:
            self.agent_actions[agent][env_ids] = 0.0
        
        self.human_forces_t[env_ids] = 0.0
        self.robot_forces_t[env_ids] = 0.0
        self.safety_distances_t1[env_ids] = 0.01
        self.is_violating_t1[env_ids] = False
        self.normal_t1[env_ids] = torch.zeros((num_resets, 3), device=self.device)

        # Reset reward state caches (no more best_progress)
        self.rejoin_streak[env_ids] = 0

        # Reset actor detail info caches
        for agent in self.cfg.possible_agents:
            self.actor_mean_forces[agent][env_ids] = 0.0
            self.actor_noise_forces[agent][env_ids] = 0.0

    # =========================================================================
    # Utility methods for state access
    # =========================================================================

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