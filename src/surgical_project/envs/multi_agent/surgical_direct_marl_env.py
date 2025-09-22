"""
Human-robot collaborative surgical MARL environment for shared networks.
Features physics-based constraints, four-zone rewards, and potential field guidance.
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
    Multi-agent surgical robot environment with shared network support.
    
    Features:
    - Multi-agent force control for surgical tasks
    - Physics-based constraint checking and collision detection
    - Four-zone reward system: Track/Surface/Danger/Rejoin
    - Potential field reward for trajectory guidance
    - Unified coordinate system handling
    - Console logging support via injected StepTracer
    """
    
    cfg: SurgicalDirectMARLEnvCfg
    
    def __init__(self, cfg: SurgicalDirectMARLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize surgical MARL environment."""
        super().__init__(cfg, render_mode, **kwargs)
        
        self._setup_core_configuration()
        self._initialize_state_variables()
        self._initialize_components()
        self._setup_gymnasium_spaces()
        
        self._trainer_global_step = None
    
    def _setup_core_configuration(self) -> None:
        """Setup core configuration from trainer injection or direct YAML reading."""
        if hasattr(self, "params") and isinstance(getattr(self, "params", None), dict):
            print("[ENV] Using configuration injected by trainer")
        else:
            print("[ENV] Configuration not injected by trainer, loading directly from YAML")
            try:
                import yaml
                import os
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
    
    def _load_constraint_parameters(self) -> None:
        """Load constraint-related parameters from configuration."""
        constraints = self.params['constraints']
        
        self.min_z_pos = constraints['min_z_position']
        
        joint_limits = constraints['joint_limits']
        self.joint_lower_limits = torch.tensor([
            joint_limits['waist'][0], joint_limits['shoulder'][0], joint_limits['elbow'][0],
            joint_limits['yaw'][0], joint_limits['pitch'][0], joint_limits['roll'][0]
        ], device=self.device, dtype=torch.float32)
        
        self.joint_upper_limits = torch.tensor([
            joint_limits['waist'][1], joint_limits['shoulder'][1], joint_limits['elbow'][1],
            joint_limits['yaw'][1], joint_limits['pitch'][1], joint_limits['roll'][1]
        ], device=self.device, dtype=torch.float32)
        
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
        self.env_base_positions = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.agent_actions = {
            agent: torch.zeros(self.num_envs, 3, device=self.device)
            for agent in self.cfg.possible_agents
        }
        
        self._initialize_physics_state()
        self._initialize_observation_cache()
        self._initialize_reward_state_caches()
        
        self.reward_components = {}
        self.stylus_body_idx = None
    
    def _initialize_physics_state(self) -> None:
        """Initialize physics interaction state variables."""
        self.human_forces_t = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_forces_t = torch.zeros(self.num_envs, 3, device=self.device)
        
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
        """Initialize reward system state caches."""
        self.rejoin_streak = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._goal_point = None
    
    def _initialize_components(self) -> None:
        """Initialize utility managers and components."""
        self.trajectory_manager = TrajectoryManager(
            device=self.device,
            params=self.params,
            num_envs=self.num_envs,
            env_base_positions=self.env_base_positions
        )
        
        self._goal_point = self.trajectory_manager.end_pos_local.to(self.device)
        print(f"[ENV] Goal point: {self._goal_point}")
        
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
        self._omni_robot = Articulation(self.cfg.phantom_omni)
        self.scene.articulations["phantom_omni"] = self._omni_robot
        
        self._constraint = RigidObject(self.cfg.constraint)
        self.scene.rigid_objects["constraint"] = self._constraint
        
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
            self.env_base_positions = self._omni_robot.data.root_link_pos_w.clone()
            self.trajectory_manager.env_base_positions = self.env_base_positions
            
    def _initialize_body_indices(self):
        """Initialize body indices for end-effector identification."""
        if self.stylus_body_idx is not None:
            return
            
        if not hasattr(self._omni_robot, 'body_names'):
            return
        
        target_name = "stylus"
        
        for i, name in enumerate(self._omni_robot.body_names):
            if target_name in name.lower():
                self.stylus_body_idx = i
                print(f"[ENV] Found stylus body: {name} (index {i})")
                return
        
        raise RuntimeError(f"stylus link not found in robot body names: {self._omni_robot.body_names}")
    
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

    def _progress_raw_signed_by_velocity(self) -> torch.Tensor:
        """Calculate velocity-based progress reward with adaptive magnitude."""
        t = self.trajectory_manager.line_direction
        v = self.stylus_vel_t1
        v_along = (v * t.unsqueeze(0)).sum(dim=-1)

        p = self.trajectory_manager.get_progress(self.stylus_pos_t1).clamp(0.0, 1.0)

        MIN_REWARD = 0.1
        MAX_REWARD = 0.2
        EPS_V = 1e-4

        reward_magnitude = MIN_REWARD + (MAX_REWARD - MIN_REWARD) * p

        pos = (v_along >  EPS_V).float()
        neg = (v_along < -EPS_V).float()
        
        return (pos - neg) * reward_magnitude
        
    def _pre_physics_step(self, actions: Dict[str, torch.Tensor]) -> None:
        """Pre-physics step processing with action validation and force application."""
        if hasattr(self, '_detail_actor_info') and self._detail_actor_info is not None:
            for agent in self.cfg.possible_agents:
                if agent in self._detail_actor_info['mean_actions']:
                    self.actor_mean_forces[agent] = self._detail_actor_info['mean_actions'][agent].clone()
                if agent in self._detail_actor_info['noise_actions']:
                    self.actor_noise_forces[agent] = self._detail_actor_info['noise_actions'][agent].clone()
        else:
            for agent in self.cfg.possible_agents:
                if agent in actions:
                    self.actor_mean_forces[agent] = actions[agent].clone()
                    self.actor_noise_forces[agent] = torch.zeros_like(actions[agent])
        
        for agent, action in actions.items():
            assert agent in self.cfg.possible_agents, f"Unknown agent: {agent}"
            assert action.shape == (self.num_envs, 3), f"Action shape mismatch for {agent}: expected ({self.num_envs}, 3), got {action.shape}"
            
            self.agent_actions[agent] = action
        
        self.robot_forces_t = self.agent_actions["robot"]
        self.human_forces_t = self.agent_actions["human"]
        
        self._apply_external_forces()
        self._enforce_joint_constraints()

    def set_detail_actor_info(self, detail_info: Dict[str, Any]) -> None:
        """Set detail information from MADDPG actor outputs for console logging."""
        self._detail_actor_info = detail_info
        
    def _apply_external_forces(self) -> None:
        """Apply external forces to the robot end-effector."""
        total_forces = self.robot_forces_t + self.human_forces_t
        stylus_quat = self._omni_robot.data.body_link_quat_w[:, self.stylus_body_idx, :]
        
        forces_local = quat_rotate_inverse(stylus_quat, total_forces)
        
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
        
        joint_pos = torch.clamp(joint_pos, self.joint_lower_limits, self.joint_upper_limits)
        
        joint_pos[:, 3:6] = self.fixed_end_joints.unsqueeze(0).expand(self.num_envs, -1)
        joint_vel[:, 3:6] = 0.0
        
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel)
        
    def _apply_action(self) -> None:
        """Apply actions to simulation."""
        self._omni_robot.write_data_to_sim()

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Compute observations for all agents and update state cache."""
        self.stylus_pos_t1 = self._get_stylus_position()
        self.stylus_vel_t1 = self._get_stylus_velocity()
        
        # Ensure constraint checker has robot reference for coordinate transformations
        if hasattr(self, 'constraint_checker') and hasattr(self, '_omni_robot'):
            self.constraint_checker._omni_robot = self._omni_robot
        
        current_base_positions = self._omni_robot.data.root_link_pos_w.clone()
        self.constraint_results_t1 = self.constraint_checker.analyze_constraint_state_batch(
            self.stylus_pos_t1, current_base_positions
        )
        self.safety_distances_t1 = self.constraint_results_t1['distances_constraint']
        self.is_violating_t1 = self.constraint_results_t1['is_overlapping']
        self.normal_t1 = self.constraint_results_t1['normal_vectors']

        constraint_distances = self.safety_distances_t1.unsqueeze(-1)

        obs = torch.cat([
            self.stylus_pos_t1,       # End-effector position (3)
            self.stylus_vel_t1,       # End-effector velocity (3) 
            constraint_distances,     # Distance measurements (1)
        ], dim=-1)
        
        observations = {}
        for agent in self.cfg.possible_agents:
            observations[agent] = obs
            
        return observations

    def _build_zone_masks(self):
        """Build zone masks for four-zone reward system."""
        D, O = 0.0075, 0.015
        d = self.safety_distances_t1
        
        surface = (d > D) & (d < O)
        outside = (d >= O) 
        danger = (d <= D)
        
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
        
        prog_raw = self._progress_raw_signed_by_velocity()
        
        deviations, _ = self.trajectory_manager.get_deviation(self.stylus_pos_t1)
        dev_raw = -torch.clamp(deviations * 5, min=0.0, max=0.3)
        
        wp = self._get_component_weight('A', 'progress', agent)
        wd = self._get_component_weight('A', 'deviation', agent)
        zw = self._get_zone_weight('A', agent)

        prog_contrib = wp * prog_raw
        dev_contrib = wd * dev_raw
        zone_total = zw * (prog_contrib + dev_contrib)

        out = torch.zeros_like(prog_raw)
        out[outside] = zone_total[outside]

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
        
        beta = 6.0e3
        d = self.safety_distances_t1
        gap_raw = -beta * (d - 0.010) ** 2
        
        gamma = 1.0
        t = self.trajectory_manager.line_direction
        n = self.normal_t1
        
        t_dot_n = (t.unsqueeze(0) * n).sum(dim=-1, keepdim=True)
        t_surf = t.unsqueeze(0) - t_dot_n * n
        t_surf = t_surf / torch.norm(t_surf, dim=-1, keepdim=True).clamp(min=1e-8)
        
        v_surf = (self.stylus_vel_t1 * t_surf).sum(dim=-1)
        surf_raw = gamma * v_surf
        
        lam = 2.0
        v_dot_n = (self.stylus_vel_t1 * self.normal_t1).sum(dim=-1)
        inward_raw = -lam * torch.clamp(v_dot_n, min=0.0)
        
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
        """Zone C (Danger): Binary penalty system for dangerous proximity."""
        outside, surface, danger, rejoin = masks
        
        R_overlap_raw = -0.3
        w_overlap = self._get_component_weight('C', 'overlap', agent)
        overlap_contrib = w_overlap * R_overlap_raw

        R_off = -0.2
        offpen_raw = R_off * torch.ones_like(self.safety_distances_t1)
        lam = 2.0
        v_dot_n = (self.stylus_vel_t1 * self.normal_t1).sum(dim=-1)
        inward_raw = -lam * torch.clamp(v_dot_n, min=0.0)
        wo = self._get_component_weight('C', 'offpen', agent)
        wi = self._get_component_weight('C', 'inward', agent)
        too_close_contrib = (wo * offpen_raw) + (wi * inward_raw)

        zone_c_total_contrib = torch.where(
            self.is_violating_t1,
            torch.tensor(overlap_contrib, device=self.device),
            too_close_contrib
        )

        zw = self._get_zone_weight('C', agent)
        zone_total = zw * zone_c_total_contrib
        
        out = torch.zeros_like(zone_total)
        out[danger] = zone_total[danger]

        RC, device = self.reward_components, zone_total.device
        RC[f'zoneC_weight_{agent}'] = torch.tensor(zw, device=device)
        RC[f'zoneC_total_{agent}'] = zone_total

        RC[f'zoneC_offpen_{agent}_contrib'] = torch.where(self.is_violating_t1, torch.zeros_like(zone_total), wo * offpen_raw)
        RC[f'zoneC_inward_{agent}_contrib'] = torch.where(self.is_violating_t1, torch.zeros_like(zone_total), wi * inward_raw)
        RC[f'zoneC_overlap_{agent}_contrib'] = torch.where(self.is_violating_t1, torch.tensor(overlap_contrib, device=self.device), torch.zeros_like(zone_total))

        RC[f'zoneC_offpen_{agent}_raw'] = offpen_raw
        RC[f'zoneC_offpen_{agent}_weight'] = torch.tensor(wo, device=device)
        RC[f'zoneC_inward_{agent}_raw'] = inward_raw
        RC[f'zoneC_inward_{agent}_weight'] = torch.tensor(wi, device=device)
        RC[f'zoneC_overlap_{agent}_raw'] = torch.full_like(zone_total, R_overlap_raw)
        RC[f'zoneC_overlap_{agent}_weight'] = torch.tensor(w_overlap, device=device)
        
        return out
    
    def _zone_D_reward(self, masks, agent: str):
        """Zone D (Rejoin): Trajectory recovery with enhanced deviation and outward movement."""
        outside, surface, danger, rejoin = masks

        prog_raw = self._progress_raw_signed_by_velocity()

        deviations, _ = self.trajectory_manager.get_deviation(self.stylus_pos_t1)
        dev_raw = -torch.clamp(deviations * 5, min=0.0, max=0.3)

        lam = 2.0
        v_dot_n = (self.stylus_vel_t1 * self.normal_t1).sum(dim=-1)
        inward_raw = torch.where(
            v_dot_n >= 0,
            -lam * v_dot_n,
            lam * (-v_dot_n)
        )

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

    def _globals_for_agent(self, agent: str):
        """Calculate global rewards for agent with three-piece structure."""
        RC, dev = self.reward_components, self.device

        z_raw = self._calculate_z_penalty()
        zw = self._get_weight(f'zpenalty_{agent}', 0.0)
        RC[f'global_zpenalty_{agent}_raw'] = z_raw
        RC[f'global_zpenalty_{agent}_weight'] = torch.tensor(zw, device=dev)
        RC[f'global_zpenalty_{agent}_contrib'] = z_raw * zw

        c_raw = self._calculate_completion_reward()
        cw = self._get_weight(f'completion_{agent}', 0.0)
        RC[f'global_completion_{agent}_raw'] = c_raw
        RC[f'global_completion_{agent}_weight'] = torch.tensor(cw, device=dev)
        RC[f'global_completion_{agent}_contrib'] = c_raw * cw

        t_raw = self._calculate_time_efficiency_reward()
        tw = self._get_weight(f'timeeff_{agent}', 0.0)
        RC[f'global_timeeff_{agent}_raw'] = t_raw
        RC[f'global_timeeff_{agent}_weight'] = torch.tensor(tw, device=dev)
        RC[f'global_timeeff_{agent}_contrib'] = t_raw * tw

        p_raw = self._calculate_potential_reward()
        pw = self._get_weight(f'potential_{agent}', 0.0)
        RC[f'global_potential_{agent}_raw'] = p_raw
        RC[f'global_potential_{agent}_weight'] = torch.tensor(pw, device=dev)
        RC[f'global_potential_{agent}_contrib'] = p_raw * pw

        f_raw = self._calculate_force_penalties()[agent]
        fw = self._get_weight(f'forceeff_{agent}', 0.0)
        RC[f'{agent}force_raw'] = f_raw
        RC[f'{agent}force_weight'] = torch.tensor(fw, device=dev)
        RC[f'{agent}force_contrib'] = f_raw * fw

        other = 'human' if agent == 'robot' else 'robot'
        aw_key = 'humanaware_robot' if agent == 'robot' else 'robotaware_human'
        aw_raw = self._calculate_force_penalties()[other]
        aw_w = self._get_weight(aw_key, 0.0)
        label = 'humanaware' if agent == 'robot' else 'robotaware'
        RC[f'{label}_raw'] = aw_raw
        RC[f'{label}_weight'] = torch.tensor(aw_w, device=dev)
        RC[f'{label}_contrib'] = aw_raw * aw_w

    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        """Four-zone reward system with robot/human symmetric rewards."""
        self.reward_components = {}
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
                RC[f'global_potential_{agent}_contrib'] +
                RC[f'{agent}force_contrib'] +
                (RC['humanaware_contrib'] if agent=='robot' else RC['robotaware_contrib'])
            )
            return (robot_zones if agent=='robot' else human_zones) + globals_sum

        rewards = {'robot': _agent_total('robot'), 'human': _agent_total('human')}

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
        
        default_keys = ['deviation', 'progress_ratio', 'min_safety_distance', 'completion_reward']
        for key in default_keys:
            if key not in self.reward_components:
                self.reward_components[key] = torch.zeros(self.num_envs, device=self.device)

        if hasattr(self, "step_tracer") and self.step_tracer is not None:
            current_step = self._trainer_global_step if self._trainer_global_step is not None else 0
            self.step_tracer.maybe_print_step(self, rewards, current_step)
        
        return rewards

    def _calculate_potential_reward(self) -> torch.Tensor:
        """Concave increasing potential based on distance to final setpoint."""
        d = torch.norm(
            self.stylus_pos_t1 - self.trajectory_manager.end_pos_local.unsqueeze(0),
            dim=-1
        )
        d0 = max(float(self.trajectory_manager.total_distance), 1e-8)
        u = torch.clamp(d / d0, min=0.0, max=1.0)
        return torch.sqrt(1 - u)

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
            torch.full_like(self.safety_distances_t1, 1.0),
            torch.zeros_like(self.safety_distances_t1)
        )

    def _calculate_time_efficiency_reward(self) -> torch.Tensor:
        """Calculate time efficiency reward based on episode progress."""
        max_steps = 1200
        current_steps = self.episode_length_buf.to(self.device).float()
        time_efficiency = (max_steps - current_steps) / max_steps
        time_efficiency = torch.clamp(time_efficiency, min=0.0, max=1.0)
        return time_efficiency * 3.0

    def _get_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Determine termination and truncation conditions for all agents."""
        z_below_zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.enable_z_termination:
            z_below_zero = self.stylus_pos_t1[:, 2] < self.min_z_pos
        
        edge_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.enable_edge_termination:
            edge_collision = self.safety_distances_t1 <= self.safety_distance_threshold
        
        final_reached = self.trajectory_manager.is_final_setpoint_reached(self.stylus_pos_t1)
        
        terminated_condition = z_below_zero | edge_collision | final_reached
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
        
        joint_pos = torch.zeros((num_resets, 6), device=self.device)
        joint_pos[:, 0] = -0.96    # waist
        joint_pos[:, 1] = 0.0      # shoulder
        joint_pos[:, 2] = 1.0      # elbow
        joint_pos[:, 3] = 0.0      # yaw
        joint_pos[:, 4] = 2.0944   # pitch (~120 degrees)
        joint_pos[:, 5] = 0.0      # roll
        
        joint_vel = torch.zeros((num_resets, 6), device=self.device)
        
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        for agent in self.cfg.possible_agents:
            self.agent_actions[agent][env_ids] = 0.0
        
        self.human_forces_t[env_ids] = 0.0
        self.robot_forces_t[env_ids] = 0.0
        self.safety_distances_t1[env_ids] = 0.01
        self.is_violating_t1[env_ids] = False
        self.normal_t1[env_ids] = torch.zeros((num_resets, 3), device=self.device)

        self.rejoin_streak[env_ids] = 0

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
        """Get stylus velocity relative to robot base frame."""
        if self.stylus_body_idx is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        
        vel_world = self._omni_robot.data.body_link_lin_vel_w[:, self.stylus_body_idx, :]
        base_quat = self._omni_robot.data.root_link_quat_w
        vel_local = quat_rotate_inverse(base_quat, vel_world)
        
        return vel_local