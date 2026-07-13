"""
Human-robot collaborative surgical MARL environment for Epigraph algorithm.
Features physics-based constraints, four-zone rewards with task/safe decomposition.

KEY DESIGN (Pure Epigraph: Self-Contained):
- Environment maintains its own step counter for logging (no trainer dependency)
- step() returns standard Gym format with r_task/r_safe in info dict
- StepTracer is initialized and controlled by env based on YAML config
- Configuration injected by trainer/play script ensures train-eval consistency
- Clean separation: env handles physics/rewards/logging, trainer handles training
"""

from __future__ import annotations

import os
import copy
import yaml
import torch
import numpy as np
from typing import Any, Dict, Optional, Tuple
from collections import defaultdict

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.math import quat_rotate_inverse

from .surgical_epigraph_env_cfg import SurgicalEpigraphEnvCfg
from .utils import CompleteConstraintChecker, TrajectoryManager, safe_get_rc, compose_task_safe_from_rc, StepTracer


class SurgicalEpigraphEnv(DirectMARLEnv):
    """
    Multi-agent surgical robot environment for Epigraph safe MARL.
    
    Features:
    - Multi-agent force control for surgical tasks
    - Physics-based constraint checking and collision detection
    - Four-zone reward system with task/safe decomposition
    - Potential field reward for trajectory guidance
    - Per-agent reward decomposition for Epigraph algorithm
    
    Key design principles (Pure Epigraph):
    - Environment maintains own step counter (_env_debug_step_counter)
    - step() returns standard Gym format with r_task/r_safe in info dict
    - StepTracer self-prints based on YAML config (no trainer control needed)
    - Configuration injected ensures train-eval consistency
    - Completely independent from RMAPPO or other algorithms
    """
    
    cfg: SurgicalEpigraphEnvCfg
    
    def __init__(self, cfg: SurgicalEpigraphEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize Epigraph surgical MARL environment."""
        super().__init__(cfg, render_mode, **kwargs)
        
        # Environment's own step counter (for self-contained logging)
        self._env_debug_step_counter = 0
        # Trainer-provided step counter (optional override)
        self._trainer_global_step = 0
        
        self._setup_core_configuration()
        self._initialize_state_variables()
        self._initialize_components()
        self._setup_gymnasium_spaces()
        
        # Initialize StepTracer based on YAML config
        enable_console_logging = bool(
            self.params.get("logging", {}).get("enable_console_logging", False)
        )
        print_every_steps = self.params.get("logging", {}).get("print_every_steps", 10)
        max_envs_to_print = self.params.get("logging", {}).get("max_envs_to_print", 2)
        strict_masks = bool(self.params.get("logging", {}).get("strict_masks", False))
        
        self.step_tracer = StepTracer(
            num_envs=self.num_envs,
            device=self.device,
            enable_console_logging=enable_console_logging,
            print_every_steps=print_every_steps,
            max_envs_to_print=max_envs_to_print,
            strict_masks=strict_masks,
        )
        self._last_z_snapshot = None
        self._current_zone_masks = None
        
        if enable_console_logging:
            strict_msg = " (strict_masks=True)" if strict_masks else ""
            print(f"[ENV/EPIGRAPH] StepTracer enabled (print_every={print_every_steps}){strict_msg}")
        else:
            print("[ENV/EPIGRAPH] StepTracer disabled (set logging.enable_console_logging=true to enable)")
    
    def _setup_core_configuration(self) -> None:
        """
        Setup core configuration from injected params or default YAML.
        
        Priority:
        1. Use self.params if injected by trainer/play script
        2. Fall back to default YAML in same directory
        """
        cfg_params = getattr(self.cfg, "params", None)
        if not hasattr(self, "params") or not isinstance(getattr(self, "params", None), dict):
            if isinstance(cfg_params, dict):
                self.params = cfg_params
                print("[ENV/EPIGRAPH] Using configuration injected via cfg.params")
            else:
                self.params = None

        if isinstance(getattr(self, "params", None), dict):
            print("[ENV/EPIGRAPH] Using configuration injected by trainer/play script")
        else:
            print("[ENV/EPIGRAPH] No injected config, loading default YAML")
            default_cfg_path = os.path.join(
                os.path.dirname(__file__),
                "agents",
                "training_params_epigraph.yaml",
            )
            if not os.path.exists(default_cfg_path):
                raise FileNotFoundError(f"[ENV/EPIGRAPH] Config file not found at {default_cfg_path}")
            
            with open(default_cfg_path, "r") as f:
                self.params = yaml.safe_load(f)
            print(f"[ENV/EPIGRAPH] Loaded default params from {default_cfg_path}")
        
        self._normalize_reward_weights_schema()

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self._load_constraint_parameters()
        self._load_safety_parameters()
        self._load_termination_parameters()
        
        print(f"[ENV/EPIGRAPH] Episode length: {self.cfg.episode_length_s}s")

    # 扁平化奖励权重
    def _normalize_reward_weights_schema(self) -> None:
        """Flatten structured reward weights into legacy per-agent keys."""
        reward_params = self.params.get("reward_parameters") if isinstance(self.params, dict) else None
        if not isinstance(reward_params, dict):
            return

        weights = reward_params.get("weights")
        if not isinstance(weights, dict):
            return

        if "robot" not in weights or "human" not in weights:
            return

        structured_weights = copy.deepcopy(weights)

        key_map = {
            "zoneA_weight": ("task", "zoneA", "weight"),
            "zoneA_progress": ("task", "zoneA", "progress"),
            "zoneA_deviation": ("task", "zoneA", "deviation"),
            "zoneB_weight": ("task", "zoneB", "weight"),
            "zoneB_gap": ("task", "zoneB", "gap"),
            "zoneB_surftangent": ("task", "zoneB", "surftangent"),
            "zoneB_inward": ("safe", "zoneB", "inward"),
            "zoneC_weight": ("safe", "zoneC", "weight"),
            "zoneC_offpen": ("safe", "zoneC", "offpen"),
            "zoneC_inward": ("safe", "zoneC", "inward"),
            "zoneC_overlap": ("safe", "zoneC", "overlap"),
            "zoneD_weight": ("task", "zoneD", "weight"),
            "zoneD_progress": ("task", "zoneD", "progress"),
            "zoneD_deviation": ("task", "zoneD", "deviation"),
            "zoneD_inward": ("safe", "zoneD", "inward"),
            "zpenalty": ("safe", "zpenalty"),
            "completion": ("task", "completion"),
            "timeeff": ("task", "timeeff"),
            "potential": ("task", "potential"),
            "forceeff": ("task", "forceeff"),
            "humanaware": ("task", "humanaware"),
            "robotaware": ("task", "robotaware"),
        }

        flattened: Dict[str, float] = {}

        for agent in ("robot", "human"):
            agent_data = structured_weights.get(agent, {})

            def fetch(path: Tuple[str, ...]) -> float:
                value: Any = agent_data
                for part in path:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return 0.0
                return float(value) if isinstance(value, (int, float)) else 0.0

            for base_key, path in key_map.items():
                flattened[f"{base_key}_{agent}"] = fetch(path)

        reward_params["weights_structured"] = structured_weights
        reward_params["weights"] = flattened

    # 约束参数
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
    
    # 安全与终止参数
    def _load_safety_parameters(self) -> None:
        """Load safety-related parameters."""        
        self.collision_threshold = self.params['constraint_geometry']['collision_threshold']
    
    def _load_termination_parameters(self) -> None:
        """Load episode termination condition parameters."""
        term_config = self.params.get('termination_conditions', {})
        self.enable_z_termination = term_config.get('z_below_zero', False)
        self.enable_edge_termination = term_config.get('edge_collision', True)
        self.safety_distance_threshold = term_config.get('safety_distance_threshold', 0.0)
    
    # 初始化环境状态
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
    
    # 初始化物理交互状态
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
    
    #初始化Observation 缓存，在get obs的时候才起效果
    def _initialize_observation_cache(self) -> None:
        """Initialize observation caching variables for efficient updates."""
        self.stylus_pos_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.stylus_vel_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.safety_distances_t1 = torch.ones(self.num_envs, device=self.device) * 0.01
        self.is_violating_t1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.normal_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.constraint_results_t1 = None

    # 初始化Reward 状态
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
        print(f"[ENV/EPIGRAPH] Goal point: {self._goal_point}")
        
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
                print(f"[ENV/EPIGRAPH] Found stylus body: {name} (index {i})")
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
        """Set detail information from actor outputs for console logging."""
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

        # Ensure constraint checker has robot reference
        if hasattr(self, 'constraint_checker') and hasattr(self, '_omni_robot'):
            self.constraint_checker._omni_robot = self._omni_robot

        current_base_positions = self._omni_robot.data.root_link_pos_w.clone()
        self.constraint_results_t1 = self.constraint_checker.analyze_constraint_state_batch(
            self.stylus_pos_t1, current_base_positions
        )

        self.safety_distances_t1 = self.constraint_results_t1['distances_constraint']
        self.is_violating_t1 = self.constraint_results_t1['is_overlapping']
        self.normal_t1 = self.constraint_results_t1['normal_vectors']

        # Compute direct-style 6D observation: deviation + velocity + safety distance + progress
        deviations, _ = self.trajectory_manager.get_deviation(self.stylus_pos_t1)
        deviations = deviations.unsqueeze(-1)

        progress_ratio = self.trajectory_manager.get_progress(self.stylus_pos_t1)
        progress_ratio = progress_ratio.unsqueeze(-1)

        obs_raw = torch.cat([
            deviations,
            self.stylus_vel_t1,
            self.safety_distances_t1.unsqueeze(-1),
            progress_ratio,
        ], dim=-1)

        obs_dict = {agent: obs_raw.clone() for agent in self.cfg.possible_agents}

        return obs_dict

    def _build_zone_masks(self) -> Dict[str, torch.Tensor]:
        """Build four-zone masks aligned with rMAPPO definition."""
        D, O = 0.0075, 0.015
        distances = self.safety_distances_t1

        outside = distances >= O
        surface = (distances > D) & (distances < O)
        danger = distances <= D

        t = self.trajectory_manager.line_direction  # unit vector
        if self._goal_point is not None:
            goal_vec = self._goal_point.unsqueeze(0) - self.stylus_pos_t1
            g = goal_vec / torch.norm(goal_vec, dim=-1, keepdim=True).clamp(min=1e-8)
        else:
            g = t.unsqueeze(0).expand_as(self.stylus_pos_t1)

        # Alignment thresholds (taken from rMAPPO env)
        c1, c2 = 0.90, 0.866
        align_goal = (g * t.unsqueeze(0)).sum(dim=-1) >= c1
        oppose_norm = (self.normal_t1 * t.unsqueeze(0)).sum(dim=-1) <= -c2
        rejoin_geom = surface & align_goal & oppose_norm

        if not hasattr(self, "rejoin_streak"):
            self.rejoin_streak = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        self.rejoin_streak = torch.where(
            rejoin_geom,
            self.rejoin_streak + 1,
            torch.zeros_like(self.rejoin_streak),
        )
        rejoin = self.rejoin_streak >= 10

        masks = {
            "A": outside,
            "B": surface & (~rejoin),
            "C": danger,
            "D": rejoin,
        }
        return masks

    def _progress_raw_signed_by_velocity(self) -> torch.Tensor:
        """Calculate signed progress reward based on velocity alignment."""
        t = self.trajectory_manager.line_direction
        v = self.stylus_vel_t1
        v_along = (v * t.unsqueeze(0)).sum(dim=-1)

        p = self.trajectory_manager.get_progress(self.stylus_pos_t1)

        MIN_REWARD, MAX_REWARD, EPS_V = 0.1, 0.2, 1e-4
        reward_magnitude = MIN_REWARD + (MAX_REWARD - MIN_REWARD) * p

        pos = (v_along >  EPS_V).float()
        neg = (v_along < -EPS_V).float()
        base = (pos - neg) * reward_magnitude

        overrun = p > 1.0

        return torch.where(overrun, -reward_magnitude, base)

    def _zone_A_reward(self, masks: Dict[str, torch.Tensor], agent: str) -> torch.Tensor:
        """Zone A (Track): Outside obstacle boundary - Progress and deviation rewards."""
        if isinstance(masks, dict):
            outside = masks["A"]
            surface = masks["B"]
            danger = masks["C"]
            rejoin = masks["D"]
        else:
            outside, surface, danger, rejoin = masks

        prog_raw = self._progress_raw_signed_by_velocity()

        deviations, _ = self.trajectory_manager.get_deviation(self.stylus_pos_t1)
        dev_raw = -torch.clamp(deviations * 5, min=0.0, max=0.3)

        wp = self._get_component_weight('A', 'progress', agent)
        wd = self._get_component_weight('A', 'deviation', agent)
        zw = self._get_zone_weight('A', agent)

        prog_contrib = torch.where(outside, wp * prog_raw, torch.zeros_like(prog_raw))
        dev_contrib = torch.where(outside, wd * dev_raw, torch.zeros_like(dev_raw))
        zone_contrib = prog_contrib + dev_contrib
        zone_total = zw * zone_contrib

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

    def _zone_B_reward(self, masks: Dict[str, torch.Tensor], agent: str) -> torch.Tensor:
        """Zone B (Surface): Near obstacle - Gap optimization and surface following."""
        if isinstance(masks, dict):
            outside = masks["A"]
            surface = masks["B"]
            danger = masks["C"]
            rejoin = masks["D"]
        else:
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

        surface_only = surface & (~rejoin)
        wg = self._get_component_weight('B', 'gap', agent)
        ws = self._get_component_weight('B', 'surftangent', agent)
        wi = self._get_component_weight('B', 'inward', agent)
        zw = self._get_zone_weight('B', agent)

        gap_contrib = torch.where(surface_only, wg * gap_raw, torch.zeros_like(gap_raw))
        surf_contrib = torch.where(surface_only, ws * surf_raw, torch.zeros_like(surf_raw))
        inward_contrib = torch.where(surface_only, wi * inward_raw, torch.zeros_like(inward_raw))
        zone_total = zw * (gap_contrib + surf_contrib + inward_contrib)

        out = torch.zeros_like(gap_raw)
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

    def _zone_C_reward(self, masks: Dict[str, torch.Tensor], agent: str) -> torch.Tensor:
        """Zone C (Danger): Binary penalty system for dangerous proximity."""
        if isinstance(masks, dict):
            outside = masks["A"]
            surface = masks["B"]
            danger = masks["C"]
            rejoin = masks["D"]
        else:
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

        mask_danger = danger
        mask_overlap = mask_danger & self.is_violating_t1
        mask_safe = mask_danger & (~self.is_violating_t1)

        offpen_contrib = torch.where(mask_safe, wo * offpen_raw, torch.zeros_like(offpen_raw))
        inward_contrib = torch.where(mask_safe, wi * inward_raw, torch.zeros_like(inward_raw))
        overlap_tensor = torch.full_like(offpen_raw, overlap_contrib)
        overlap_contrib_tensor = torch.where(mask_overlap, overlap_tensor, torch.zeros_like(overlap_tensor))

        zone_c_total_contrib = offpen_contrib + inward_contrib + overlap_contrib_tensor

        zw = self._get_zone_weight('C', agent)
        zone_total = zw * zone_c_total_contrib

        out = torch.zeros_like(zone_total)
        out[mask_danger] = zone_total[mask_danger]

        RC, device = self.reward_components, zone_total.device
        RC[f'zoneC_weight_{agent}'] = torch.tensor(zw, device=device)
        RC[f'zoneC_total_{agent}'] = zone_total

        RC[f'zoneC_offpen_{agent}_contrib'] = offpen_contrib
        RC[f'zoneC_inward_{agent}_contrib'] = inward_contrib
        RC[f'zoneC_overlap_{agent}_contrib'] = overlap_contrib_tensor

        RC[f'zoneC_offpen_{agent}_raw'] = offpen_raw
        RC[f'zoneC_offpen_{agent}_weight'] = torch.tensor(wo, device=device)
        RC[f'zoneC_inward_{agent}_raw'] = inward_raw
        RC[f'zoneC_inward_{agent}_weight'] = torch.tensor(wi, device=device)
        RC[f'zoneC_overlap_{agent}_raw'] = torch.full_like(zone_total, R_overlap_raw)
        RC[f'zoneC_overlap_{agent}_weight'] = torch.tensor(w_overlap, device=device)

        return out

    def _zone_D_reward(self, masks: Dict[str, torch.Tensor], agent: str) -> torch.Tensor:
        """Zone D (Rejoin): Trajectory recovery with enhanced deviation and outward movement."""
        if isinstance(masks, dict):
            outside = masks["A"]
            surface = masks["B"]
            danger = masks["C"]
            rejoin = masks["D"]
        else:
            outside, surface, danger, rejoin = masks

        prog_raw = self._progress_raw_signed_by_velocity()

        deviations, _ = self.trajectory_manager.get_deviation(self.stylus_pos_t1)
        dev_raw = -torch.clamp(deviations * 5, min=0.0, max=0.3)

        v_dot_n = (self.stylus_vel_t1 * self.normal_t1).sum(dim=-1)
        lam = 2.0
        inward_raw = torch.where(
            v_dot_n >= 0,
            -lam * v_dot_n,
            lam * (-v_dot_n)
        )

        wp = self._get_component_weight('D', 'progress', agent)
        wd = self._get_component_weight('D', 'deviation', agent)
        wi = self._get_component_weight('D', 'inward', agent)
        zw = self._get_zone_weight('D', agent)

        prog_contrib = torch.where(rejoin, wp * prog_raw, torch.zeros_like(prog_raw))
        dev_contrib = torch.where(rejoin, wd * dev_raw, torch.zeros_like(dev_raw))
        inw_contrib = torch.where(rejoin, wi * inward_raw, torch.zeros_like(inward_raw))
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
        """Calculate global rewards for agent with five components."""
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
        self._current_zone_masks = masks

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

        # StepTracer: Use environment's own step counter (self-contained logging)
        if (
            hasattr(self, "step_tracer") and
            self.step_tracer is not None and
            self.step_tracer.enable_console_logging
        ):
            r_task_dict, r_safe_risk_dict = self._compose_task_and_safe()
            r_task_for_tracer = {
                agent: r_task_dict[agent].detach().unsqueeze(-1)
                for agent in self.cfg.possible_agents
            }
            r_safe_cost = {
                agent: torch.relu(-r_safe_risk_dict[agent]).detach().unsqueeze(-1)
                for agent in self.cfg.possible_agents
            }
            z_snapshot = getattr(self, "_last_z_snapshot", None)
            if isinstance(z_snapshot, torch.Tensor):
                z_tensor = z_snapshot.detach()
                if z_tensor.dim() == 1:
                    z_tensor = z_tensor.unsqueeze(-1)
                z_for_tracer = {
                    agent: z_tensor.clone()
                    for agent in self.cfg.possible_agents
                }
            else:
                z_for_tracer = None
            trainer_step = getattr(self, "_trainer_global_step", None)
            global_step = self._env_debug_step_counter if trainer_step is None else trainer_step
            self.step_tracer.maybe_print_step(
                env=self,
                r_task=r_task_for_tracer,
                r_safe_cost=r_safe_cost,
                z=z_for_tracer,
                global_step=global_step,
                force_print=False,
            )

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

    # ============================================================================
    # EPIGRAPH-SPECIFIC: Reward decomposition into task/safe
    # ============================================================================
    
    def _compose_task_and_safe(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Decompose reward_components into r_task and r_safe for each agent.
        
        r_task = Zone A/B/D (exclude inward) + global positive (completion/potential) + force penalties
        r_safe = Zone C + Zone B/D inward + zpenalty (optional)
        """
        rc = self.reward_components
        if rc is None or len(rc) == 0:
            zeros = torch.zeros(self.num_envs, device=self.device)
            return ({ag: zeros for ag in self.cfg.possible_agents},
                    {ag: zeros for ag in self.cfg.possible_agents})

        r_task, r_safe = {}, {}
        for agent in self.cfg.possible_agents:
            t, s = compose_task_safe_from_rc(
                rc=rc,
                agent=agent,
                device=self.device,
                num_envs=self.num_envs,
            )
            r_task[agent] = t
            r_safe[agent] = s

        return r_task, r_safe

    # ============================================================================
    # FIXED: step() returns standard Gym format
    # ============================================================================

    def step(self, actions: Dict[str, torch.Tensor]) -> tuple:
        """
        Execute one environment step with Epigraph reward decomposition.
        
        Returns standard Gym format (obs, rewards, terminated, truncated, info)
        with info carrying r_task / r_safe for Epigraph training.
        
        Returns:
            obs: Dict[agent, tensor] - Observations
            rewards: Dict[agent, tensor] - Total rewards (for logging)
            terminated: Dict[agent, tensor] - Episode termination flags
            truncated: Dict[agent, tensor] - Episode truncation flags
            info: dict - Additional information including:
                - info["r_task"]: Dict[agent, [N,1]] - Task quality rewards
                - info["r_safe"]: Dict[agent, [N,1]] - Safety costs (>=0)
        """
        # 1. Call parent DirectMARLEnv.step() to handle physics and standard processing
        obs, rewards, terminated, truncated, info = super().step(actions)
        
        # 2. Decompose rewards for Epigraph using compose_task_safe_from_rc
        r_task_robot, r_safe_risk_robot = compose_task_safe_from_rc(
            rc=self.reward_components,
            agent="robot",
            device=self.device,
            num_envs=self.num_envs,
        )
        r_task_human, r_safe_risk_human = compose_task_safe_from_rc(
            rc=self.reward_components,
            agent="human",
            device=self.device,
            num_envs=self.num_envs,
        )
        
        # Ensure shape is [num_envs, 1]
        r_task_robot = r_task_robot.view(self.num_envs, 1)
        r_task_human = r_task_human.view(self.num_envs, 1)
        
        # 3. Convert safety risk (negative=dangerous) to safety cost (positive=dangerous)
        r_safe_cost_robot = torch.relu(-r_safe_risk_robot).view(self.num_envs, 1)
        r_safe_cost_human = torch.relu(-r_safe_risk_human).view(self.num_envs, 1)
        
        # 4. Prepare info dict with r_task and r_safe (trainer will extract these)
        info = dict(info) if info is not None else {}
        info.update({
            "r_task": {
                "robot": r_task_robot,   # [num_envs, 1]
                "human": r_task_human,   # [num_envs, 1]
            },
            "r_safe": {
                "robot": r_safe_cost_robot,  # [num_envs, 1], >=0 (cost)
                "human": r_safe_cost_human,  # [num_envs, 1], >=0 (cost)
            },
            # Additional debugging information for logging / tracer
            "is_violating": self.is_violating_t1.clone(),
            "safety_distance": self.safety_distances_t1.clone(),
            "rejoin_streak": self.rejoin_streak.clone(),
            "progress_ratio": self.reward_components.get(
                "progress_ratio", 
                torch.zeros(self.num_envs, device=self.device)
            ).clone(),
        })
        
        # 5. Increment environment's own step counter (for self-contained logging)
        self._env_debug_step_counter += 1
        
        # Return standard Gym format
        return obs, rewards, terminated, truncated, info

    # ============================================================================
    # Remaining methods (identical to reference environment)
    # ============================================================================

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

        if hasattr(self, "_last_z_snapshot"):
            self._last_z_snapshot = None
        self._trainer_global_step = 0
        self._current_zone_masks = None
        
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
