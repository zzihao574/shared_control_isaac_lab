"""Surgical direct environment - Y-axis linear movement with optimized state management"""

from __future__ import annotations

import torch
import numpy as np
import yaml
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform, quat_rotate_inverse

from .surgical_direct_env_cfg import SurgicalDirectEnvCfg


class SurgicalDirectEnv(DirectRLEnv):
    """Surgical environment: Y-axis movement (0.14,-0.2,0.03) → (0.14,0.2,0.03)"""
    
    cfg: SurgicalDirectEnvCfg
    
    def __init__(self, cfg: SurgicalDirectEnvCfg, render_mode: str | None = None, 
                 training_params_path: str = None, **kwargs):
        # Initialize body indices before calling super().__init__
        self.stylus_body_idx = None
        
        # Load parameters BEFORE super().__init__ since _setup_scene needs them
        self.params = self._load_training_params(training_params_path)
        self.dt = cfg.sim.dt * cfg.decimation
        
        # Initialize parameters needed by _setup_scene
        self._load_yaml_parameters(cfg)
        
        # Now call super().__init__ which will call _setup_scene
        super().__init__(cfg, render_mode, **kwargs)
        
        # Physics interfaces
        try:
            from omni.physx.bindings._physx import acquire_physx_attachment_interface, acquire_physx_scene_query_interface
            self.physics_attachment = acquire_physx_attachment_interface()
            self.physics_scene_query = acquire_physx_scene_query_interface()
        except ImportError:
            self.physics_attachment = None
            self.physics_scene_query = None
        
        # Initialize trajectory and state variables after scene is created
        self._init_trajectory()
        self._init_state_variables()
        
        # Manual call to _post_init to ensure stylus index is set
        if hasattr(self, '_omni_robot'):
            self._post_init()
        
        print(f"[INFO] Surgical environment initialized:")
        print(f"  - Parallel environments: {self.num_envs}")
        print(f"  - Trajectory: Y-axis {self.start_pos.cpu().numpy()} → {self.end_pos.cpu().numpy()}")
        print(f"  - Human equilibrium: y≤0→{self.eq_middle.cpu().numpy()}, y>0→{self.eq_end.cpu().numpy()}")
        print(f"  - Observation: [x, ẋ, q, q̇, f] = 21D")
    
    def _load_training_params(self, path: str = None) -> dict:
        """Load training parameters from YAML file"""
        if path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, "agents", "training_params.yaml")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training params file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_yaml_parameters(self, cfg):
        """Load all parameters directly from YAML"""
        # Get device and num_envs from config
        device = torch.device(cfg.sim.device if torch.cuda.is_available() else "cpu")
        num_envs = cfg.scene.num_envs
        
        # Trajectory parameters
        traj = self.params['trajectory']
        self.start_pos = torch.tensor(traj['start_point'], device=device, dtype=torch.float32)
        self.end_pos = torch.tensor(traj['end_point'], device=device, dtype=torch.float32)
        
        # Human equilibrium parameters
        eq = self.params['human_equilibrium']
        self.eq_middle = torch.tensor(eq['middle_point'], device=device, dtype=torch.float32)
        self.eq_end = torch.tensor(eq['end_point'], device=device, dtype=torch.float32)
        
        # Human dynamics parameters
        hd = self.params['human_dynamics']
        self.base_damping = torch.tensor(hd['base_damping'], device=device, dtype=torch.float32)
        self.damping_var = torch.tensor(hd['damping_variation'], device=device, dtype=torch.float32)
        self.base_stiffness = torch.tensor(hd['base_stiffness'], device=device, dtype=torch.float32)
        self.stiffness_var = torch.tensor(hd['stiffness_variation'], device=device, dtype=torch.float32)
        
        # Control parameters
        ctrl = self.params['control_parameters']
        self.K1_gain = ctrl['K1_gain']
        self.K2_gain = ctrl['K2_gain']
        
        # Constraints
        constraints = self.params['constraints']
        self.max_cartesian_vel = constraints['max_cartesian_velocity']
        self.min_z_pos = constraints['min_z_position']
        self.max_robot_force = constraints['max_robot_force']
        self.max_human_force = constraints['max_human_force']
        
        # Joint limits from YAML
        joint_limits = constraints['joint_limits']
        self.joint_lower_limits = torch.tensor([
            joint_limits['waist'][0],
            joint_limits['shoulder'][0], 
            joint_limits['elbow'][0],
            0.0, 2.0944, 0.0  # Fixed end joints
        ], device=device, dtype=torch.float32)
        
        self.joint_upper_limits = torch.tensor([
            joint_limits['waist'][1],
            joint_limits['shoulder'][1],
            joint_limits['elbow'][1], 
            0.0, 2.0944, 0.0  # Fixed end joints
        ], device=device, dtype=torch.float32)
        
        # Safety parameters
        safety = self.params['safety']
        self.safety_margin = safety['safety_margin']
        
        # Initial conditions
        init_joints = self.params['initial_conditions']['joint_positions']
        self.fixed_end_joints = torch.tensor([
            init_joints['yaw'], init_joints['pitch'], init_joints['roll']
        ], device=device, dtype=torch.float32)
        
        # Will be determined in _post_init based on actual robot structure
        self.end_effector_body_id = None
    
    def _init_trajectory(self):
        """Initialize trajectory tracking for all environments"""
        self.traj_direction = (self.end_pos - self.start_pos)
        self.traj_direction = self.traj_direction / torch.norm(self.traj_direction)
        
        # Current trajectory state at time t (per environment)
        self.xd_t = self.start_pos.clone().unsqueeze(0).expand(self.num_envs, -1)
        self.xd_dot_t = torch.zeros(self.num_envs, 3, device=self.device)
        self.tracking_speed = torch.zeros(self.num_envs, device=self.device)
    
    def _init_state_variables(self):
        """Initialize state variables for all environments"""
        # States at time t (for reward computation)
        self.stylus_pos_t = torch.zeros(self.num_envs, 3, device=self.device)
        self.stylus_vel_t = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_forces_t = torch.zeros(self.num_envs, 3, device=self.device)
        self.human_forces_t = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Constraint info at time t
        self.safety_distances_t = torch.zeros(self.num_envs, device=self.device)
        self.is_violating_t = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Human forces at time t+1 (for observation)
        self.human_forces_t1 = torch.zeros(self.num_envs, 3, device=self.device)
    
    def _setup_scene(self):
        """Setup simulation scene"""
        self._omni_robot = Articulation(self.cfg.phantom_omni)
        self.scene.articulations["phantom_omni"] = self._omni_robot
        
        self._constraint = RigidObject(self.cfg.constraint)
        self.scene.rigid_objects["constraint"] = self._constraint
        
        self.scene.clone_environments(copy_from_source=False)
    
    def _setup_post_scene_creation(self):
        """Called after scene creation is complete"""
        super()._setup_post_scene_creation()
        # Now we can safely access body information
        self._initialize_body_indices()
    
    def _initialize_body_indices(self):
        """Initialize body indices after robot is fully loaded"""
        try:
            if not hasattr(self._omni_robot, 'body_names') or not hasattr(self._omni_robot.data, 'body_link_pos_w'):
                return
            
            # Search for stylus body
            stylus_found = False
            for i, name in enumerate(self._omni_robot.body_names):
                if 'stylus' in name.lower() or 'tip' in name.lower() or 'end' in name.lower():
                    self.stylus_body_idx = i
                    stylus_found = True
                    break
            
            # If no stylus found, use the last body as end effector
            num_bodies = len(self._omni_robot.body_names)
            if not stylus_found and num_bodies > 0:
                self.stylus_body_idx = num_bodies - 1
            
            # Validate the body index is within bounds of actual tensor dimensions
            body_tensor_size = self._omni_robot.data.body_link_pos_w.shape[1]
            if self.stylus_body_idx is not None and self.stylus_body_idx >= body_tensor_size:
                self.stylus_body_idx = body_tensor_size - 1 if body_tensor_size > 0 else 0
            
            # Set end effector body id (same as stylus for simplicity)
            self.end_effector_body_id = self.stylus_body_idx
            
        except Exception as e:
            self.stylus_body_idx = 0
            self.end_effector_body_id = 0
    
    def _post_init(self):
        """Fallback method to initialize body indices if not done in setup"""
        if self.stylus_body_idx is None or self.end_effector_body_id is None:
            self._initialize_body_indices()
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply robot control and human forces at time t (parallel)"""
        # DEBUG: Check environment consistency
        if hasattr(self, '_step_counter'):
            self._step_counter += 1
        else:
            self._step_counter = 0
        
        if self._step_counter == 0:
            print(f"[DEBUG] First physics step - verifying {self.num_envs} parallel environments")
            print(f"[DEBUG] Action shape: {actions.shape}")
        
        # Save current states at time t (before physics step) - already in local coordinates
        self.stylus_pos_t = self._get_stylus_position()  # x(t) in local coords relative to base
        self.stylus_vel_t = self._get_stylus_velocity()  # ẋ(t)
        
        # DEBUG: Verify local coordinates are consistent
        if self._step_counter < 3:
            print(f"[DEBUG] Step {self._step_counter} - Local stylus positions (relative to base):")
            for i in range(min(3, self.num_envs)):
                print(f"  Env {i}: {self.stylus_pos_t[i].cpu().numpy()}")
        
        # Store and limit robot forces at time t (vectorized)
        self.robot_forces_t = torch.clamp(actions, -self.max_robot_force, self.max_robot_force)
        
        # Compute human forces at time t (vectorized)
        self._compute_human_forces()
        
        # Analyze constraints at time t (vectorized)
        self._analyze_constraints(self.stylus_pos_t)
        
        # Apply total forces (vectorized)
        self._apply_forces_to_stylus()
        
        # Fix end joints (vectorized)
        self._fix_end_joints()
    
    def _compute_human_forces(self):
        """Compute human forces at time t (fully vectorized)"""
        # Use saved states at time t (already in local coordinates)
        x_t = self.stylus_pos_t
        x_dot_t = self.stylus_vel_t
        
        # Get human equilibrium (Y-axis based switching) - vectorized
        xH_t = torch.where(
            x_t[:, 1:2] > 0.0,  # y > 0
            self.eq_end.unsqueeze(0).expand(self.num_envs, -1),
            self.eq_middle.unsqueeze(0).expand(self.num_envs, -1)
        )
        
        # Dynamic impedance: CHt = diag{[0.14 - 0.133*cos(ẋi)]} - vectorized
        vel_cos = torch.cos(x_dot_t)
        CHt = self.base_damping.unsqueeze(0) - self.damping_var.unsqueeze(0) * vel_cos
        KHt = self.base_stiffness.unsqueeze(0) - self.stiffness_var.unsqueeze(0) * vel_cos
        
        # Human force: f = -(CHt*ẋ + KHt*(x - xH)) - vectorized
        pos_error = x_t - xH_t
        self.human_forces_t = -(CHt * x_dot_t + KHt * pos_error)
        self.human_forces_t = torch.clamp(self.human_forces_t, -self.max_human_force, self.max_human_force)
    
    def _compute_human_forces_at_t1(self):
        """Compute human forces at time t+1 (for observation) - vectorized"""
        # Use current physics state (after step) - in local coordinates
        x_t1 = self._get_stylus_position()
        x_dot_t1 = self._get_stylus_velocity()
        
        # Get human equilibrium (Y-axis based switching) - vectorized
        xH_t1 = torch.where(
            x_t1[:, 1:2] > 0.0,  # y > 0
            self.eq_end.unsqueeze(0).expand(self.num_envs, -1),
            self.eq_middle.unsqueeze(0).expand(self.num_envs, -1)
        )
        
        # Dynamic impedance at t+1 - vectorized
        vel_cos = torch.cos(x_dot_t1)
        CHt1 = self.base_damping.unsqueeze(0) - self.damping_var.unsqueeze(0) * vel_cos
        KHt1 = self.base_stiffness.unsqueeze(0) - self.stiffness_var.unsqueeze(0) * vel_cos
        
        # Human force at t+1 - vectorized
        pos_error = x_t1 - xH_t1
        self.human_forces_t1 = -(CHt1 * x_dot_t1 + KHt1 * pos_error)
        self.human_forces_t1 = torch.clamp(self.human_forces_t1, -self.max_human_force, self.max_human_force)
    
    def _apply_forces_to_stylus(self):
        """Apply total forces (robot + human) to stylus at time t (vectorized)"""
        if self.stylus_body_idx is None or self.end_effector_body_id is None:
            self._post_init()
            if self.stylus_body_idx is None:
                return
        
        body_idx = self.stylus_body_idx if self.stylus_body_idx is not None else self.end_effector_body_id
        
        # Total force: robot control + human force
        total_forces = self.robot_forces_t + self.human_forces_t
        
        try:
            # Get quaternion for the specific body - vectorized
            stylus_quat = self._omni_robot.data.body_link_quat_w[:, body_idx, :]  # [num_envs, 4]
            
            # Transform to local coordinates - vectorized
            forces_local = quat_rotate_inverse(stylus_quat, total_forces)
            
            # Apply forces - reshape for API
            forces = forces_local.unsqueeze(1)  # [num_envs, 1, 3]
            torques = torch.zeros_like(forces)
            
            self._omni_robot.set_external_force_and_torque(forces, torques, body_ids=[body_idx])
        except Exception as e:
            pass
    
    def _fix_end_joints(self):
        """Fix last 3 joints for stylus orientation (vectorized)"""
        joint_pos = self._omni_robot.data.joint_pos.clone()
        joint_vel = self._omni_robot.data.joint_vel.clone()
        
        joint_pos[:, 3:6] = self.fixed_end_joints.unsqueeze(0).expand(self.num_envs, -1)
        joint_vel[:, 3:6] = 0.0
        
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel)
    
    def get_complete_trajectory_state(self, x_current: torch.Tensor) -> dict:
        """Get complete trajectory state at time t (vectorized)"""
        # x_current is already [num_envs, 3] in local coordinates
        
        # Tracking error computation - vectorized
        tracking_error = torch.norm(x_current - self.xd_t, dim=-1)  # [num_envs]
        max_tracking_error = 0.03  # 3cm threshold
        
        # Speed computation based on tracking error - vectorized
        speed = torch.where(
            tracking_error >= max_tracking_error,
            torch.zeros_like(tracking_error),
            torch.where(
                tracking_error <= 1e-6,
                torch.full_like(tracking_error, self.max_cartesian_vel),
                self.max_cartesian_vel * (1.0 - tracking_error / max_tracking_error)
            )
        )
        
        # Trajectory derivatives at time t - vectorized
        xd_dot_t = self.traj_direction.unsqueeze(0) * speed.unsqueeze(-1)
        speed_change = speed - self.tracking_speed
        xd_ddot_t = self.traj_direction.unsqueeze(0) * (speed_change / self.dt).unsqueeze(-1)
        
        # Store for trajectory update
        self.tracking_speed = speed
        self.xd_dot_t = xd_dot_t
        
        return {
            'xd': self.xd_t,  # [num_envs, 3]
            'xd_dot': xd_dot_t,  # [num_envs, 3]
            'xd_ddot': xd_ddot_t,  # [num_envs, 3]
            'tracking_error': tracking_error,  # [num_envs]
            'speed': speed  # [num_envs]
        }
    
    def reset_trajectory(self):
        """Reset trajectory to start for all environments"""
        self.xd_t = self.start_pos.clone().unsqueeze(0).expand(self.num_envs, -1)
        self.xd_dot_t = torch.zeros(self.num_envs, 3, device=self.device)
        self.tracking_speed = torch.zeros(self.num_envs, device=self.device)
    
    def step_trajectory(self):
        """Step trajectory from t to t+1 (vectorized)"""
        # Update trajectory: xd(t+1) = xd(t) + ẋd(t) * dt
        self.xd_t = self.xd_t + self.xd_dot_t * self.dt
        
        # Keep trajectory within bounds - vectorized
        to_current = self.xd_t - self.start_pos.unsqueeze(0)
        projection = torch.sum(to_current * self.traj_direction.unsqueeze(0), dim=-1)
        max_length = torch.norm(self.end_pos - self.start_pos)
        projection = torch.clamp(projection, 0.0, max_length)
        self.xd_t = self.start_pos.unsqueeze(0) + self.traj_direction.unsqueeze(0) * projection.unsqueeze(-1)
    
    def _check_constraints(self) -> torch.Tensor:
        """Check all constraints at time t+1 (vectorized)"""
        violations = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Z position: z > 0 (check at t+1) - in local coordinates
        stylus_pos_t1 = self._get_stylus_position()
        violations |= (stylus_pos_t1[:, 2] <= self.min_z_pos)
        
        # Joint limits (first 3 joints only) - vectorized
        joint_pos_t1 = self.get_joint_positions()
        for i in range(3):
            violations |= (joint_pos_t1[:, i] < self.joint_lower_limits[i]) | (joint_pos_t1[:, i] > self.joint_upper_limits[i])
        
        # Cartesian velocity: |ẋ| ≤ 4cm/s - vectorized
        stylus_vel_t1 = self._get_stylus_velocity()
        violations |= torch.any(torch.abs(stylus_vel_t1) > self.max_cartesian_vel, dim=1)
        
        # Constraint collision - vectorized
        self._analyze_constraints(stylus_pos_t1)
        violations |= self.is_violating_t
        
        return violations
    
    def _analyze_constraints(self, stylus_pos: torch.Tensor):
        """Analyze constraint violations (simplified for parallel envs)"""
        # Simple distance check in local coordinates
        constraint_pos = torch.tensor([0.14, 0.0, 0.0], device=self.device).unsqueeze(0)
        self.safety_distances_t = torch.norm(stylus_pos - constraint_pos, dim=-1)
        self.is_violating_t = self.safety_distances_t < self.safety_margin
    
    def _apply_action(self) -> None:
        """Apply processed actions"""
        self._omni_robot.write_data_to_sim()
    
    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Get observations at time t+1: [x, ẋ, q, q̇, f] (21D) - vectorized"""
        # Physical states at t+1 (after physics step) - in local coordinates
        stylus_pos_t1 = self._get_stylus_position()
        stylus_vel_t1 = self._get_stylus_velocity()
        joint_pos_t1 = self.get_joint_positions()
        joint_vel_t1 = self.get_joint_velocities()
        
        # Apply velocity constraints for observation
        stylus_vel_constrained = torch.clamp(stylus_vel_t1, -self.max_cartesian_vel, self.max_cartesian_vel)
        
        # Compute human forces at t+1 (vectorized)
        self._compute_human_forces_at_t1()
        
        # Ensure all components have correct dimensions
        if joint_pos_t1.shape[-1] != 6:
            if joint_pos_t1.shape[-1] < 6:
                padding = torch.zeros(joint_pos_t1.shape[0], 6 - joint_pos_t1.shape[-1], device=self.device)
                joint_pos_t1 = torch.cat([joint_pos_t1, padding], dim=-1)
            else:
                joint_pos_t1 = joint_pos_t1[..., :6]
        
        if joint_vel_t1.shape[-1] != 6:
            if joint_vel_t1.shape[-1] < 6:
                padding = torch.zeros(joint_vel_t1.shape[0], 6 - joint_vel_t1.shape[-1], device=self.device)
                joint_vel_t1 = torch.cat([joint_vel_t1, padding], dim=-1)
            else:
                joint_vel_t1 = joint_vel_t1[..., :6]
        
        # Construct observation: [x(t+1), ẋ(t+1), q(t+1), q̇(t+1), f(t+1)] = 21D
        obs = torch.cat([
            stylus_pos_t1,           # 3D: x(t+1) in local coords
            stylus_vel_constrained,  # 3D: ẋ(t+1)
            joint_pos_t1,           # 6D: q(t+1)
            joint_vel_t1,           # 6D: q̇(t+1)
            self.human_forces_t1    # 3D: f(t+1)
        ], dim=-1)
        
        return {"policy": torch.clamp(obs, -10.0, 10.0)}
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on time t states (vectorized)"""
        rp = self.params['reward_parameters']
        
        # All reward computation based on saved time t states - in local coordinates
        target = self.end_pos.unsqueeze(0).expand(self.num_envs, -1)
        pos_error = self.stylus_pos_t - target
        tracking_reward = -torch.sum(pos_error**2, dim=-1) * rp['position_tracking_scale']
        
        # Velocity regulation - vectorized
        vel_penalty = -torch.sum(self.stylus_vel_t**2, dim=-1) * rp['velocity_regulation_scale']
        
        # Force regulation - vectorized
        force_penalty = -torch.sum(self.human_forces_t**2, dim=-1) * rp['force_regulation_scale']
        
        # Control penalty - vectorized
        control_penalty = -torch.sum(self.robot_forces_t**2, dim=-1) * rp['control_penalty_scale']
        
        # Safety penalty - vectorized
        safety_penalty = -self.safety_distances_t * rp['safety_penalty_scale']
        
        # Constraint violation penalty - vectorized
        violation_penalty = self.is_violating_t.float() * rp['constraint_violation_penalty']
        
        # Completion reward - vectorized
        distance_to_target = torch.norm(pos_error, dim=-1)
        completion_reward = torch.where(
            distance_to_target < rp['completion_threshold'],
            torch.full_like(distance_to_target, rp['completion_reward']),
            torch.zeros_like(distance_to_target)
        )
        
        total_reward = (tracking_reward + vel_penalty + force_penalty + 
                       control_penalty + safety_penalty + violation_penalty + completion_reward)
        
        # Store logs
        self.extras["log"] = {
            "tracking_reward": tracking_reward.mean().item(),
            "vel_penalty": vel_penalty.mean().item(),
            "force_penalty": force_penalty.mean().item(),
            "control_penalty": control_penalty.mean().item(),
            "safety_penalty": safety_penalty.mean().item(),
            "violation_penalty": violation_penalty.mean().item(),
            "completion_reward": completion_reward.mean().item(),
            "total_reward": total_reward.mean().item(),
            "distance_to_target": distance_to_target.mean().item(),
            "safety_distance": self.safety_distances_t.mean().item(),
            "violation_rate": self.is_violating_t.float().mean().item(),
            "robot_force_norm": torch.norm(self.robot_forces_t, dim=-1).mean().item(),
            "human_force_norm": torch.norm(self.human_forces_t, dim=-1).mean().item(),
        }
        
        return torch.clamp(total_reward, rp.get('reward_min', -100.0), rp.get('reward_max', 75.0))
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions (vectorized)"""
        # Check all constraints (at t+1) - vectorized
        constraint_violations = self._check_constraints()
        
        # Check target reached (at t+1) - vectorized
        stylus_pos = self._get_stylus_position()
        target_distance = torch.norm(stylus_pos - self.end_pos.unsqueeze(0).expand(self.num_envs, -1), dim=-1)
        target_reached = target_distance < self.params['reward_parameters']['completion_threshold']
        
        terminated = constraint_violations | target_reached
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        
        return terminated, truncated
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments (vectorized)"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # DEBUG: Print reset information
        print(f"[DEBUG] Resetting environments: {env_ids.cpu().numpy()}")
        
        super()._reset_idx(env_ids)
        
        # Ensure body indices are set after reset
        if self.stylus_body_idx is None or self.end_effector_body_id is None:
            self._post_init()
        
        num_resets = len(env_ids)
        
        # Set initial joint positions from YAML
        init_joints = self.params['initial_conditions']['joint_positions']
        joint_pos = torch.zeros((num_resets, 6), device=self.device)
        joint_pos[:, 0] = init_joints['waist']
        joint_pos[:, 1] = init_joints['shoulder']
        joint_pos[:, 2] = init_joints['elbow']
        joint_pos[:, 3] = init_joints['yaw']
        joint_pos[:, 4] = init_joints['pitch']
        joint_pos[:, 5] = init_joints['roll']
        
        # Add noise
        noise = sample_uniform(-0.05, 0.05, (num_resets, 6), self.device)
        joint_pos += noise
        
        joint_vel = torch.zeros((num_resets, 6), device=self.device)
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # Reset all states for specified environments
        self.stylus_pos_t[env_ids] = 0.0
        self.stylus_vel_t[env_ids] = 0.0
        self.robot_forces_t[env_ids] = 0.0
        self.human_forces_t[env_ids] = 0.0
        self.human_forces_t1[env_ids] = 0.0
        self.safety_distances_t[env_ids] = 0.0
        self.is_violating_t[env_ids] = False
        
        # Reset trajectory for specified environments
        self.xd_t[env_ids] = self.start_pos.clone()
        self.xd_dot_t[env_ids] = 0.0
        self.tracking_speed[env_ids] = 0.0
        
        # DEBUG: Verify reset positions
        reset_pos = self._get_stylus_position()
        print(f"[DEBUG] After reset - Local stylus positions:")
        for i in env_ids[:min(3, len(env_ids))]:
            print(f"  Env {i}: {reset_pos[i].cpu().numpy()}")
    
    # Getter methods
    def _get_stylus_position(self):
        """Get stylus position relative to robot base frame"""
        if self.end_effector_body_id is None:
            self._post_init()
            if self.end_effector_body_id is None:
                return torch.zeros(self.num_envs, 3, device=self.device)
        
        try:
            # Get robot base position (root link)
            base_pos = self._omni_robot.data.root_link_pos_w  # [num_envs, 3]
            
            # Get end effector position in world frame
            ee_pos_world = self._omni_robot.data.body_link_pos_w[:, self.end_effector_body_id, :]
            
            # Calculate relative position (end effector relative to base)
            position_local = ee_pos_world - base_pos
            
            # DEBUG: Print position (only first few steps)
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
            
            if self._debug_counter < 5:
                print(f"[DEBUG] Stylus position relative to base (step {self._debug_counter}):")
                for i in range(min(2, self.num_envs)):
                    print(f"  Env {i}: Base {base_pos[i].cpu().numpy()}, EE {ee_pos_world[i].cpu().numpy()}, Relative {position_local[i].cpu().numpy()}")
            
            return position_local
        except Exception as e:
            print(f"[ERROR] Failed to get stylus position: {e}")
            return torch.zeros(self.num_envs, 3, device=self.device)
    
    def _get_stylus_velocity(self):
        """Get stylus velocity in robot frame"""
        if self.end_effector_body_id is None:
            self._post_init()
            if self.end_effector_body_id is None:
                return torch.zeros(self.num_envs, 3, device=self.device)
        
        try:
            # Use body_link_lin_vel_w property which gives velocities
            velocity = self._omni_robot.data.body_link_lin_vel_w[:, self.end_effector_body_id, :]
            return velocity
        except Exception as e:
            return torch.zeros(self.num_envs, 3, device=self.device)
    
    def get_joint_positions(self):
        """Get joint positions (vectorized)"""
        return self._omni_robot.data.joint_pos
    
    def get_joint_velocities(self):
        """Get joint velocities (vectorized)"""
        return self._omni_robot.data.joint_vel
    
    def get_training_params(self):
        """Return loaded training parameters"""
        return self.params
    
    @property
    def unwrapped(self):
        return self