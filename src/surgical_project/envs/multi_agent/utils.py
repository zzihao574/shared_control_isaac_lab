"""
Environment utilities for shared network MADDPG training.
Essential environment components for physics-based constraint analysis and console debugging.

Features:
- CompleteConstraintChecker: Physics-based constraint analysis
- TrajectoryManager: Trajectory management for path following
- StepTracer: Console debugging with four-zone reward system monitoring
- MODIFIED: Added eval mode support to disable printing during evaluation
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# Force import dependencies - fail fast if not installed
from omni.physx.bindings._physx import (
    acquire_physx_attachment_interface, acquire_physx_scene_query_interface
)
from carb._carb import Float3


class CompleteConstraintChecker:
    """
    Physics-based constraint analysis for surgical robot environment.
    Two-state detection: Outside / Overlapping based on raycast validation.
    
    Features:
    - Batch constraint state analysis
    - Raycast-based overlapping detection
    - Consistent distance reporting (raycast distance for outside, 0.0 for overlapping)
    - Normal vector calculation pointing from stylus toward obstacle
    """
    
    def __init__(self, device: torch.device, collision_threshold: float = 0.001):
        self.device = device
        self.collision_threshold = collision_threshold        
        try:
            self.physics_attachment_interface = acquire_physx_attachment_interface()
            self.physics_scene_query_interface = acquire_physx_scene_query_interface()
        except ImportError:
            self.physics_attachment_interface = None
            self.physics_scene_query_interface = None
    
    def analyze_constraint_state_batch(self, stylus_positions: torch.Tensor, env_base_positions: torch.Tensor):
        """Analyze constraint states for batch of environments."""
        num_envs = stylus_positions.shape[0]
        
        current_base_positions = self._omni_robot.data.root_link_pos_w if hasattr(self, '_omni_robot') else env_base_positions
        
        batch_results = {
            'distances_constraint': torch.ones(num_envs, device=self.device) * 0.02,
            'closest_points': torch.zeros(num_envs, 3, device=self.device),
            'normal_vectors': torch.ones(num_envs, 3, device=self.device),
            'is_overlapping': torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        }
        
        if self.physics_attachment_interface is None or self.physics_scene_query_interface is None:
            return batch_results
        
        for env_id in range(num_envs):
            stylus_world_pos = stylus_positions[env_id] + current_base_positions[env_id]
            constraint_path = f"/World/envs/env_{env_id}/Constraint/Sphere"
            
            try:
                result = self._analyze_single_constraint(stylus_world_pos, constraint_path)
                if result is not None:
                    batch_results['distances_constraint'][env_id] = result['distance']
                    batch_results['closest_points'][env_id] = torch.tensor(result['closest_point'], device=self.device)
                    batch_results['normal_vectors'][env_id] = torch.tensor(result['normal_vector'], device=self.device)
                    batch_results['is_overlapping'][env_id] = result['is_overlapping']
            except Exception:
                pass
        
        return batch_results
    
    def _analyze_single_constraint(self, stylus_position: torch.Tensor, constraint_path: str, verbose: bool = False):
        """
        Two-state constraint analysis using raycast validation.
        
        Logic:
        - If raycast hits constraint with valid distance: outside (returns raycast distance)
        - If raycast doesn't hit or distance is 0: overlapping (returns 0.0)
        
        Returns result with consistent distance reporting.
        """
        if self.physics_attachment_interface is None or self.physics_scene_query_interface is None:
            return None

        from carb._carb import Float3
        import numpy as np

        # 1) Get stylus position and closest point on constraint
        pos = stylus_position.detach().cpu().numpy().astype(np.float64)
        current_point = Float3(float(pos[0]), float(pos[1]), float(pos[2]))

        result = self.physics_attachment_interface.get_closest_points([current_point], constraint_path)
        if not (result and "closest_points" in result and result["closest_points"]):
            return None

        cp = result["closest_points"][0]
        closest = np.array([cp.x, cp.y, cp.z], dtype=np.float64)
        to_closest = closest - pos
        dist = float(np.linalg.norm(to_closest))

        # Handle degenerate case
        if dist < 1e-9:
            return {
                "distance": 0.0,
                "closest_point": closest,
                "normal_vector": np.array([1.0, 0.0, 0.0], dtype=np.float64),
                "state": "overlapping",
                "is_overlapping": True,
                "raycast_result": None,
            }

        # 2) Raycast from stylus toward closest point
        dir_norm = to_closest / dist
        ray_dir = Float3(float(dir_norm[0]), float(dir_norm[1]), float(dir_norm[2]))
        ray_res = self.physics_scene_query_interface.raycast_closest(current_point, ray_dir, dist + 0.01)

        # Filter for hits against target constraint only
        hit = None
        if ray_res and "collision" in ray_res and (constraint_path in str(ray_res["collision"])):
            hit = ray_res

        # 3) Two-state decision with collision threshold
        hit_distance = hit.get("distance", 0.0) if hit else 0.0
        
        if hit is not None and hit_distance > self.collision_threshold:
            is_overlapping = False
            state = "outside"
            # For outside state, use the raycast hit distance (surface distance)
            final_distance = hit_distance
        else:
            is_overlapping = True 
            state = "overlapping"
            # For overlapping state, always return 0.0 to indicate penetration
            final_distance = 0.0

        # 4) Normal vector calculation (pointing from stylus toward obstacle)
        if hit is not None and "normal" in hit:
            n = hit["normal"]
            outward = np.array([n.x, n.y, n.z], dtype=np.float64)
            normal_vec = -outward  # negate: stylus -> obstacle interior
        else:
            # No hit: use stylus -> closest as direction
            normal_vec = dir_norm  # already stylus -> obstacle direction

        return {
            "distance": final_distance,
            "closest_point": closest,
            "normal_vector": normal_vec,
            "state": state,
            "is_overlapping": is_overlapping,
            "raycast_result": ray_res,
        }


class TrajectoryManager:
    """
    Trajectory management for surgical robot path following.
    
    Features:
    - Progress calculation along trajectory
    - Deviation measurement from ideal path
    - Task completion detection
    """
    
    def __init__(self, device: torch.device, params: dict, num_envs: int, env_base_positions: torch.Tensor):
        self.device = device
        self.num_envs = num_envs
        self.env_base_positions = env_base_positions
        
        traj = params['trajectory']
        self.start_pos_local = torch.tensor(traj['start_point'], device=device, dtype=torch.float32)
        self.end_pos_local = torch.tensor(traj['end_point'], device=device, dtype=torch.float32)
        self.total_distance = torch.norm(self.end_pos_local - self.start_pos_local).item()
        
        self.line_direction = (self.end_pos_local - self.start_pos_local) / self.total_distance
        
        # Unified completion threshold
        reward_params = params.get('reward_parameters', {})
        self.completion_threshold = reward_params.get('completion_threshold', 0.01)
        
        print(f"[INFO] Trajectory completion threshold: {self.completion_threshold}m")
    
    def get_progress(self, current_pos_local: torch.Tensor) -> torch.Tensor:
        """Calculate progress along trajectory (0 to 1)."""
        vec_to_current = current_pos_local - self.start_pos_local.unsqueeze(0)
        progress_distance = torch.sum(vec_to_current * self.line_direction.unsqueeze(0), dim=-1)
        progress_distance = torch.clamp(progress_distance, 0, self.total_distance)
        return progress_distance / self.total_distance
    
    def get_deviation(self, current_pos_local: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate perpendicular distance to trajectory."""
        vec_to_current = current_pos_local - self.start_pos_local.unsqueeze(0)
        progress_distance = torch.sum(vec_to_current * self.line_direction.unsqueeze(0), dim=-1)
        progress_distance = torch.clamp(progress_distance, 0, self.total_distance)
        
        closest_points = self.start_pos_local.unsqueeze(0) + progress_distance.unsqueeze(-1) * self.line_direction.unsqueeze(0)
        deviations = torch.norm(current_pos_local - closest_points, dim=-1)
        
        return deviations, closest_points
    
    def is_final_setpoint_reached(self, current_pos_local: torch.Tensor) -> torch.Tensor:
        """Check if final setpoint is reached."""
        distances_to_final = torch.norm(current_pos_local - self.end_pos_local.unsqueeze(0), dim=-1)
        return distances_to_final < self.completion_threshold


class StepTracer:
    """
    Console debugging with four-zone reward system monitoring.
    Simplified version - removed unnecessary debug checks, let it crash if there are issues.
    """
    
    def __init__(self, num_envs: int, device: torch.device,
                 enable_console_logging: bool = False,
                 print_every_steps: int = 10,
                 max_envs_to_print: int = 2):
        self.num_envs = num_envs
        self.device = device
        self.enable_console_logging = enable_console_logging
        self.print_every_steps = print_every_steps
        self.max_envs_to_print = max_envs_to_print

    def maybe_print_step(self, env, rewards: Dict, global_step: int, force_print: bool = False):
        """Console printing with force_print bypass for evaluation."""
        if not self.enable_console_logging:
            return
            
        # Skip step frequency check if force_print is True (evaluation mode)
        if not force_print and global_step % self.print_every_steps != 0:
            return

        to_show = list(range(min(self.max_envs_to_print, self.num_envs)))

        print("=" * 80)
        print(f"STEP {global_step} - Four-Zone Reward System (A/B/C/D) - Shared Network")
        print("=" * 80)
        print(f"Showing first {len(to_show)} of {self.num_envs} environments")

        for env_id in to_show:
            self._print_env_snapshot(env, env_id)

        print("\n[REWARDS FOR THIS STEP]")
        for env_id in to_show:
            robot_total = rewards["robot"][env_id].item()
            human_total = rewards["human"][env_id].item()
            combined_total = robot_total + human_total
            print(f"  Env {env_id}: Robot: {robot_total:+.3f} | Human: {human_total:+.3f} | Combined: {combined_total:+.3f}")

    def _print_env_snapshot(self, env, env_id: int):
        """Print environment snapshot."""
        stylus = env.stylus_pos_t1[env_id]
        safety = float(env.safety_distances_t1[env_id].item())
        normal = env.normal_t1[env_id]
        
        # Safe access to reward components
        dev = self._safe_get_reward_component(env, 'deviation', env_id, 0.0)
        prog = self._safe_get_reward_component(env, 'progress_ratio', env_id, 0.0)
        dist = self._safe_get_reward_component(env, 'distance_to_final', env_id, 0.0)

        print(f"\n--- Environment {env_id} ---")
        print(f"Stylus: [{stylus[0]:.4f}, {stylus[1]:.4f}, {stylus[2]:.4f}] | Safety: {safety:.4f}m")
        print(f"Normal: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")

        self._print_actor_forces(env, env_id)

        print(f"Deviation: {dev:.4f}m | Progress: {prog:.1%} | Distance to End: {dist:.4f}m")

        # Determine active zone
        D, O = 0.0075, 0.015
        is_colliding = env.is_violating_t1[env_id].item()

        if safety >= O:
            active_zone = "A (Track)"
        elif safety <= D:
            active_zone = "C (Danger - Overlapping)" if is_colliding else "C (Danger - Outside)"
        else:
            active_zone = "D (Rejoin)" if env.rejoin_streak[env_id] >= 10 else "B (Surface)"
        print(f"Active Zone: {active_zone}")

        self._print_agent_rewards(env, env_id, "ROBOT")
        self._print_agent_rewards(env, env_id, "HUMAN")

    def _print_agent_rewards(self, env, env_id: int, agent_label: str):
        """Print zone rewards for agent - simplified."""
        agent_key = agent_label.lower()
        RC = env.reward_components
        is_colliding = env.is_violating_t1[env_id].item()
        
        print(f"\n[{agent_label}] ZONE REWARDS:")

        def _val(x):
            if torch.is_tensor(x):
                return float(x[env_id].item()) if x.ndim > 0 else float(x.item())
            return float(x)

        for Z in ['A', 'B', 'C', 'D']:
            zone_w = _val(RC[f'zone{Z}_weight_{agent_key}'])
            zone_total = _val(RC[f'zone{Z}_total_{agent_key}'])
            
            if Z == 'C':
                mode = "Overlapping" if is_colliding else "Outside"
                print(f"  C Danger-{mode} (zone_w={zone_w:.2f}): {zone_total:+.3f}")
                
                if is_colliding:
                    raw = _val(RC[f'zoneC_overlap_{agent_key}_raw'])
                    weight = _val(RC[f'zoneC_overlap_{agent_key}_weight'])
                    contrib = _val(RC[f'zoneC_overlap_{agent_key}_contrib'])
                    print(f"    {'Overlap_Pen':<12} {raw:.3f} * {weight:.2f} = {contrib:+.3f}")
                else:
                    for comp_key, comp_label in [('offpen', 'Off_Penalty'), ('inward', 'Inward_Pen')]:
                        raw = _val(RC[f'zoneC_{comp_key}_{agent_key}_raw'])
                        weight = _val(RC[f'zoneC_{comp_key}_{agent_key}_weight'])
                        contrib = _val(RC[f'zoneC_{comp_key}_{agent_key}_contrib'])
                        print(f"    {comp_label:<12} {raw:.3f} * {weight:.2f} = {contrib:+.3f}")
                continue
            
            zone_names = {'A': 'A Track   ', 'B': 'B Surface ', 'D': 'D Rejoin  '}
            components = {
                'A': [('progress', 'Progress'), ('deviation', 'Deviation')],
                'B': [('gap', 'Gap'), ('surftangent', 'Surf_Tangent'), ('inward', 'Inward_Pen')],
                'D': [('progress', 'Progress'), ('deviation', 'Deviation'), ('inward', 'Inward_Pen')],
            }
            
            print(f"  {zone_names[Z]} (zone_w={zone_w:.2f}): {zone_total:+.3f}")
            
            for comp_key, comp_label in components[Z]:
                raw = _val(RC[f'zone{Z}_{comp_key}_{agent_key}_raw'])
                weight = _val(RC[f'zone{Z}_{comp_key}_{agent_key}_weight'])
                contrib = _val(RC[f'zone{Z}_{comp_key}_{agent_key}_contrib'])
                print(f"    {comp_label:<12} {raw:.3f} * {weight:.2f} = {contrib:+.3f}")

        # Global rewards
        print("  Global Rewards:")
        globals_list = [
            ('zpenalty', 'Z Penalty      '),
            ('completion', 'Completion     '),
            ('timeeff', 'Time Efficiency')
        ]
        
        for gk, glabel in globals_list:
            raw = _val(RC[f'global_{gk}_{agent_key}_raw'])
            weight = _val(RC[f'global_{gk}_{agent_key}_weight'])
            contrib = _val(RC[f'global_{gk}_{agent_key}_contrib'])
            print(f"    {glabel:<15} {raw:.3f} * {weight:.2f} = {contrib:+.3f}")

        # Agent force
        raw = _val(RC[f'{agent_key}force_raw'])
        weight = _val(RC[f'{agent_key}force_weight'])
        contrib = _val(RC[f'{agent_key}force_contrib'])
        print(f"    {agent_label} Force   {raw:.3f} * {weight:.2f} = {contrib:+.3f}")

        # Cross-agent awareness
        aware_key = 'humanaware' if agent_key == 'robot' else 'robotaware'
        raw = _val(RC[f'{aware_key}_raw'])
        weight = _val(RC[f'{aware_key}_weight'])
        contrib = _val(RC[f'{aware_key}_contrib'])
        label = "Human Aware" if agent_key == 'robot' else "Robot Aware"
        print(f"    {label:<15} {raw:.3f} * {weight:.2f} = {contrib:+.3f}")

    def _print_actor_forces(self, env, env_id: int):
        """Print actor forces - simplified."""
        robot_mean = env.actor_mean_forces['robot'][env_id]
        robot_noise = env.actor_noise_forces['robot'][env_id]
        human_mean = env.actor_mean_forces['human'][env_id]
        human_noise = env.actor_noise_forces['human'][env_id]
        
        print(f"Physical Forces (Robot): Fx={robot_mean[0]:+.6f}N, Fy={robot_mean[1]:+.6f}N, Fz={robot_mean[2]:+.6f}N")
        print(f"  Noise: Nx={robot_noise[0]:+.6f}N, Ny={robot_noise[1]:+.6f}N, Nz={robot_noise[2]:+.6f}N")
        print(f"Physical Forces (Human): Fx={human_mean[0]:+.6f}N, Fy={human_mean[1]:+.6f}N, Fz={human_mean[2]:+.6f}N")
        print(f"  Noise: Nx={human_noise[0]:+.6f}N, Ny={human_noise[1]:+.6f}N, Nz={human_noise[2]:+.6f}N")

    def _safe_get_reward_component(self, env, key: str, env_id: int, default: float) -> float:
        """Safely get reward component value with explicit checks."""
        if not hasattr(env, 'reward_components') or key not in env.reward_components:
            return default
        
        component = env.reward_components[key]
        if not torch.is_tensor(component):
            return float(component) if isinstance(component, (int, float)) else default
        
        if component.shape[0] <= env_id:
            return default
            
        return float(component[env_id].item())