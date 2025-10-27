# SPDX-License-Identifier: BSD-3-Clause
"""
Environment utilities for Epigraph MARL training.
Copied from multi_agent/utils.py with Epigraph helpers appended at the end.
"""

# ============================================================================
# ORIGINAL UTILS (完整复制 surgical_direct_marl_env 的 utils.py)
# ============================================================================

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from omni.physx.bindings._physx import (
    acquire_physx_attachment_interface, acquire_physx_scene_query_interface
)
from carb._carb import Float3


class CompleteConstraintChecker:
    """Physics-based constraint analysis for surgical robot environment."""
    
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
        
        if hasattr(self, '_omni_robot') and self._omni_robot is not None:
            base_quats = self._omni_robot.data.root_link_quat_w.float()
        else:
            base_quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device, dtype=torch.float32).repeat(num_envs, 1)
        
        for env_id in range(num_envs):
            stylus_world_pos = stylus_positions[env_id] + current_base_positions[env_id]
            constraint_path = f"/World/envs/env_{env_id}/Constraint/Torus"
            
            try:
                result = self._analyze_single_constraint(stylus_world_pos, constraint_path)
                if result is not None:
                    batch_results['distances_constraint'][env_id] = result['distance']
                    batch_results['is_overlapping'][env_id] = result['is_overlapping']
                    
                    closest_world = torch.tensor(result['closest_point'], device=self.device, dtype=torch.float32)
                    closest_local = closest_world - current_base_positions[env_id]
                    batch_results['closest_points'][env_id] = closest_local
                    
                    try:
                        from isaaclab.utils.math import quat_rotate_inverse
                        normal_world = torch.tensor(result['normal_vector'], device=self.device, dtype=torch.float32)
                        normal_local = quat_rotate_inverse(base_quats[env_id:env_id+1], normal_world.unsqueeze(0))
                        batch_results['normal_vectors'][env_id] = normal_local.squeeze(0)
                    except Exception:
                        batch_results['normal_vectors'][env_id] = torch.tensor(result['normal_vector'], device=self.device, dtype=torch.float32)
            except Exception:
                pass
        
        return batch_results
    
    def _analyze_single_constraint(self, stylus_position: torch.Tensor, constraint_path: str, verbose: bool = False):
        """Two-state constraint analysis using raycast validation."""
        if self.physics_attachment_interface is None or self.physics_scene_query_interface is None:
            return None

        pos = stylus_position.detach().cpu().numpy().astype(np.float64)
        current_point = Float3(float(pos[0]), float(pos[1]), float(pos[2]))

        result = self.physics_attachment_interface.get_closest_points([current_point], constraint_path)
        if not (result and "closest_points" in result and result["closest_points"]):
            return None

        cp = result["closest_points"][0]
        closest = np.array([cp.x, cp.y, cp.z], dtype=np.float64)
        to_closest = closest - pos
        dist = float(np.linalg.norm(to_closest))

        if dist < 1e-9:
            return {
                "distance": 0.0,
                "closest_point": closest,
                "normal_vector": np.array([1.0, 0.0, 0.0], dtype=np.float64),
                "state": "overlapping",
                "is_overlapping": True,
                "raycast_result": None,
            }

        dir_norm = to_closest / dist
        ray_dir = Float3(float(dir_norm[0]), float(dir_norm[1]), float(dir_norm[2]))
        ray_res = self.physics_scene_query_interface.raycast_closest(current_point, ray_dir, dist + 0.01)

        hit = None
        if ray_res and "collision" in ray_res and (constraint_path in str(ray_res["collision"])):
            hit = ray_res

        hit_distance = hit.get("distance", 0.0) if hit else 0.0
        
        if hit is not None and hit_distance > self.collision_threshold:
            is_overlapping = False
            state = "outside"
            final_distance = hit_distance
        else:
            is_overlapping = True 
            state = "overlapping"
            final_distance = 0.0

        if hit is not None and "normal" in hit:
            n = hit["normal"]
            outward = np.array([n.x, n.y, n.z], dtype=np.float64)
            normal_vec = -outward
        else:
            normal_vec = dir_norm

        return {
            "distance": final_distance,
            "closest_point": closest,
            "normal_vector": normal_vec,
            "state": state,
            "is_overlapping": is_overlapping,
            "raycast_result": ray_res,
        }


class TrajectoryManager:
    """Trajectory management for surgical robot path following."""
    
    def __init__(self, device: torch.device, params: dict, num_envs: int, env_base_positions: torch.Tensor):
        self.device = device
        self.num_envs = num_envs
        self.env_base_positions = env_base_positions
        
        traj = params['trajectory']
        self.start_pos_local = torch.tensor(traj['start_point'], device=device, dtype=torch.float32)
        self.end_pos_local = torch.tensor(traj['end_point'], device=device, dtype=torch.float32)
        self.total_distance = torch.norm(self.end_pos_local - self.start_pos_local).item()
        
        self.line_direction = (self.end_pos_local - self.start_pos_local) / self.total_distance
        
        reward_params = params.get('reward_parameters', {})
        self.completion_threshold = reward_params.get('completion_threshold', 0.01)
        
        print(f"[INFO] Trajectory completion threshold: {self.completion_threshold}m")
    
    def get_progress(self, current_pos_local: torch.Tensor) -> torch.Tensor:
        """Calculate progress along trajectory (0 to 1)."""
        vec_to_current = current_pos_local - self.start_pos_local.unsqueeze(0)
        progress_distance = torch.sum(vec_to_current * self.line_direction.unsqueeze(0), dim=-1)
        return progress_distance / self.total_distance
    
    def get_deviation(self, current_pos_local: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate perpendicular distance to trajectory line."""
        vec_to_current = current_pos_local - self.start_pos_local.unsqueeze(0)
        progress_distance = torch.sum(vec_to_current * self.line_direction.unsqueeze(0), dim=-1)
        progress_distance = torch.clamp(progress_distance, 0, self.total_distance)
        
        closest_points = self.start_pos_local.unsqueeze(0) + progress_distance.unsqueeze(-1) * self.line_direction.unsqueeze(0)
        deviations = torch.norm(current_pos_local - closest_points, dim=-1)
        
        return deviations, closest_points
    
    def is_final_setpoint_reached(self, current_pos_local: torch.Tensor) -> torch.Tensor:
        """Check if final setpoint is reached within completion threshold."""
        distances_to_final = torch.norm(current_pos_local - self.end_pos_local.unsqueeze(0), dim=-1)
        return distances_to_final < self.completion_threshold


class StepTracer:
    """Console debugging with four-zone reward system monitoring."""
    
    def __init__(self, num_envs: int, device: torch.device,
                 enable_console_logging: bool = False,
                 print_every_steps: int = 10,
                 max_envs_to_print: int = 2):
        self.num_envs = num_envs
        self.device = device
        self.enable_console_logging = enable_console_logging
        self.print_every_steps = print_every_steps
        self.max_envs_to_print = max_envs_to_print

    def maybe_print_step(
        self,
        env,
        r_task: torch.Tensor,        # [num_envs, num_agents]
        r_safe_cost: torch.Tensor,   # [num_envs, num_agents], >=0
        z: torch.Tensor,             # [num_envs, num_agents]
        global_step: int,
        force_print: bool = False,
    ):
        """
        Print step information with Epigraph decomposition (task/safe/z).
        
        Args:
            env: Environment instance
            r_task: Task rewards [num_envs, num_agents]
            r_safe_cost: Safety costs [num_envs, num_agents], non-negative
            z: Risk budget values [num_envs, num_agents]
            global_step: Current training step
            force_print: Force printing regardless of print_every_steps
        """
        if not self.enable_console_logging:
            return
            
        if not force_print and global_step % self.print_every_steps != 0:
            return

        to_show = list(range(min(self.max_envs_to_print, self.num_envs)))

        print("=" * 80)
        print(f"[STEP {global_step}] Four-Zone Safety Monitor (Epigraph)")
        print("=" * 80)
        print(f"Showing first {len(to_show)} of {self.num_envs} environments")

        # 1. Environment state snapshot: trajectory progress, normals, zone classification
        for env_id in to_show:
            self._print_env_snapshot(env, env_id)

        # 2. Reward decomposition display
        print("\n[REWARD BREAKDOWN THIS STEP]")
        for env_id in to_show:
            rt_robot  = r_task[env_id, 0].item()
            rt_human  = r_task[env_id, 1].item()
            rc_robot  = r_safe_cost[env_id, 0].item()
            rc_human  = r_safe_cost[env_id, 1].item()
            print(f"  Env {env_id}:")
            print(f"    task_reward        robot {rt_robot:+.3f} | human {rt_human:+.3f}")
            print(f"    safe_cost (>=0)    robot {rc_robot:+.3f} | human {rc_human:+.3f}")

        # 3. Risk budget z
        print("\n[Z (RISK BUDGET) THIS STEP]")
        for env_id in to_show:
            z_robot = z[env_id, 0].item()
            z_human = z[env_id, 1].item()
            z_team  = 0.5 * (z_robot + z_human)  # Team average for reference
            print(f"  Env {env_id}: z_robot={z_robot:.4f} | z_human={z_human:.4f} | z_team(ref)={z_team:.4f}")

        # 4. Agent-level reward decomposition (original detailed breakdown)
        for env_id in to_show:
            self._print_agent_rewards(env, env_id, "ROBOT")
            self._print_agent_rewards(env, env_id, "HUMAN")


    def _print_env_snapshot(self, env, env_id: int):
        """Print detailed environment state snapshot."""
        stylus = env.stylus_pos_t1[env_id]
        safety = float(env.safety_distances_t1[env_id].item())
        normal = env.normal_t1[env_id]
        
        dev = self._safe_get_reward_component(env, 'deviation', env_id, 0.0)
        prog = self._safe_get_reward_component(env, 'progress_ratio', env_id, 0.0)
        dist = self._safe_get_reward_component(env, 'distance_to_final', env_id, 0.0)

        print(f"\n--- Environment {env_id} ---")
        print(f"Stylus: [{stylus[0]:.4f}, {stylus[1]:.4f}, {stylus[2]:.4f}] | Safety: {safety:.4f}m")
        print(f"Normal: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")

        self._print_actor_forces(env, env_id)

        print(f"Deviation: {dev:.4f}m | Progress: {prog:.1%} | Distance to End: {dist:.4f}m")

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
        """Print detailed reward breakdown for specified agent."""
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

        print("  Global Rewards:")
        globals_list = [
            ('zpenalty', 'Z Penalty      '),
            ('completion', 'Completion     '),
            ('timeeff', 'Time Efficiency'),
            ('potential', 'Potential      '),
        ]
        
        for gk, glabel in globals_list:
            raw = _val(RC[f'global_{gk}_{agent_key}_raw'])
            weight = _val(RC[f'global_{gk}_{agent_key}_weight'])
            contrib = _val(RC[f'global_{gk}_{agent_key}_contrib'])
            print(f"    {glabel:<15} {raw:.3f} * {weight:.2f} = {contrib:+.3f}")

        raw = _val(RC[f'{agent_key}force_raw'])
        weight = _val(RC[f'{agent_key}force_weight'])
        contrib = _val(RC[f'{agent_key}force_contrib'])
        print(f"    {agent_label} Force   {raw:.3f} * {weight:.2f} = {contrib:+.3f}")

        aware_key = 'humanaware' if agent_key == 'robot' else 'robotaware'
        raw = _val(RC[f'{aware_key}_raw'])
        weight = _val(RC[f'{aware_key}_weight'])
        contrib = _val(RC[f'{aware_key}_contrib'])
        label = "Human Aware" if agent_key == 'robot' else "Robot Aware"
        print(f"    {label:<15} {raw:.3f} * {weight:.2f} = {contrib:+.3f}")

    def _print_actor_forces(self, env, env_id: int):
        """Print current actor force outputs for both agents."""
        robot_mean = env.actor_mean_forces['robot'][env_id]
        robot_noise = env.actor_noise_forces['robot'][env_id]
        human_mean = env.actor_mean_forces['human'][env_id]
        human_noise = env.actor_noise_forces['human'][env_id]
        
        print(f"Physical Forces (Robot): Fx={robot_mean[0]:+.6f}N, Fy={robot_mean[1]:+.6f}N, Fz={robot_mean[2]:+.6f}N")
        print(f"  Noise: Nx={robot_noise[0]:+.6f}N, Ny={robot_noise[1]:+.6f}N, Nz={robot_noise[2]:+.6f}N")
        print(f"Physical Forces (Human): Fx={human_mean[0]:+.6f}N, Fy={human_mean[1]:+.6f}N, Fz={human_mean[2]:+.6f}N")
        print(f"  Noise: Nx={human_noise[0]:+.6f}N, Ny={human_noise[1]:+.6f}N, Nz={human_noise[2]:+.6f}N")

    def _safe_get_reward_component(self, env, key: str, env_id: int, default: float) -> float:
        """Safely extract reward component value with fallback handling."""
        if not hasattr(env, 'reward_components') or key not in env.reward_components:
            return default
        
        component = env.reward_components[key]
        if not torch.is_tensor(component):
            return float(component) if isinstance(component, (int, float)) else default
        
        if component.shape[0] <= env_id:
            return default
            
        return float(component[env_id].item())


# ============================================================================
# EPIGRAPH HELPERS (追加在文件末尾)
# ============================================================================

def _zeros(device, n):
    """Create zero tensor helper."""
    return torch.zeros(n, device=device)


def safe_get_rc(rc: Dict[str, torch.Tensor], key: str, device, num_envs: int) -> torch.Tensor:
    """
    Safely read reward_components key; return zeros if missing.
    
    Args:
        rc: reward_components dictionary
        key: component key to retrieve
        device: torch device
        num_envs: number of environments
        
    Returns:
        Component tensor or zeros
    """
    if rc is None:
        return _zeros(device, num_envs)
    return rc.get(key, _zeros(device, num_envs))


def compose_task_safe_from_rc(
    rc: Dict[str, torch.Tensor],
    agent: str,
    device,
    num_envs: int,
    use_time_eff_in_task: bool = True,
    include_zpenalty_in_safe: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose reward_components into task reward and safety risk for Epigraph training.
    
    Task includes:
        - Zone A/B/D (excluding inward penalties)
        - Global: potential, completion, (optional) time_eff
        - Force penalties (already negative)
        - Awareness: humanaware (robot) / robotaware (human)
        
    Safe includes:
        - Zone C (danger zone)
        - B/D inward penalties
        - (optional) Z penalty
    
    Args:
        rc: reward_components from environment
        agent: "human" or "robot"
        device: torch device
        num_envs: number of environments
        use_time_eff_in_task: whether to include time efficiency in task reward
        include_zpenalty_in_safe: whether to include z penalty in safety cost
        
    Returns:
        Tuple of (r_task, r_safe_risk) tensors
    """
    z = lambda k: safe_get_rc(rc, k, device, num_envs)

    aware_key = "humanaware" if agent == "robot" else "robotaware"

    # Task quality reward (for training Vl)
    r_task = (
        z(f"zoneA_total_{agent}")
        + z(f"zoneB_gap_{agent}_contrib")
        + z(f"zoneB_surftangent_{agent}_contrib")
        + z(f"zoneD_progress_{agent}_contrib")
        + z(f"zoneD_deviation_{agent}_contrib")
        + z(f"global_potential_{agent}_contrib")      # potential_xxx
        + z(f"global_completion_{agent}_contrib")     # completion_xxx
        + (z(f"global_timeeff_{agent}_contrib") if use_time_eff_in_task else _zeros(device, num_envs))
        + z(f"{agent}force_contrib")                  # forceeff_xxx
        + z(f"{aware_key}_contrib")                   # humanaware / robotaware
    )

    # Safety risk (for training Vh) - negative values indicating danger
    r_safe_risk = (
        z(f"zoneC_total_{agent}")
        + z(f"zoneB_inward_{agent}_contrib")
        + z(f"zoneD_inward_{agent}_contrib")
        + (z(f"global_zpenalty_{agent}_contrib") if include_zpenalty_in_safe else _zeros(device, num_envs))
    )
    
    return r_task, r_safe_risk