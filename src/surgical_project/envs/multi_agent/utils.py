"""
Environment utilities for shared network MADDPG training.
FINAL VERSION: Essential components only, minimal complexity.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

class CompleteConstraintChecker:
    """Physics-based constraint analysis."""
    
    def __init__(self, device: torch.device, collision_threshold: float = 0.001):
        self.device = device
        self.collision_threshold = collision_threshold        
        try:
            from omni.physx.bindings._physx import acquire_physx_attachment_interface, acquire_physx_scene_query_interface
            self.physics_attachment_interface = acquire_physx_attachment_interface()
            self.physics_scene_query_interface = acquire_physx_scene_query_interface()
        except ImportError:
            self.physics_attachment_interface = None
            self.physics_scene_query_interface = None
    
    def analyze_constraint_state_batch(self, stylus_positions: torch.Tensor, env_base_positions: torch.Tensor):
        """Analyze constraint states for batch of environments."""
        num_envs = stylus_positions.shape[0]

        batch_results = {
            'distances_constraint': torch.ones(num_envs, device=self.device) * 0.02,
            'closest_points': torch.zeros(num_envs, 3, device=self.device),
            'normal_vectors': torch.ones(num_envs, 3, device=self.device),
            'is_overlapping': torch.zeros(num_envs, dtype=torch.bool, device=self.device),
            'is_inside': torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        }
        
        if self.physics_attachment_interface is None:
            return batch_results
        
        for env_id in range(num_envs):
            stylus_world_pos = stylus_positions[env_id] + env_base_positions[env_id]
            constraint_path = f"/World/envs/env_{env_id}/Constraint/Sphere"
            
            try:
                result = self._analyze_single_constraint(stylus_world_pos, constraint_path)
                if result is not None:
                    batch_results['distances_constraint'][env_id] = result['distance']
                    batch_results['closest_points'][env_id] = torch.tensor(result['closest_point'], device=self.device)
                    batch_results['normal_vectors'][env_id] = torch.tensor(result['normal_vector'], device=self.device)
                    batch_results['is_overlapping'][env_id] = result['is_overlapping']
                    batch_results['is_inside'][env_id] = result['is_inside']
            except Exception:
                pass
        
        return batch_results
    
    def _analyze_single_constraint(self, stylus_position: torch.Tensor, constraint_path: str):
        """Analyze constraint state for single environment."""
        try:
            from carb._carb import Float3
            
            pos = stylus_position.cpu().numpy()
            current_point = Float3(float(pos[0]), float(pos[1]), float(pos[2]))
            
            result = self.physics_attachment_interface.get_closest_points([current_point], constraint_path)
            
            if not (result and 'closest_points' in result and result['closest_points']):
                return None
            
            closest_pt = result['closest_points'][0]
            closest_pos = np.array([closest_pt.x, closest_pt.y, closest_pt.z])
            distance = float(np.linalg.norm(pos - closest_pos))
            
            direction_vec = closest_pos - pos
            direction_length = np.linalg.norm(direction_vec)
            
            if direction_length < 1e-8:
                return {
                    'distance': 0.0, 'closest_point': closest_pos,
                    'normal_vector': np.array([1.0, 0.0, 0.0]),
                    'is_overlapping': True, 'is_inside': False
                }
            
            direction_normalized = direction_vec / direction_length
            direction_to_closest = Float3(
                float(direction_normalized[0]),
                float(direction_normalized[1]),
                float(direction_normalized[2])
            )
            
            raycast_result = self.physics_scene_query_interface.raycast_closest(
                current_point, direction_to_closest, direction_length + 0.01
            )
            
            filtered_result = None
            if raycast_result and 'collision' in raycast_result:
                if constraint_path in raycast_result['collision']:
                    filtered_result = raycast_result
            
            # Determine state
            if distance > 1.2 or distance < 1e-8:
                is_overlapping, is_inside = True, False
                distance = 0.0
            elif distance < 0.002:
                is_overlapping, is_inside = True, False
                distance = 0.0
            elif not filtered_result or 'faceIndex' not in filtered_result:
                is_overlapping, is_inside = False, True
            else:
                is_overlapping, is_inside = False, False
            
            normal_vector = np.array([1.0, 0.0, 0.0])
            if filtered_result and 'normal' in filtered_result:
                normal_carb = filtered_result['normal']
                normal_vector = np.array([normal_carb.x, normal_carb.y, normal_carb.z])
                if not is_inside:
                    normal_vector = -normal_vector
            
            return {
                'distance': distance, 'closest_point': closest_pos,
                'normal_vector': normal_vector,
                'is_overlapping': is_overlapping, 'is_inside': is_inside
            }
            
        except Exception:
            return None


class TrajectoryManager:
    """Trajectory management for path following."""
    
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
    Unified global_step integration for shared network training.
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

    def maybe_print_step(self, env, rewards: Dict, global_step: int):
        """
        Complete zone-based console printing using unified global_step.
        Console print (zero storage) - only when enabled and throttled by step frequency.
        
        Args:
            env: Environment instance
            rewards: Reward dictionary
            global_step: Unified global step from trainer (hand-maintained)
        """
        if not self.enable_console_logging:
            return
        if global_step % self.print_every_steps != 0:
            return

        to_show = list(range(min(self.max_envs_to_print, self.num_envs)))

        print("=" * 80)
        print(f"STEP {global_step} - Four-Zone Reward System (A/B/C/D) - Shared Network")
        print("=" * 80)
        print(f"Showing first {len(to_show)} of {self.num_envs} environments")

        for env_id in to_show:
            self._print_env_snapshot(env, env_id)

        robot_total = rewards.get("robot", torch.zeros(self.num_envs, device=self.device)).mean().item()
        human_total = rewards.get("human", torch.zeros(self.num_envs, device=self.device)).mean().item()
        print(f"\n[TOTALS] Robot: {robot_total:+.3f} | Human: {human_total:+.3f} | Combined: {robot_total + human_total:+.3f}")

    def _print_env_snapshot(self, env, env_id: int):
        """Print complete four-zone reward breakdown for a single environment."""
        stylus = env.stylus_pos_t1[env_id]
        safety = float(env.safety_distances_t1[env_id].item())
        normal = env.normal_t1[env_id]
        dev = float(env.reward_components['deviation'][env_id].item())
        prog = float(env.reward_components['progress_ratio'][env_id].item())
        dist = float(env.reward_components['distance_to_final'][env_id].item())

        print(f"\n--- Environment {env_id} ---")
        print(f"Stylus: [{stylus[0]:.4f}, {stylus[1]:.4f}, {stylus[2]:.4f}] | Safety: {safety:.4f}m")
        print(f"Normal: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")

        self._print_actor_debug_info(env, env_id)

        print(f"Deviation: {dev:.4f}m | Progress: {prog:.1%} | Distance to End: {dist:.4f}m")

        D, O = 0.0075, 0.015
        if safety >= O:      active_zone = "A (Track)"
        elif safety <= D:    active_zone = "C (Danger)"
        else:
            if hasattr(env, 'rejoin_streak') and env.rejoin_streak[env_id] >= 10:
                active_zone = "D (Rejoin)"
            else:
                active_zone = "B (Surface)"
        print(f"Active Zone: {active_zone}")

        self._print_agent_zones_with_globals_flat(env, env_id, "ROBOT")
        self._print_agent_zones_with_globals_flat(env, env_id, "HUMAN")

    def _print_actor_debug_info(self, env, env_id: int):
        """Print actor network outputs and noise information with force consistency check."""
        if not (hasattr(env, 'actor_mean_forces') and hasattr(env, 'actor_noise_forces')):
            return
            
        # Get robot forces (mean and noise)
        robot_mean = env.actor_mean_forces.get('robot')
        robot_noise = env.actor_noise_forces.get('robot')
        
        # Get human forces (mean and noise) 
        human_mean = env.actor_mean_forces.get('human')
        human_noise = env.actor_noise_forces.get('human')
        
        if robot_mean is not None and robot_noise is not None:
            rm = robot_mean[env_id]
            rn = robot_noise[env_id]
            print(f"Forces (Robot): Fx={rm[0]:+.3f}, Fy={rm[1]:+.3f}, Fz={rm[2]:+.3f}")
            print(f"  Noise: Nx={rn[0]:+.3f}, Ny={rn[1]:+.3f}, Nz={rn[2]:+.3f}")
        
        if human_mean is not None and human_noise is not None:
            hm = human_mean[env_id]
            hn = human_noise[env_id]
            print(f"Forces (Human): Fx={hm[0]:+.3f}, Fy={hm[1]:+.3f}, Fz={hm[2]:+.3f}")
            print(f"  Noise: Nx={hn[0]:+.3f}, Ny={hn[1]:+.3f}, Nz={hn[2]:+.3f}")

    def _print_agent_zones_with_globals_flat(self, env, env_id: int, agent_label: str):
        """Print complete zone breakdown and global rewards for a single agent using flat keys."""
        agent_key = agent_label.lower()
        print(f"\n[{agent_label}] ZONE REWARDS:")

        zone_title = {'A':'A Track   ', 'B':'B Surface ', 'C':'C Danger  ', 'D':'D Rejoin  '}
        
        # Complete zone mapping
        mapping = {
            'A': [('progress','Progress'), ('deviation','Deviation')],
            'B': [('gap','Gap'), ('surftangent','Surf_Tangent'), ('inward','Inward_Pen')],
            'C': [('offpen','Off_Penalty'), ('inward','Inward_Pen')],
            'D': [('progress','Progress'), ('deviation','Deviation'), ('inward','Inward_Pen')],
        }

        def _val(x):
            if torch.is_tensor(x):
                return float(x[env_id].item()) if x.ndim>0 else float(x.item())
            return float(x)

        for Z in ['A','B','C','D']:
            zw_key = f'zone{Z}_weight_{agent_key}'
            zt_key = f'zone{Z}_total_{agent_key}'
            zone_w = env.reward_components.get(zw_key, 0.0)
            zone_total = env.reward_components.get(zt_key, 0.0)
            print(f"  {zone_title[Z]} (zone_w={_val(zone_w):.2f}): {_val(zone_total):+.3f}")

            for comp_key, comp_label in mapping[Z]:
                base = f"zone{Z}_{comp_key}_{agent_key}"
                raw     = env.reward_components.get(f"{base}_raw", 0.0)
                weight  = env.reward_components.get(f"{base}_weight", 0.0)
                contrib = env.reward_components.get(f"{base}_contrib", 0.0)
                print(f"    {comp_label:<12} {_val(raw):.3f} * {_val(weight):>.2f} = {_val(contrib):+.3f}")

        print("  Global Rewards:")
        for gk, glabel in [('zpenalty','Z Penalty      '),
                           ('completion','Completion     '),
                           ('timeeff','Time Efficiency')]:
            base = f"global_{gk}_{agent_key}"
            raw     = env.reward_components.get(f"{base}_raw", 0.0)
            weight  = env.reward_components.get(f"{base}_weight", 0.0)
            contrib = env.reward_components.get(f"{base}_contrib", 0.0)
            print(f"    {glabel:<15} {_val(raw):.3f} * {_val(weight):>.2f} = {_val(contrib):+.3f}")

        # Self force
        raw     = env.reward_components.get(f"{agent_key}force_raw", 0.0)
        weight  = env.reward_components.get(f"{agent_key}force_weight", 0.0)
        contrib = env.reward_components.get(f"{agent_key}force_contrib", 0.0)
        print(f"    {agent_label} Force   {_val(raw):.3f} * {_val(weight):>.2f} = {_val(contrib):+.3f}")

        # Awareness of other agent
        aware_key = 'humanaware' if agent_key == 'robot' else 'robotaware'
        raw     = env.reward_components.get(f"{aware_key}_raw", 0.0)
        weight  = env.reward_components.get(f"{aware_key}_weight", 0.0)
        contrib = env.reward_components.get(f"{aware_key}_contrib", 0.0)
        label = "Human Aware" if agent_key == 'robot' else "Robot Aware"
        print(f"    {label:<15} {_val(raw):.3f} * {_val(weight):>.2f} = {_val(contrib):+.3f}")


class MilestoneManager:
    """
    Simple milestone management for shared networks.
    NOTE: This is now primarily handled by MADDPGTrainer directly.
    Kept for backward compatibility.
    """
    
    def __init__(self, milestones: List[int]):
        self.milestones = sorted(milestones)
        self.completed = set()
        self.callback = None
    
    def set_callback(self, callback_fn):
        """Set milestone callback."""
        self.callback = callback_fn
    
    def check(self, global_episodes: int) -> Optional[int]:
        """Check if milestone reached."""
        for milestone in self.milestones:
            if global_episodes >= milestone and milestone not in self.completed:
                self.completed.add(milestone)
                print(f"[MILESTONE {milestone}] Reached at {global_episodes} episodes")
                
                if self.callback:
                    self.callback(milestone)
                
                return milestone
        return None


class RewardLogger:
    """
    Simple reward logger for shared networks with unified global_step integration.
    Milestone management now primarily handled by MADDPGTrainer.
    """

    def __init__(self, num_envs: int, device: torch.device, metrics_hub=None, 
                 enable_console_logging: bool = False, milestones: List[int] = None):
        self.num_envs = num_envs
        self.device = device
        self.metrics_hub = metrics_hub
        
        # Console debugging with unified global_step support
        self.step_tracer = StepTracer(
            num_envs, device,
            enable_console_logging=enable_console_logging,
            print_every_steps=10,
            max_envs_to_print=2
        )
        
        # Milestone management (NOTE: Legacy compatibility - now handled by MADDPGTrainer)
        self.milestone_manager = MilestoneManager(milestones or [100, 200, 300, 500])
        
        print(f"[RewardLogger] Initialized with {num_envs} environments")
        print(f"  Console logging: {'enabled' if enable_console_logging else 'disabled'}")
        print(f"  Milestone management: Legacy compatibility mode")
    
    def configure_logging(self, params: Dict):
        """Configure logging parameters."""
        logging_config = params.get('logging', {})
        enable_console = logging_config.get('enable_console_logging', False)
        
        # Get milestones from config
        training_monitor = params.get('training_monitor', {})
        milestones = training_monitor.get('milestone_episodes', [100, 200, 300, 500])
        
        self.milestone_manager = MilestoneManager(milestones)
        self.step_tracer.enable_console_logging = enable_console
        
        print(f"[RewardLogger] Configuration updated:")
        print(f"  Console logging: {'enabled' if enable_console else 'disabled'}")
        print(f"  Milestones: {milestones}")
    
    def set_topk_update_callback(self, callback_fn):
        """Set TopK callback (legacy compatibility)."""
        self.milestone_manager.set_callback(callback_fn)
        print(f"[RewardLogger] TopK callback registered (legacy mode)")
    
    def update_step_metrics_batch(self, reward_components: Dict, safety_distances: torch.Tensor, rewards: Dict):
        """Update step metrics for MetricsHub."""
        if self.metrics_hub:
            robot_avg = rewards.get("robot", torch.zeros(1, device=self.device)).mean().item()
            human_avg = rewards.get("human", torch.zeros(1, device=self.device)).mean().item()
            safety_avg = safety_distances.mean().item()
            
            self.metrics_hub.push_update(0, {
                "reward/robot_avg": robot_avg,
                "reward/human_avg": human_avg, 
                "safety/distance_avg": safety_avg,
            })
    
    def log_console_if_enabled(self, env, rewards: Dict, global_step: int):
        """
        Console logging using unified global_step from trainer.
        
        Args:
            env: Environment instance
            rewards: Reward dictionary  
            global_step: Unified global step from trainer (hand-maintained)
        """
        self.step_tracer.maybe_print_step(env, rewards, global_step)
    
    def check_milestone(self, global_episodes: int):
        """Check milestone (legacy compatibility - now handled by MADDPGTrainer)."""
        return self.milestone_manager.check(global_episodes)
    
    def close_all_files(self):
        """Cleanup (no-op for this implementation)."""
        pass