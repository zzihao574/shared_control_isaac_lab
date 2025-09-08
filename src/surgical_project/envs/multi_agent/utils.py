"""
Environment utilities for surgical robot training.
Contains constraint checking, trajectory management, and reward logging utilities.
Modified to support active environment filtering and dual protection mechanism.
MODIFIED: Added EpisodeAccumulator, updated RewardLogger and MilestoneManager for MetricsHub integration
"""

import torch
import numpy as np
import yaml
import os
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

class CompleteConstraintChecker:
    """Physics-based constraint analysis for surgical robot environments."""
    
    def __init__(self, device: torch.device, collision_threshold: float = 0.001):
        """Initialize constraint checker with collision detection."""
        self.device = device
        self.collision_threshold = collision_threshold        
        from omni.physx.bindings._physx import acquire_physx_attachment_interface, acquire_physx_scene_query_interface
        self.physics_attachment_interface = acquire_physx_attachment_interface()
        self.physics_scene_query_interface = acquire_physx_scene_query_interface()
    
    def analyze_constraint_state_batch(self, stylus_positions: torch.Tensor, env_base_positions: torch.Tensor):
        """Analyze constraint states for batch of environments."""
        num_envs = stylus_positions.shape[0]
        current_base_positions = env_base_positions

        batch_results = {
            'distances_constraint': torch.ones(num_envs, device=self.device) * 0.02,
            'closest_points': torch.zeros(num_envs, 3, device=self.device),
            'normal_vectors': torch.ones(num_envs, 3, device=self.device),
            'is_overlapping': torch.zeros(num_envs, dtype=torch.bool, device=self.device),
            'is_inside': torch.zeros(num_envs, dtype=torch.bool, device=self.device)
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
                    batch_results['is_inside'][env_id] = result['is_inside']
            except Exception:
                pass
        
        return batch_results
    
    def _analyze_single_constraint(self, stylus_position: torch.Tensor, constraint_path: str):
        """Analyze constraint state for single environment using physics raycast."""
        try:
            from carb._carb import Float3
            
            pos = stylus_position.cpu().numpy()
            current_point = Float3(float(pos[0]), float(pos[1]), float(pos[2]))
            
            # Get closest point on constraint surface
            result = self.physics_attachment_interface.get_closest_points([current_point], constraint_path)
            
            if not (result and 'closest_points' in result and result['closest_points']):
                return None
            
            closest_pt = result['closest_points'][0]
            closest_pos = np.array([closest_pt.x, closest_pt.y, closest_pt.z])
            
            # Calculate distance to constraint
            distance = float(np.linalg.norm(pos - closest_pos))
            
            # Raycast from stylus to closest point for normal detection
            direction_vec = closest_pos - pos
            direction_length = np.linalg.norm(direction_vec)
            
            if direction_length < 1e-8:
                return {
                    'distance': 0.0,
                    'closest_point': closest_pos,
                    'normal_vector': np.array([1.0, 0.0, 0.0]),
                    'state': "overlapping",
                    'is_overlapping': True,
                    'is_inside': False
                }
            
            # Execute raycast for state detection
            direction_normalized = direction_vec / direction_length
            direction_to_closest = Float3(
                float(direction_normalized[0]),
                float(direction_normalized[1]),
                float(direction_normalized[2])
            )
            
            raycast_result = self.physics_scene_query_interface.raycast_closest(
                current_point, direction_to_closest, direction_length + 0.01
            )
            
            # Filter raycast results for constraint object only
            filtered_raycast_result = None
            if raycast_result and 'collision' in raycast_result:
                collision_path = raycast_result['collision']
                if constraint_path in collision_path:
                    filtered_raycast_result = raycast_result
            
            # Determine constraint state
            is_overlapping = False
            is_inside = False
            normal_vector = np.array([1.0, 0.0, 0.0])
            
            # Handle abnormal distances (physics interface error)
            if distance > 1.2 or distance < 1e-8:
                is_overlapping = True
                is_inside = False
                distance = 0.0
                state = "overlapping"
                normal_vector = np.array([1.0, 0.0, 0.0])
                    
            elif distance < 0.002:  # 2mm overlap threshold
                is_overlapping = True
                is_inside = False
                distance = 0.0
                state = "overlapping"
                
                if filtered_raycast_result and 'normal' in filtered_raycast_result:
                    normal_carb = filtered_raycast_result['normal']
                    normal_vector = np.array([normal_carb.x, normal_carb.y, normal_carb.z])
                else:
                    normal_vector = np.array([1.0, 0.0, 0.0])
                
            elif not filtered_raycast_result or 'faceIndex' not in filtered_raycast_result:
                # No ray intersection - stylus is inside constraint
                is_inside = True
                is_overlapping = False
                state = "inside"
                normal_vector = np.array([1.0, 0.0, 0.0])
            
            else:
                # Ray intersection exists - stylus is outside constraint
                is_inside = False
                is_overlapping = False
                state = "outside"
                
                if 'normal' in filtered_raycast_result:
                    normal_carb = filtered_raycast_result['normal']
                    normal_vector = -np.array([normal_carb.x, normal_carb.y, normal_carb.z])
            
            return {
                'distance': distance,
                'closest_point': closest_pos,
                'normal_vector': normal_vector,
                'state': state,
                'is_overlapping': is_overlapping,
                'is_inside': is_inside
            }
            
        except Exception:
            return None


class TrajectoryManager:
    """Trajectory management and progress tracking for linear path following."""
    
    def __init__(self, device: torch.device, params: dict, num_envs: int, env_base_positions: torch.Tensor):
        """Initialize trajectory manager with start/end points."""
        self.device = device
        self.num_envs = num_envs
        self.env_base_positions = env_base_positions
        
        traj = params['trajectory']
        self.start_pos_local = torch.tensor(traj['start_point'], device=device, dtype=torch.float32)
        self.end_pos_local = torch.tensor(traj['end_point'], device=device, dtype=torch.float32)
        self.total_distance = torch.norm(self.end_pos_local - self.start_pos_local).item()
        
        # Unit direction vector for straight line trajectory
        self.line_direction = (self.end_pos_local - self.start_pos_local) / self.total_distance
        self.switch_threshold = traj.get('switch_threshold', 0.01)  # 1cm threshold
    
    def get_progress(self, current_pos_local: torch.Tensor) -> torch.Tensor:
        """Calculate progress along trajectory (0 to 1)."""
        vec_to_current = current_pos_local - self.start_pos_local.unsqueeze(0)
        progress_distance = torch.sum(vec_to_current * self.line_direction.unsqueeze(0), dim=-1)
        progress_distance = torch.clamp(progress_distance, 0, self.total_distance)
        return progress_distance / self.total_distance
    
    def get_deviation(self, current_pos_local: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate perpendicular distance to trajectory and closest points."""
        vec_to_current = current_pos_local - self.start_pos_local.unsqueeze(0)
        progress_distance = torch.sum(vec_to_current * self.line_direction.unsqueeze(0), dim=-1)
        progress_distance = torch.clamp(progress_distance, 0, self.total_distance)
        
        # Project to closest points on trajectory line
        closest_points = self.start_pos_local.unsqueeze(0) + progress_distance.unsqueeze(-1) * self.line_direction.unsqueeze(0)
        
        # Calculate perpendicular deviations
        deviations = torch.norm(current_pos_local - closest_points, dim=-1)
        
        return deviations, closest_points
    
    def is_final_setpoint_reached(self, current_pos_local: torch.Tensor) -> torch.Tensor:
        """Check if final trajectory setpoint is reached."""
        distances_to_final = torch.norm(current_pos_local - self.end_pos_local.unsqueeze(0), dim=-1)
        return distances_to_final < self.switch_threshold
    
    def get_trajectory_info(self) -> Dict:
        """Get trajectory configuration information."""
        return {
            'start_point': self.start_pos_local.cpu().numpy(),
            'end_point': self.end_pos_local.cpu().numpy(),
            'total_distance': self.total_distance
        }


# === NEW CLASS: EpisodeAccumulator ===
class EpisodeAccumulator:
    """Per-episode accumulator per env (no printing, no cross-episode stats)."""
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.reset_all()

    def reset_all(self):
        self.basic = [{
            'steps': 0,
            'total_reward': 0.0,
            'final_progress': 0.0,
            'completed': False,
            'collision': False,
            'min_safety_distance': float('inf'),
            'deviation_sum': 0.0,
        } for _ in range(self.num_envs)]

    def update_batch(self, env_ids, reward_components, safety_distance, deviation, progress):
        for env_id in env_ids:
            b = self.basic[env_id]
            b['steps'] += 1
            
            # Handle total reward from reward_components if available
            if isinstance(reward_components, dict) and 'total' in reward_components:
                if torch.is_tensor(reward_components['total']):
                    b['total_reward'] += float(reward_components['total'][env_id].item())
                else:
                    b['total_reward'] += float(reward_components['total'][env_id])
            
            if progress is not None:
                if torch.is_tensor(progress):
                    b['final_progress'] = float(progress[env_id].item())
                else:
                    b['final_progress'] = float(progress[env_id])
            
            if deviation is not None:
                if torch.is_tensor(deviation):
                    b['deviation_sum'] += float(deviation[env_id].item())
                else:
                    b['deviation_sum'] += float(deviation[env_id])
            
            if safety_distance is not None:
                if torch.is_tensor(safety_distance):
                    sd = float(safety_distance[env_id].item())
                else:
                    sd = float(safety_distance[env_id])
                if sd < b['min_safety_distance']:
                    b['min_safety_distance'] = sd

    def mark_completed(self, env_id, completed: bool):
        self.basic[env_id]['completed'] = bool(completed)

    def mark_collision(self, env_id, collided: bool):
        self.basic[env_id]['collision'] = bool(collided)

    def summary(self, env_id: int) -> dict:
        b = self.basic[env_id]
        steps = max(1, b['steps'])
        avg_dev = b['deviation_sum'] / steps
        return {
            'episode_len': b['steps'],
            'episode_return': b['total_reward'],
            'final_progress': b['final_progress'],
            'success': b['completed'],
            'collision': b['collision'],
            'min_safety_distance': (0.0 if b['min_safety_distance'] == float('inf') else b['min_safety_distance']),
            'avg_deviation': avg_dev,
        }

    def reset_envs(self, env_ids):
        for env_id in env_ids:
            self.basic[env_id] = {
                'steps': 0, 'total_reward': 0.0, 'final_progress': 0.0,
                'completed': False, 'collision': False,
                'min_safety_distance': float('inf'), 'deviation_sum': 0.0,
            }


class StepTracer:
    """
    Tracks per-episode lightweight stats with optional step-wise console prints.
    Enhanced with active environment filtering for dual protection mechanism.
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

        # Per-environment current episode lightweight scalars
        self.current_episode_basic = {
            env_id: {
                'steps': 0,
                'total_reward': 0.0,
                'final_progress': 0.0,
                'completed': False,
                'collision': False,
                'min_safety_distance': float('inf'),
            } for env_id in range(num_envs)
        }

        # Minimal cross-episode accumulation
        self.basic_stats = {
            env_id: {
                'total_episodes': 0,
                'completed_episodes': 0,
                'collision_episodes': 0,
            } for env_id in range(num_envs)
        }
        
        # Track previously active environments to show one-time notifications
        self.previously_active_envs = set(range(num_envs))

    def update_step_metrics_batch(self, reward_components: Dict, safety_distances: torch.Tensor, rewards: Dict):
        """Vectorized batch update for all environments to reduce CPU-GPU sync."""
        robot_rewards = rewards.get("robot", torch.zeros(self.num_envs, device=self.device))
        human_rewards = rewards.get("human", torch.zeros(self.num_envs, device=self.device))
        
        # Convert tensors to numpy for faster CPU operations
        robot_rewards_np = robot_rewards.detach().cpu().numpy()
        human_rewards_np = human_rewards.detach().cpu().numpy()
        total_rewards_np = robot_rewards_np + human_rewards_np
        progress_ratios_np = reward_components['progress_ratio'].detach().cpu().numpy()
        safety_distances_np = safety_distances.detach().cpu().numpy()
        completion_rewards_np = reward_components['completion_reward'].detach().cpu().numpy()
        
        # Batch update all environments
        for env_id in range(self.num_envs):
            basic = self.current_episode_basic[env_id]
            basic['total_reward'] += total_rewards_np[env_id]
            basic['final_progress'] = progress_ratios_np[env_id]
            basic['min_safety_distance'] = min(basic['min_safety_distance'], safety_distances_np[env_id])
            
            if completion_rewards_np[env_id] > 0:
                basic['completed'] = True
            if safety_distances_np[env_id] < 0.001:
                basic['collision'] = True

    def maybe_print_step(self, env, rewards: Dict, robot_weights: dict, human_weights: dict, current_step: int, active_envs: Optional[List[int]] = None):
        """
        Console print (zero storage) - only for active environments when enabled and throttled by step frequency.
        Enhanced with active environment filtering for dual protection mechanism.
        Shows weighted contributions to match actual training.
        """
        if not self.enable_console_logging:
            return
        if current_step % self.print_every_steps != 0:
            return

        # Show notifications for newly inactive environments (one time only)
        if active_envs is not None:
            current_active_set = set(active_envs)
            newly_inactive = self.previously_active_envs - current_active_set
            if newly_inactive:
                print(f"\n[INACTIVE] Environments {sorted(newly_inactive)} are no longer active (dual protection)")
            self.previously_active_envs = current_active_set

        # Only show first N active environments to avoid screen flooding
        print(f"\n{'='*80}")
        print(f"STEP {current_step} - Environment State (Active Environments Only)")
        print(f"{'='*80}")
        
        if active_envs is None:
            # Fallback: print all environments
            ids_to_print = range(min(self.max_envs_to_print, env.num_envs))
            print("(Fallback mode: showing all environments)")
        else:
            ids_to_print = active_envs[:self.max_envs_to_print]
            print(f"Active environments: {len(active_envs)}/{env.num_envs} | Showing first {min(len(active_envs), self.max_envs_to_print)}")
        
        for env_id in ids_to_print:
            print(f"\n--- [ACTIVE] Environment {env_id} ---")
            
            # Position information
            stylus_pos = env.stylus_pos_t1[env_id]
            print(f"Stylus Position (local): [{stylus_pos[0]:.4f}, {stylus_pos[1]:.4f}, {stylus_pos[2]:.4f}]")
            
            # Trajectory metrics
            deviation = env.reward_components['deviation'][env_id].item()
            progress = env.reward_components['progress_ratio'][env_id].item()
            distance_to_final = env.reward_components['distance_to_final'][env_id].item()
            print(f"Trajectory - Deviation: {deviation:.4f}m, Progress: {progress:.1%}, Distance to Final: {distance_to_final:.4f}m")
            
            # Constraint status
            safety_distance = env.safety_distances_t1[env_id].item()
            is_overlapping = env.is_violating_t1[env_id].item()
            normals = env.normal_t1[env_id]
            print(f"Constraint - Safety Distance: {safety_distance:.4f}m, Overlapping: {is_overlapping}, Normals: {normals}")
            
            # Force magnitudes
            robot_force = env.robot_forces_t[env_id]
            human_force = env.human_forces_t[env_id]
            robot_force_mag = torch.norm(robot_force).item()
            human_force_mag = torch.norm(human_force).item()
            print(f"Forces - Robot: {robot_force_mag:.3f}N, Human: {human_force_mag:.3f}N")
            
            # ENHANCED: Detailed reward breakdown with weighted contributions
            print(f"\nReward Breakdown (Weighted Contributions):")
            print(f"Robot Agent:")
            traj_r = env.reward_components['trajectory_reward'][env_id].item()
            prog_r = env.reward_components['progress_reward'][env_id].item()
            potential_r = env.reward_components.get('potential_field_reward', env.reward_components['trajectory_reward'])[env_id].item()
            robot_force_pen = env.reward_components['robot_force_penalty'][env_id].item()
            human_force_pen = env.reward_components['human_force_penalty'][env_id].item()
            z_pen = env.reward_components['z_penalty'][env_id].item()
            comp_r = env.reward_components['completion_reward'][env_id].item()
            time_eff_r = env.reward_components.get('time_efficiency_reward', env.reward_components['trajectory_reward'])[env_id].item()
            
            print(f"  Trajectory: {traj_r:.3f} * {robot_weights['trajectory_tracking']:.2f} = {traj_r * robot_weights['trajectory_tracking']:.3f}")
            print(f"  Progress: {prog_r:.3f} * {robot_weights['progress']:.2f} = {prog_r * robot_weights['progress']:.3f}")
            print(f"  Potential Field: {potential_r:.3f} * {robot_weights['potential_field']:.2f} = {potential_r * robot_weights['potential_field']:.3f}")
            print(f"  Robot Force: {robot_force_pen:.3f} * {robot_weights['force_efficiency']:.2f} = {robot_force_pen * robot_weights['force_efficiency']:.3f}")
            print(f"  Human Awareness: {human_force_pen:.3f} * {robot_weights['human_awareness']:.2f} = {human_force_pen * robot_weights['human_awareness']:.3f}")
            # ENHANCED: Show weighted global components
            print(f"  Z Penalty: {z_pen:.3f} * {robot_weights.get('z_penalty',0.0):.2f} = {z_pen * robot_weights.get('z_penalty',0.0):.3f}")
            print(f"  Completion: {comp_r:.3f} * {robot_weights.get('completion_reward',0.0):.2f} = {comp_r * robot_weights.get('completion_reward',0.0):.3f}")
            print(f"  Time Efficiency: {time_eff_r:.3f} * {robot_weights.get('time_efficiency',0.0):.2f} = {time_eff_r * robot_weights.get('time_efficiency',0.0):.3f}")
            robot_total = rewards["robot"][env_id].item()
            print(f"  ROBOT TOTAL: {robot_total:.3f}")
            
            print(f"Human Agent:")
            print(f"  Trajectory: {traj_r:.3f} * {human_weights['trajectory_tracking']:.2f} = {traj_r * human_weights['trajectory_tracking']:.3f}")
            print(f"  Progress: {prog_r:.3f} * {human_weights['progress']:.2f} = {prog_r * human_weights['progress']:.3f}")
            print(f"  Potential Field: {potential_r:.3f} * {human_weights['potential_field']:.2f} = {potential_r * human_weights['potential_field']:.3f}")
            print(f"  Human Force: {human_force_pen:.3f} * {human_weights['force_efficiency']:.2f} = {human_force_pen * human_weights['force_efficiency']:.3f}")
            print(f"  Robot Awareness: {robot_force_pen:.3f} * {human_weights['robot_awareness']:.2f} = {robot_force_pen * human_weights['robot_awareness']:.3f}")
            # ENHANCED: Show weighted global components
            print(f"  Z Penalty: {z_pen:.3f} * {human_weights.get('z_penalty',0.0):.2f} = {z_pen * human_weights.get('z_penalty',0.0):.3f}")
            print(f"  Completion: {comp_r:.3f} * {human_weights.get('completion_reward',0.0):.2f} = {comp_r * human_weights.get('completion_reward',0.0):.3f}")
            print(f"  Time Efficiency: {time_eff_r:.3f} * {human_weights.get('time_efficiency',0.0):.2f} = {time_eff_r * human_weights.get('time_efficiency',0.0):.3f}")
            human_total = rewards["human"][env_id].item()
            print(f"  HUMAN TOTAL: {human_total:.3f}")
            
            print(f"Combined Total Reward: {robot_total + human_total:.3f}")

    def on_episode_end(self, env_ids):
        """Roll up current-episode stats into basic_stats, then reset episode state."""
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor([env_ids] if isinstance(env_ids, int) else env_ids, device=self.device)

        for env_id_t in env_ids:
            env_id = env_id_t.item()
            basic = self.current_episode_basic[env_id]

            # Minimal accumulation
            stats = self.basic_stats[env_id]
            stats['total_episodes'] += 1
            if basic['completed']:
                stats['completed_episodes'] += 1
            if basic['collision']:
                stats['collision_episodes'] += 1

            # Reset current episode data
            self.current_episode_basic[env_id] = {
                'steps': 0,
                'total_reward': 0.0,
                'final_progress': 0.0,
                'completed': False,
                'collision': False,
                'min_safety_distance': float('inf'),
            }


class MilestoneManager:
    """MODIFIED: Manages milestone episode detection and performance tracking using MetricsHub."""
    
    def __init__(self, num_envs: int, milestones: List[int]):
        self.num_envs = num_envs
        self.milestones = milestones
        self.milestone_performances = {m: {} for m in milestones}
        self.milestone_completion_status = {m: {'completed_envs': 0, 'reported': False} for m in milestones}
        self.topk_update_callback = None
        
        # NEW: Reference to MetricsHub (will be injected)
        self.metrics_hub = None
        
        # NEW: Reference to progress manager (will be injected)
        self.progress_manager = None
    
    def set_topk_update_callback(self, callback_fn):
        """Set callback function triggered at milestone completion."""
        self.topk_update_callback = callback_fn
    
    def is_milestone_episode(self, episode_num: int) -> bool:
        """Check if episode number is a milestone."""
        return episode_num in self.milestones
    
    def _compute_score_from_episode(self, ep: dict) -> float:
        """NEW: Calculate score based on episode summary instead of detailed data"""
        # completion 40, progress 20, trajectory 20, safety 20
        score = 0.0
        score += 40.0 if ep.get('success', False) else 0.0
        score += 20.0 * max(0.0, min(1.0, float(ep.get('final_progress', 0.0))))
        
        # Use avg_deviation from episode summary
        dev = float(ep.get('avg_deviation', 0.0))
        traj_term = max(0.0, min(1.0, (0.05 - dev) / 0.04))  # Same formula as before
        score += 20.0 * traj_term
        
        msd = float(ep.get('min_safety_distance', 0.0))
        safety_term = max(0.0, min(1.0, msd / 0.02))
        score += 20.0 * safety_term
        
        return max(0.0, min(100.0, score))
    
    def process_milestone_completion(self, env_id: int, episode_num: int, basic_data: Dict):
        """MODIFIED: Process milestone using episode summary from MetricsHub"""
        if not self.is_milestone_episode(episode_num):
            return
        
        # Get episode summary from MetricsHub if available
        if self.metrics_hub:
            ep_summary = self.metrics_hub.get_episode_state(env_id)
            if ep_summary:
                score = self._compute_score_from_episode(ep_summary)
            else:
                # Fallback: estimate from basic_data
                score = PerformanceEvaluator.calculate_performance_score(basic_data, None)
        else:
            # Fallback: use old method
            score = PerformanceEvaluator.calculate_performance_score(basic_data, None)
        
        # Log milestone completion
        print(f"[MILESTONE {episode_num}] Env {env_id}: "
              f"Steps={basic_data['steps']}, Progress={basic_data['final_progress']:.1%}, "
              f"Completed={basic_data['completed']}, Collision={basic_data['collision']}, "
              f"Score={score:.2f}/100")
        
        # Store performance data
        self.milestone_performances[episode_num][env_id] = {
            'score': score,
            'completed': basic_data['completed'],
            'steps': basic_data['steps'],
            'collision': basic_data['collision'],
            'final_progress': basic_data['final_progress'],
            'avg_reward': basic_data['total_reward'] / basic_data['steps'] if basic_data['steps'] > 0 else 0.0
        }
        
        # Update completion status
        self.milestone_completion_status[episode_num]['completed_envs'] += 1
        
        # Check for milestone completion
        self._check_milestone_completion(episode_num)
    
    def _check_milestone_completion(self, milestone: int):
        """MODIFIED: Check if milestone is complete and push to MetricsHub"""
        status = self.milestone_completion_status[milestone]
        
        # Trigger callback when all environments reach milestone
        if status['completed_envs'] >= self.num_envs and not status['reported']:
            if self.topk_update_callback:
                print(f"[CALLBACK] Triggering Top-K update for milestone {milestone}...")
                self.topk_update_callback(milestone)

            # Push milestone summary to MetricsHub
            if self.metrics_hub:
                scores = {}
                for env_id in range(self.num_envs):
                    if env_id in self.milestone_performances[milestone]:
                        scores[env_id] = self.milestone_performances[milestone][env_id]['score']
                
                if self.progress_manager:
                    step = getattr(self.progress_manager, 'global_step', 0)
                else:
                    step = 0
                
                self.metrics_hub.push_milestone_summary(milestone, {
                    "scores": scores,
                    "completed_envs": len(scores),
                    "total_envs": self.num_envs,
                    "step": step,
                    "milestone": milestone
                })

            self._report_milestone_summary(milestone)
            status['reported'] = True
    
    def _report_milestone_summary(self, milestone: int):
        """Generate milestone completion summary report."""
        performances = self.milestone_performances[milestone]
        
        if len(performances) != self.num_envs:
            print(f"[WARNING] Milestone {milestone}: Only {len(performances)}/{self.num_envs} environments have data")
            return
        
        scores = [p['score'] for p in performances.values()]
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        completion_rate = sum(1 for p in performances.values() if p['completed']) / len(performances)
        avg_steps = np.mean([p['steps'] for p in performances.values()])
        collision_rate = sum(1 for p in performances.values() if p['collision']) / len(performances)
        avg_progress = np.mean([p['final_progress'] for p in performances.values()])
        
        print(f"\n{'='*60}")
        print(f"[MILESTONE {milestone} COMPLETE] All environments reached")
        print(f"{'='*60}")
        print(f"  Average Score: {avg_score:.2f} ± {std_score:.2f} / 100")
        print(f"  Completion Rate: {completion_rate:.1%}")
        print(f"  Collision Rate: {collision_rate:.1%}")
        print(f"  Average Progress: {avg_progress:.1%}")
        print(f"  Average Steps: {avg_steps:.1f}")
        print(f"{'='*60}\n")
    
    def get_next_milestone_progress(self, episode_counts):
        """Get progress toward next milestone using external counter."""
        if isinstance(episode_counts, list):
            min_episodes = min(episode_counts)
        elif torch.is_tensor(episode_counts):
            min_episodes = episode_counts.min().item()
        else:
            # Handle other numeric types
            min_episodes = int(episode_counts)
            
        for milestone in self.milestones:
            if milestone > min_episodes:
                if isinstance(episode_counts, list):
                    reached = sum(1 for count in episode_counts if count >= milestone)
                elif torch.is_tensor(episode_counts):
                    reached = (episode_counts >= milestone).sum().item()
                else:
                    reached = 1 if episode_counts >= milestone else 0
                return milestone, reached, self.num_envs - reached
        return None, 0, 0


class PerformanceEvaluator:
    """Evaluates agent performance and calculates composite scores."""
    
    @staticmethod
    def calculate_performance_score(basic_data: Dict, detailed_data: Optional[Dict] = None) -> float:
        """Calculate performance score (0-100) based on episode data."""
        if basic_data['steps'] == 0:
            return 0.0
        
        # Completion bonus (40 points)
        completion_score = 40.0 if basic_data['completed'] else 0.0
        
        # Progress score (20 points)
        progress_score = 20.0 * min(1.0, basic_data['final_progress'])
        
        # Trajectory precision (20 points)
        trajectory_score = 0.0
        if detailed_data and 'deviations' in detailed_data and detailed_data['deviations']:
            avg_deviation = np.mean(detailed_data['deviations'])
            trajectory_score = 20.0 * max(0, min(1.0, (0.05 - avg_deviation) / 0.04))
        else:
            # Estimate trajectory score from completion
            trajectory_score = 10.0 if basic_data['completed'] else 5.0
        
        # Safety score (20 points)
        min_safety = basic_data.get('min_safety_distance', float('inf'))
        
        if basic_data['collision']:
            safety_score = 0.0
        else:
            if min_safety == float('inf'):
                safety_score = 10.0
            else:
                safety_score = 20.0 * max(0, min(1.0, (min_safety - 0.001) / 0.009))
        
        total_score = completion_score + progress_score + trajectory_score + safety_score
        return np.clip(total_score, 0, 100)
    
    @staticmethod
    def get_final_evaluation(milestone_performances: Dict, basic_stats: Dict, 
                           env_id: int, target_episodes: int, milestones: List[int]) -> float:
        """Get final evaluation score for an environment."""
        # Use recent milestone scores for evaluation
        recent_milestones = [m for m in milestones if m <= target_episodes][-3:]
        scores = []
        
        for milestone in recent_milestones:
            if milestone in milestone_performances:
                if env_id in milestone_performances[milestone]:
                    score = milestone_performances[milestone][env_id]['score']
                    scores.append(score)
        
        if scores:
            final_score = np.mean(scores)
        else:
            # Fallback estimation from basic statistics
            stats = basic_stats[env_id]
            if stats['total_episodes'] > 0:
                completion_rate = stats['completed_episodes'] / stats['total_episodes']
                collision_rate = stats['collision_episodes'] / stats['total_episodes']
                final_score = (completion_rate * 60) + (1 - collision_rate) * 40
            else:
                final_score = 0.0
        
        return np.clip(final_score, 0, 100)


class RewardLogger:
    """
    MODIFIED: Optimized reward logger using MetricsHub for data pipeline
    """

    def __init__(self, num_envs: int, device: torch.device, metrics_hub=None, 
                 enable_console_logging: bool = False, milestones: List[int] = None):
        self.num_envs = num_envs
        self.device = device
        
        # NEW: MetricsHub reference
        self.metrics_hub = metrics_hub
        
        # Progress manager reference (set by trainer)
        self.progress_manager = None
        
        # NEW: Use EpisodeAccumulator instead of StepTracer for episode tracking
        self.episode_acc = EpisodeAccumulator(num_envs)
        
        # Keep StepTracer for console logging only
        self.step_tracer = StepTracer(
            num_envs, device,
            enable_console_logging=enable_console_logging,
            print_every_steps=10,
            max_envs_to_print=2
        )
        
        # Initialize milestone manager with hub reference
        if milestones:
            self.milestone_manager = MilestoneManager(num_envs, milestones)
            self.milestone_manager.metrics_hub = metrics_hub
        else:
            self.milestone_manager = None
        
        # External episode counter reference (set by trainer)
        self.external_episode_counter = None
        
        # Compatibility aliases
        self.current_episode_basic = self.step_tracer.current_episode_basic
        self.basic_stats = self.step_tracer.basic_stats
    
    def set_progress_manager(self, pm):
        """Set progress manager reference."""
        self.progress_manager = pm
        if self.milestone_manager:
            self.milestone_manager.progress_manager = pm
    
    def set_episode_counter(self, counter_reference):
        """Set reference to external episode counter."""
        self.external_episode_counter = counter_reference

    def configure_logging(self, params: Dict):
        """Configure logging components from parameters."""
        logging_config = params.get('logging', {})
        enable_console_logging = logging_config.get('enable_console_logging', False)
        
        # Load milestone configuration
        training_monitor = params.get('training_monitor', {})
        yaml_milestones = training_monitor.get('milestone_episodes', None)
        
        if yaml_milestones:
            milestones = yaml_milestones
            print(f"[INFO] Using YAML configured milestones: {milestones}")
        else:
            milestones = [2, 10, 20, 30, 50, 100]
            print(f"[INFO] Using default milestones: {milestones}")
        
        # Initialize milestone manager if not already done
        if not self.milestone_manager:
            self.milestone_manager = MilestoneManager(self.num_envs, milestones)
            self.milestone_manager.metrics_hub = self.metrics_hub
        
        # Override StepTracer's console logging setting
        self.step_tracer.enable_console_logging = enable_console_logging
        print(f"[INFO] Console logging {'enabled' if enable_console_logging else 'disabled'} (via StepTracer)")
        print(f"[INFO] RewardLogger initialized: {self.num_envs} environments")
        print(f"[INFO] Performance evaluation at episodes: {milestones}")
    
    def set_topk_update_callback(self, callback_fn):
        """Set callback for Top-K model updates at milestones."""
        if self.milestone_manager:
            self.milestone_manager.set_topk_update_callback(callback_fn)
    
    def update_step_metrics_batch(self, reward_components: Dict, safety_distances: torch.Tensor, rewards: Dict):
        """MODIFIED: Update metrics and push to hub"""
        # Update step counts from progress manager
        steps_batch_np = None
        if self.progress_manager:
            steps_batch = self.progress_manager.step_counts
            # Write to StepTracer for console logging
            for env_id in range(self.num_envs):
                self.step_tracer.current_episode_basic[env_id]['steps'] = int(steps_batch[env_id].item())
        
        # Update StepTracer for console logging
        self.step_tracer.update_step_metrics_batch(reward_components, safety_distances, rewards)
        
        # NEW: Update EpisodeAccumulator with batch data
        env_ids = list(range(self.num_envs))
        
        # Calculate total rewards for accumulator
        robot_rewards = rewards.get("robot", torch.zeros(self.num_envs, device=self.device))
        human_rewards = rewards.get("human", torch.zeros(self.num_envs, device=self.device))
        total_rewards = robot_rewards + human_rewards
        
        # Create reward_components dict with total field for accumulator
        reward_comp_with_total = dict(reward_components)
        reward_comp_with_total['total'] = total_rewards
        
        self.episode_acc.update_batch(
            env_ids, 
            reward_comp_with_total,
            safety_distances,
            reward_components.get('deviation'),
            reward_components.get('progress_ratio')
        )
        
        # Check for completion/collision markers
        if 'completion_reward' in reward_components:
            completion_rewards = reward_components['completion_reward']
            for env_id in range(self.num_envs):
                if completion_rewards[env_id] > 0:
                    self.episode_acc.mark_completed(env_id, True)
        
        if safety_distances is not None:
            for env_id in range(self.num_envs):
                if safety_distances[env_id] < 0.001:
                    self.episode_acc.mark_collision(env_id, True)
        
        # Push step data to hub if available
        if self.metrics_hub and self.progress_manager:
            # Get global step from trainer's progress manager
            global_step = getattr(self.progress_manager, 'global_step', 0)
            if hasattr(self.progress_manager, 'training_logger'):
                # Try to get from trainer
                global_step = getattr(self.progress_manager.training_logger, 'global_step', global_step)
            
            self.metrics_hub.push_step(
                step=global_step,
                env_ids=env_ids,
                payload={
                    "reward_components": reward_components,
                    "safety_distance": safety_distances,
                    "progress_ratio": reward_components.get('progress_ratio'),
                    "deviation": reward_components.get('deviation'),
                }
            )
    
    def log_console_if_enabled(self, env, rewards: Dict, robot_weights: dict, human_weights: dict):
        """Log to console if enabled with active environment filtering."""
        # Get active environments from progress manager if available
        active_envs = None
        if self.progress_manager:
            active_envs = self.progress_manager.get_active_environments()
        
        # StepTracer controls whether to print and frequency throttling
        current_step = env.common_step_counter
        self.step_tracer.maybe_print_step(env, rewards, robot_weights, human_weights, current_step, active_envs)
    
    def on_episode_end(self, env_ids):
        """MODIFIED: Handle episode completion using MetricsHub"""
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor([env_ids] if isinstance(env_ids, int) else env_ids, device=self.device)
        
        for env_id_t in env_ids:
            env_id = env_id_t.item()
            
            # Get episode summary from accumulator
            summary = self.episode_acc.summary(env_id)
            
            # Push to MetricsHub
            if self.metrics_hub and self.progress_manager:
                # Get global step
                global_step = 0
                if hasattr(self.progress_manager, 'training_logger'):
                    global_step = getattr(self.progress_manager.training_logger, 'global_step', 0)
                
                self.metrics_hub.push_episode(global_step, env_id, summary)
            
            # Get episode number from external counter
            current_episode_num = int(self.external_episode_counter[env_id]) if self.external_episode_counter is not None else 1
            basic = self.step_tracer.current_episode_basic[env_id]
            
            # Process milestones using hub data
            if self.milestone_manager and self.milestone_manager.is_milestone_episode(current_episode_num):
                if basic['steps'] > 0:
                    self.milestone_manager.process_milestone_completion(env_id, current_episode_num, basic)
        
        # Reset accumulators
        self.episode_acc.reset_envs([e.item() for e in env_ids])
        self.step_tracer.on_episode_end([e.item() for e in env_ids])
    
    def get_final_evaluation(self, env_id: int, target_episodes: int) -> float:
        """Get final evaluation score for environment."""
        if not self.milestone_manager:
            return 50.0  # Default score
            
        return PerformanceEvaluator.get_final_evaluation(
            self.milestone_manager.milestone_performances,
            self.step_tracer.basic_stats,
            env_id,
            target_episodes,
            self.milestone_manager.milestones
        )
    
    def get_next_milestone_progress(self):
        """Get next milestone progress using external counter."""
        if (self.milestone_manager is None) or (self.external_episode_counter is None):
            return None, 0, 0
        return self.milestone_manager.get_next_milestone_progress(self.external_episode_counter)
    
    def close_all_files(self):
        """Close logging files (no-op for console logging)."""
        pass
    
    # Compatibility properties
    @property
    def episode_count(self):
        """Access external episode counter."""
        return self.external_episode_counter
    
    @property
    def env_episode_counts(self):
        """Access external episode counter."""
        return self.external_episode_counter
    
    @property
    def milestone_performances(self):
        """Access milestone performance data."""
        return self.milestone_manager.milestone_performances if self.milestone_manager else {}
    
    @property
    def milestones(self):
        """Access milestone list."""
        return self.milestone_manager.milestones if self.milestone_manager else []