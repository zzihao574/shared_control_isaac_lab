# SPDX-License-Identifier: BSD-3-Clause
"""
Environment utilities for Epigraph MARL training.

Provides:
- CompleteConstraintChecker: physics-based safety analysis
- TrajectoryManager: task progress tracking
- StepTracer: debug printing for four-zone rewards
- compose_task_safe_from_rc: reward decomposition into task/safety signals

ARCHITECTURE NOTE (Pure Epigraph: Self-Contained):
- Environment maintains its own step counter (_env_debug_step_counter)
- StepTracer is called by environment's own reward logic (_get_rewards)
- Printing controlled purely by YAML config (logging.enable_console_logging)
- No dependency on trainer or global_step from external sources
- Completely independent from RMAPPO or other algorithms
"""

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
            constraint_path = f"/World/envs/env_{env_id}/Constraint/Sphere"
            
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
    """
    Console debug printer for Epigraph four-zone reward monitoring.
    
    HOW IT'S USED IN EPIGRAPH (Pure Self-Contained Design):
    - The environment constructs a StepTracer in its __init__
    - The environment calls step_tracer.maybe_print_step(...) from inside its own
      reward logic (_get_rewards), but ONLY if logging.enable_console_logging == true
      in the YAML
    - No trainer/external dependency: the environment maintains its own local counter
      (env._env_debug_step_counter) for step numbers
    - This keeps Epigraph completely self-contained and independent
    
    DESIGN PRINCIPLES:
    - RMAPPO is NOT involved
    - Trainer does NOT need to inject step counts
    - Environment controls its own logging based purely on YAML config
    - Works identically in training and evaluation
    """
    
    def __init__(self, num_envs: int, device: torch.device,
                 enable_console_logging: bool = False,
                 print_every_steps: int = 10,
                 max_envs_to_print: int = 2,
                 strict_masks: bool = False):
        self.num_envs = num_envs
        self.device = device
        self.enable_console_logging = enable_console_logging
        self.print_every_steps = print_every_steps
        self.max_envs_to_print = max_envs_to_print
        self.strict_masks = strict_masks

    def maybe_print_step(
        self,
        env,
        r_task: torch.Tensor = None,        # [num_envs, num_agents] or Dict[agent, tensor]
        r_safe_cost: torch.Tensor = None,   # [num_envs, num_agents], >=0
        z: torch.Tensor = None,             # [num_envs, num_agents]
        global_step: int = 0,
        force_print: bool = False,
    ):
        """
        Print Epigraph decomposition (task/safe/z) for debugging.

        NOTE: Environment calls this from its own _get_rewards() method,
        using its own step counter (env._env_debug_step_counter).
        No external dependency on trainer or global training step.

        Args:
            env: Environment instance
            r_task: Task rewards (can be Dict[agent, tensor] or [num_envs, num_agents])
            r_safe_cost: Safety costs [num_envs, num_agents], non-negative (optional)
            z: Risk budget [num_envs, num_agents] (optional)
            global_step: Step counter from environment (env._env_debug_step_counter)
            force_print: Override print_every_steps frequency
        """
        if not self.enable_console_logging:
            return
            
        if not force_print and global_step % self.print_every_steps != 0:
            return

        agent_order = getattr(getattr(env, 'cfg', None), 'possible_agents', ['robot', 'human'])
        zone_masks = getattr(env, "_current_zone_masks", None)

        try:
            r_task_mat = self._agent_matrix(r_task, agent_order, label='r_task')
            r_safe_mat = self._agent_matrix(r_safe_cost, agent_order, label='r_safe_cost', default_zero=True)
            z_mat = None
            if z is not None:
                try:
                    z_mat = self._agent_matrix(z, agent_order, label='z', default_zero=False)
                except KeyError:
                    z_mat = None

            idx_robot = agent_order.index('robot') if 'robot' in agent_order else 0
            idx_human = agent_order.index('human') if 'human' in agent_order else min(1, r_task_mat.shape[1] - 1)

            to_show = list(range(min(self.max_envs_to_print, self.num_envs)))

            print('=' * 80)
            print(f"[STEP {global_step}] Four-Zone Safety Monitor (Epigraph)")
            print('=' * 80)
            print(f"Showing first {len(to_show)} of {self.num_envs} environments")

            print("\n[TASK/SAFE/Z SUMMARY]")
            for env_id in to_show:
                zone_label, _, _ = self._resolve_zone_label(env, env_id, zone_masks)
                task_robot = r_task_mat[env_id, idx_robot].item()
                task_human = r_task_mat[env_id, idx_human].item()
                safe_robot = r_safe_mat[env_id, idx_robot].item()
                safe_human = r_safe_mat[env_id, idx_human].item()
                progress = self._safe_get_reward_component(env, 'progress_ratio', env_id, 0.0)
                distance = self._safe_get_reward_component(env, 'distance_to_final', env_id, 0.0)
                if z_mat is not None:
                    z_value = float(z_mat[env_id].mean().item())
                else:
                    z_value = 0.0
                print(
                    f"  Env {env_id}: zone={zone_label} | "
                    f"task/robot {task_robot:+.3f} | task/human {task_human:+.3f} | "
                    f"safe_cost/robot {safe_robot:+.3f} | safe_cost/human {safe_human:+.3f} | "
                    f"z/mean {z_value:+.4f} | progress {progress:.1%} | dist_to_end {distance:.4f}"
                )

            print("\n[DETAILED SNAPSHOT]")
            detailed_agents: List[str] = []
            # Prefer explicit robot/human ordering when available
            if 'robot' in agent_order:
                detailed_agents.append('robot')
            if 'human' in agent_order:
                detailed_agents.append('human')
            # Append any remaining agents (while preserving cfg order)
            for agent_name in agent_order:
                if agent_name not in detailed_agents:
                    detailed_agents.append(agent_name)

            for env_id in to_show:
                self._print_env_snapshot(env, env_id, zone_masks)
                printed_agents = set()
                for agent_name in detailed_agents:
                    label = agent_name.upper()
                    if label in printed_agents:
                        continue
                    printed_agents.add(label)
                    self._print_agent_rewards(env, env_id, label)
        except Exception as exc:
            print('[STEPTRACER][ERROR] Exception during maybe_print_step:', repr(exc))
            print(f'  agent_order: {agent_order}')
            print(f"  r_task type: {type(r_task)}")
            if torch.is_tensor(r_task):
                print(f"  r_task shape: {tuple(r_task.shape)}")
            elif isinstance(r_task, dict):
                for key, val in r_task.items():
                    shape = tuple(val.shape) if torch.is_tensor(val) else type(val)
                    print(f"    r_task[{key}] -> {shape}")
            print(f"  r_safe_cost type: {type(r_safe_cost)}")
            if torch.is_tensor(r_safe_cost):
                print(f"  r_safe_cost shape: {tuple(r_safe_cost.shape)}")
            elif isinstance(r_safe_cost, dict):
                for key, val in r_safe_cost.items():
                    shape = tuple(val.shape) if torch.is_tensor(val) else type(val)
                    print(f"    r_safe_cost[{key}] -> {shape}")
            print(f"  z provided: {z is not None}")
            if z is not None and torch.is_tensor(z):
                print(f"  z shape: {tuple(z.shape)}")
            reward_keys = []
            if hasattr(env, 'reward_components'):
                reward_keys = list(env.reward_components.keys())
            print(f"  env.reward_components keys: {reward_keys}")
            print(f"  global_step: {global_step}")
            raise

    def _agent_matrix(self, data, agent_order, label, default_zero=False):
        num_agents = len(agent_order)
        if data is None:
            if default_zero:
                return torch.zeros(self.num_envs, num_agents, device=self.device, dtype=torch.float32)
            raise KeyError(f"{label} is None")

        if torch.is_tensor(data):
            if data.dim() == 2:
                if data.shape[0] != self.num_envs:
                    raise ValueError(f"{label} tensor has invalid shape {tuple(data.shape)}")
                if data.shape[1] == num_agents:
                    return data.to(self.device)
                if data.shape[1] == 1 and num_agents > 1:
                    return data.repeat(1, num_agents).to(self.device)
                raise ValueError(f"{label} tensor second dim mismatch {data.shape[1]} vs {num_agents}")
            if data.dim() == 1:
                base = data.view(self.num_envs, 1)
                base = base.to(self.device)
                return base.repeat(1, num_agents) if num_agents > 1 else base
            raise ValueError(f"{label} tensor unsupported dims {data.dim()}")

        if isinstance(data, dict):
            cols = []
            for agent in agent_order:
                value = data.get(agent)
                if value is None:
                    if default_zero:
                        cols.append(torch.zeros(self.num_envs, device=self.device, dtype=torch.float32))
                        continue
                    raise KeyError((agent, label))
                if not torch.is_tensor(value):
                    raise TypeError(f"{label}[{agent}] must be tensor, got {type(value)}")
                if value.dim() == 1:
                    cols.append(value.to(self.device))
                elif value.dim() == 2:
                    if value.shape[1] == 1:
                        cols.append(value[:, 0].to(self.device))
                    else:
                        raise ValueError(f"{label}[{agent}] tensor shape {tuple(value.shape)} unsupported")
                else:
                    cols.append(value.reshape(self.num_envs).to(self.device))
            stacked = torch.stack(cols, dim=1)
            return stacked

        raise TypeError(f"{label} has unsupported type {type(data)}")
    def _print_env_snapshot(self, env, env_id: int, zone_masks: Optional[Dict[str, torch.Tensor]] = None):
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

        zone_label, hits, fallback_used = self._resolve_zone_label(env, env_id, zone_masks)
        print(f"Active Zone: {zone_label}")
        if zone_masks is not None:
            status_entries = []
            for zone in ("A", "B", "C", "D"):
                mask_tensor = zone_masks.get(zone)
                flag = bool(mask_tensor[env_id].item()) if mask_tensor is not None else False
                status_entries.append(f"{zone}={flag}")
            mask_status = ", ".join(status_entries)
            print(f"Zone Masks: {mask_status}")
            if len(hits) > 1:
                print(f"[STEPTRACER][WARN] Multiple zone masks active for env {env_id}: {hits}")
            if fallback_used and not hits:
                print(f"[STEPTRACER][INFO] Fallback zone label used for env {env_id} (no mask active).")

    def _print_agent_rewards(self, env, env_id: int, agent_label: str):
        """Print detailed reward breakdown for specified agent."""
        agent_key = agent_label.lower()
        RC = env.reward_components

        def _val(x):
            if x is None:
                return 0.0
            if torch.is_tensor(x):
                if x.ndim == 0:
                    return float(x.item())
                if env_id < x.shape[0]:
                    return float(x[env_id].item())
                return 0.0
            return float(x)

        def _triplet(base: str) -> Tuple[float, float, float]:
            raw = _val(RC.get(f"{base}_raw"))
            weight = _val(RC.get(f"{base}_weight"))
            contrib = _val(RC.get(f"{base}_contrib"))
            return raw, weight, contrib

        def _safe_cost(value: float) -> float:
            return max(0.0, -value)

        # Zone A (task)
        a_prog = _triplet(f'zoneA_progress_{agent_key}')
        a_dev = _triplet(f'zoneA_deviation_{agent_key}')
        a_weight = _val(RC.get(f'zoneA_weight_{agent_key}'))
        a_total = _val(RC.get(f'zoneA_total_{agent_key}'))
        a_sum = a_prog[2] + a_dev[2]

        # Zone B task / safe
        b_gap = _triplet(f'zoneB_gap_{agent_key}')
        b_surf = _triplet(f'zoneB_surftangent_{agent_key}')
        b_inward = _triplet(f'zoneB_inward_{agent_key}')
        b_weight = _val(RC.get(f'zoneB_weight_{agent_key}'))
        b_task_raw = b_gap[2] + b_surf[2]
        b_safe_raw = b_inward[2]
        b_task_total = b_task_raw * b_weight
        b_safe_total = b_safe_raw * b_weight

        # Zone C safe components
        c_off = _triplet(f'zoneC_offpen_{agent_key}')
        c_inward = _triplet(f'zoneC_inward_{agent_key}')
        c_overlap = _triplet(f'zoneC_overlap_{agent_key}')
        c_weight = _val(RC.get(f'zoneC_weight_{agent_key}'))
        c_total = _val(RC.get(f'zoneC_total_{agent_key}'))
        c_raw_sum = c_off[2] + c_inward[2] + c_overlap[2]

        # Zone D task / safe
        d_prog = _triplet(f'zoneD_progress_{agent_key}')
        d_dev = _triplet(f'zoneD_deviation_{agent_key}')
        d_inward = _triplet(f'zoneD_inward_{agent_key}')
        d_weight = _val(RC.get(f'zoneD_weight_{agent_key}'))
        d_task_raw = d_prog[2] + d_dev[2]
        d_safe_raw = d_inward[2]
        d_task_total = d_task_raw * d_weight
        d_safe_total = d_safe_raw * d_weight

        # Global task / safe
        g_potential = _triplet(f'global_potential_{agent_key}')
        g_completion = _triplet(f'global_completion_{agent_key}')
        g_timeeff = _triplet(f'global_timeeff_{agent_key}')
        g_force = _triplet(f'{agent_key}force')
        aware_key = 'humanaware' if agent_key == 'robot' else 'robotaware'
        g_aware = _triplet(aware_key)
        g_task_sum = (
            g_potential[2] +
            g_completion[2] +
            g_timeeff[2] +
            g_force[2] +
            g_aware[2]
        )

        g_zpenalty = _triplet(f'global_zpenalty_{agent_key}')

        task_total = a_total + b_task_total + d_task_total + g_task_sum
        safe_total_risk = b_safe_total + c_total + d_safe_total + g_zpenalty[2]
        safe_total_cost = _safe_cost(safe_total_risk)

        print(f"\n[{agent_label}] TASK COMPONENTS:")
        self._print_task_zone(
            "A Track",
            a_sum,
            a_weight,
            a_total,
            [
                ("Progress", a_prog),
                ("Deviation", a_dev),
            ],
        )
        self._print_task_zone(
            "B Surface",
            b_task_raw,
            b_weight,
            b_task_total,
            [
                ("Gap", b_gap),
                ("Surf_Tangent", b_surf),
            ],
        )
        self._print_task_zone(
            "D Rejoin",
            d_task_raw,
            d_weight,
            d_task_total,
            [
                ("Progress", d_prog),
                ("Deviation", d_dev),
            ],
        )
        self._print_task_zone(
            "Global",
            g_task_sum,
            1.0,
            g_task_sum,
            [
                ("Potential", g_potential),
                ("Completion", g_completion),
                ("Time Efficiency", g_timeeff),
                (f"{agent_label} Force", g_force),
                ("Awareness", g_aware),
            ],
        )

        print(f"\n[{agent_label}] SAFE COMPONENTS (risk, cost=relu(-risk)):")
        self._print_safe_zone(
            "B Surface",
            b_safe_raw,
            b_weight,
            b_safe_total,
            [
                ("Inward Pen", b_inward),
            ],
        )
        c_mode = "Overlap" if float(env.safety_distances_t1[env_id].item()) <= 0.0 else "Outside"
        self._print_safe_zone(
            f"C Danger-{c_mode}",
            c_raw_sum,
            c_weight,
            c_total,
            [
                ("Off_Penalty", c_off),
                ("Inward_Pen", c_inward),
                ("Overlap_Pen", c_overlap),
            ],
        )
        self._print_safe_zone(
            "D Rejoin",
            d_safe_raw,
            d_weight,
            d_safe_total,
            [
                ("Inward_Pen", d_inward),
            ],
        )
        self._print_safe_zone(
            "Global",
            g_zpenalty[2],
            1.0,
            g_zpenalty[2],
            [
                ("Z Penalty", g_zpenalty),
            ],
        )

        print(f"\n  -> Task total: {task_total:+.3f}")
        print(f"  -> Safe risk total: {safe_total_risk:+.3f}")
        print(f"     Safe cost total (relu(-risk)): {safe_total_cost:+.3f}  [summary value]")

    def _print_task_zone(
        self,
        title: str,
        contrib_sum: float,
        zone_weight: float,
        zone_total: float,
        items: List[Tuple[str, Tuple[float, float, float]]],
    ) -> None:
        zone_contrib = contrib_sum
        total = zone_total if zone_weight != 0 else zone_contrib
        print(f"  {title}: {zone_contrib:+.3f} * {zone_weight:.2f} = {total:+.3f}")
        for name, (raw, weight, contrib) in items:
            print(f"    {name:<24}{raw:+.3f} * {weight:.2f} = {contrib:+.3f}")

    def _print_safe_zone(
        self,
        title: str,
        raw_sum: float,
        zone_weight: float,
        zone_total: float,
        items: List[Tuple[str, Tuple[float, float, float]]],
    ) -> None:
        cost_value = max(0.0, -zone_total)
        print(f"  {title}: {raw_sum:+.3f} * {zone_weight:.2f} = {zone_total:+.3f} (cost={cost_value:+.3f})")
        for name, (raw, weight, contrib) in items:
            cost = max(0.0, -contrib)
            print(f"    {name:<24}{raw:+.3f} * {weight:.2f} = {contrib:+.3f} (cost={cost:+.3f})")

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

    def _resolve_zone_label(
        self,
        env,
        env_id: int,
        zone_masks: Optional[Dict[str, torch.Tensor]]
    ) -> Tuple[str, List[str], bool]:
        """Resolve zone label using training masks; fall back to safety heuristic."""
        hits: List[str] = []
        label: Optional[str] = None
        fallback_used = False

        if zone_masks is None:
            if self.strict_masks:
                raise RuntimeError(
                    "[STEPTRACER] strict_masks=True but env._current_zone_masks is missing."
                )
            fallback_used = True
            return self._get_zone_label_fallback(env, env_id), hits, fallback_used

        try:
            for zone in ("A", "B", "C", "D"):
                mask_tensor = zone_masks.get(zone)
                if mask_tensor is not None and bool(mask_tensor[env_id].item()):
                    hits.append(zone)
        except Exception as exc:
            if self.strict_masks:
                raise RuntimeError(
                    f"[STEPTRACER] strict_masks=True but failed to inspect zone masks for env {env_id}: {exc}"
                ) from exc
            print(f"[STEPTRACER][WARN] Failed to inspect zone masks for env {env_id}: {exc}")
            hits = []

        if hits:
            label = self._format_zone_label_from_masks(env, env_id, hits)
            return label, hits, fallback_used

        if self.strict_masks:
            raise RuntimeError(
                f"[STEPTRACER] strict_masks=True but no active zone mask detected for env {env_id}."
            )

        fallback_used = True
        label = self._get_zone_label_fallback(env, env_id)
        return label, hits, fallback_used

    def _format_zone_label_from_masks(self, env, env_id: int, hits: List[str]) -> str:
        """Format zone label string from training masks."""
        labels = []
        for zone in hits:
            if zone == "A":
                labels.append("A (Track)")
            elif zone == "B":
                labels.append("B (Surface)")
            elif zone == "C":
                safety = float(env.safety_distances_t1[env_id].item())
                is_colliding = bool(env.is_violating_t1[env_id].item())
                mode = "Danger-Overlap" if safety <= 0.0 and is_colliding else "Danger"
                labels.append(f"C ({mode})")
            elif zone == "D":
                labels.append("D (Rejoin)")
        return " / ".join(labels) if labels else "Unknown"

    def _get_zone_label_fallback(self, env, env_id: int) -> str:
        """Fallback zone label based on safety distance thresholds."""
        safety = float(env.safety_distances_t1[env_id].item())
        is_colliding = bool(env.is_violating_t1[env_id].item())
        rejoin = env.rejoin_streak[env_id] >= 10
        D, O = 0.0075, 0.015

        if safety >= O:
            return "A (Track)"
        if safety <= D:
            return "C (Danger-Overlap)" if is_colliding else "C (Danger)"
        return "D (Rejoin)" if rejoin else "B (Surface)"


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
    Decompose reward_components into task reward and safety risk for Epigraph.
    
    Returns:
        r_task: Task performance reward (for Vl critic)
        r_safe_risk: Safety risk signal (for Vh critic)
    
    SIGN CONVENTION:
        r_safe_risk is NEGATIVE or zero in danger zones (penalties).
        Trainer/env converts to positive cost via: cost = relu(-r_safe_risk)
        Higher cost = more dangerous.
    
    Task components:
        - Zone A/B/D (progress, deviation, gap, surface tangent)
        - Global: potential, completion, (optional) time efficiency
        - Force penalties (negative comfort penalties)
        - Awareness (humanaware/robotaware)
        
    Safety components:
        - Zone C danger penalties (offpen, inward, overlap)
        - B/D inward penalties
        - (optional) Global z-penalty
    
    Args:
        rc: reward_components from environment
        agent: "human" or "robot"
        device: torch device
        num_envs: number of environments
        use_time_eff_in_task: include time efficiency in task
        include_zpenalty_in_safe: include z penalty in safety
    """
    z = lambda k: safe_get_rc(rc, k, device, num_envs)

    aware_key = "humanaware" if agent == "robot" else "robotaware"

    # Task quality reward (for Vl critic)
    zoneA_total = z(f"zoneA_total_{agent}")

    zoneB_weight = z(f"zoneB_weight_{agent}")
    zoneB_gap = z(f"zoneB_gap_{agent}_contrib")
    zoneB_surf = z(f"zoneB_surftangent_{agent}_contrib")
    zoneB_task = zoneB_weight * (zoneB_gap + zoneB_surf)

    zoneD_weight = z(f"zoneD_weight_{agent}")
    zoneD_prog = z(f"zoneD_progress_{agent}_contrib")
    zoneD_dev = z(f"zoneD_deviation_{agent}_contrib")
    zoneD_task = zoneD_weight * (zoneD_prog + zoneD_dev)

    r_task = (
        zoneA_total
        + zoneB_task
        + zoneD_task
        + z(f"global_potential_{agent}_contrib")
        + z(f"global_completion_{agent}_contrib")
        + (z(f"global_timeeff_{agent}_contrib") if use_time_eff_in_task else _zeros(device, num_envs))
        + z(f"{agent}force_contrib")
        + z(f"{aware_key}_contrib")
    )

    # Safety risk (NEGATIVE when dangerous, for Vh critic)
    # Trainer converts to positive cost: cost = relu(-r_safe_risk)
    zoneB_inward = z(f"zoneB_inward_{agent}_contrib")
    zoneB_safe = zoneB_weight * zoneB_inward

    zoneC_total = z(f"zoneC_total_{agent}")

    zoneD_inward = z(f"zoneD_inward_{agent}_contrib")
    zoneD_safe = zoneD_weight * zoneD_inward

    r_safe_risk = (
        zoneC_total
        + zoneB_safe
        + zoneD_safe
        + (z(f"global_zpenalty_{agent}_contrib") if include_zpenalty_in_safe else _zeros(device, num_envs))
    )
    
    return r_task, r_safe_risk
