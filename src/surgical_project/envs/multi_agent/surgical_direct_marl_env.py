# surgical_direct_marl_env.py - 完整修改版（改进障碍物跨越）

from __future__ import annotations

import torch
import numpy as np
import yaml
import os
import gymnasium as gym
from typing import Any, Dict, List, Optional

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.math import sample_uniform, quat_rotate_inverse

from .surgical_direct_marl_env_cfg import SurgicalDirectMARLEnvCfg


class CompleteConstraintChecker:
    """约束状态检测类"""
    
    def __init__(self, device, collision_threshold=0.001):
        self.device = device
        self.collision_threshold = collision_threshold
        
        try:
            from omni.physx.bindings._physx import acquire_physx_attachment_interface, acquire_physx_scene_query_interface
            self.physics_attachment_interface = acquire_physx_attachment_interface()
            self.physics_scene_query_interface = acquire_physx_scene_query_interface()
            print("[INFO] Physics query interfaces available")
        except ImportError:
            print("[ERROR] Physics query interfaces not available")
            self.physics_attachment_interface = None
            self.physics_scene_query_interface = None
    
    def analyze_constraint_state_batch(self, stylus_positions: torch.Tensor, env_base_positions: torch.Tensor):
        """批量分析约束状态"""
        num_envs = stylus_positions.shape[0]
        
        current_base_positions = self._omni_robot.data.root_link_pos_w if hasattr(self, '_omni_robot') else env_base_positions
        
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
            constraint_path = f"/World/envs/env_{env_id}/Constraint/mesh"
            
            try:
                result = self._analyze_single_constraint(stylus_world_pos, constraint_path)
                if result is not None and result['distance'] > 0:
                    batch_results['distances_constraint'][env_id] = result['distance']
                    batch_results['closest_points'][env_id] = torch.tensor(result['closest_point'], device=self.device)
                    batch_results['normal_vectors'][env_id] = torch.tensor(result['normal_vector'], device=self.device)
                    batch_results['is_overlapping'][env_id] = result['is_overlapping']
                    batch_results['is_inside'][env_id] = result['is_inside']
            except Exception as e:
                pass
        
        return batch_results
    
    def _analyze_single_constraint(self, stylus_position: torch.Tensor, constraint_path: str):
        """分析单个约束状态"""
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
            
            direction_to_closest = Float3(closest_pt.x - pos[0], closest_pt.y - pos[1], closest_pt.z - pos[2])
            raycast_result = self.physics_scene_query_interface.raycast_closest(current_point, direction_to_closest, 10000)
            
            is_overlapping = False
            if (('faceIndex' not in raycast_result) and distance < 0.5) or \
               ('faceIndex' in raycast_result and raycast_result.get('faceIndex', -1) == 0 and distance < 0.5):
                is_overlapping = True
            
            direction_is_inside_1 = Float3(closest_pt.x - pos[0], closest_pt.y - pos[1], 0)
            direction_is_inside_2 = Float3(-closest_pt.x + pos[0], -closest_pt.y + pos[1], 0)
            
            len1 = np.sqrt((closest_pt.x - pos[0])**2 + (closest_pt.y - pos[1])**2)
            len2 = np.sqrt((-closest_pt.x + pos[0])**2 + (-closest_pt.y + pos[1])**2)
            
            if len1 > 1e-8:
                direction_is_inside_1 = Float3((closest_pt.x - pos[0])/len1, (closest_pt.y - pos[1])/len1, 0)
            else:
                direction_is_inside_1 = Float3(1.0, 0.0, 0.0)
                
            if len2 > 1e-8:
                direction_is_inside_2 = Float3((-closest_pt.x + pos[0])/len2, (-closest_pt.y + pos[1])/len2, 0)
            else:
                direction_is_inside_2 = Float3(-1.0, 0.0, 0.0)
            
            is_inside_1 = self.physics_scene_query_interface.raycast_any(current_point, direction_is_inside_1, 10000)
            is_inside_2 = self.physics_scene_query_interface.raycast_any(current_point, direction_is_inside_2, 10000)
            is_inside = bool(is_inside_1) and bool(is_inside_2)
            
            normal_vector = np.array([1.0, 0.0, 0.0])
            
            if is_overlapping:
                mirror_point_pos = 2 * closest_pos - pos
                mirror_point = Float3(float(mirror_point_pos[0]), float(mirror_point_pos[1]), float(mirror_point_pos[2]))
                direction_mirror_to_current = Float3(pos[0] - mirror_point_pos[0], pos[1] - mirror_point_pos[1], pos[2] - mirror_point_pos[2])
                mirror_raycast = self.physics_scene_query_interface.raycast_closest(mirror_point, direction_mirror_to_current, 10000)
                
                if mirror_raycast and 'normal' in mirror_raycast:
                    normal_carb = mirror_raycast['normal']
                    if is_inside:
                        normal_vector = np.array([normal_carb.x, normal_carb.y, normal_carb.z])
                    else:
                        normal_vector = -np.array([normal_carb.x, normal_carb.y, normal_carb.z])
            else:
                if raycast_result and 'normal' in raycast_result:
                    normal_carb = raycast_result['normal']
                    if is_inside:
                        normal_vector = np.array([normal_carb.x, normal_carb.y, normal_carb.z])
                    else:
                        normal_vector = -np.array([normal_carb.x, normal_carb.y, normal_carb.z])
            
            return {
                'distance': distance,
                'closest_point': closest_pos,
                'normal_vector': normal_vector,
                'is_overlapping': is_overlapping,
                'is_inside': is_inside
            }
                
        except Exception as e:
            return None


class TrajectoryManager:
    """轨迹管理器 - 简化版，用于进度跟踪"""
    
    def __init__(self, device: torch.device, params: dict, num_envs: int, env_base_positions: torch.Tensor):
        self.device = device
        self.num_envs = num_envs
        self.env_base_positions = env_base_positions
        
        traj = params['trajectory']
        self.start_pos_local = torch.tensor(traj['start_point'], device=device, dtype=torch.float32)
        self.end_pos_local = torch.tensor(traj['end_point'], device=device, dtype=torch.float32)
        self.total_distance = torch.norm(self.end_pos_local - self.start_pos_local).item()
        
        # 直线方向单位向量
        self.line_direction = (self.end_pos_local - self.start_pos_local) / self.total_distance
        
        # 10cm设置一个checkpoint，1cm切换
        self.checkpoint_interval = traj.get('checkpoint_interval', 0.1)  # 10cm
        self.switch_threshold = traj.get('switch_threshold', 0.01)      # 1cm
        
        self._generate_setpoints(params)
        self.current_setpoint_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.setpoints_tensor = torch.stack(self.setpoints_local)
        self.num_setpoints = len(self.setpoints_local)
        
    def _generate_setpoints(self, params):
        """生成轨迹设置点"""
        num_checkpoints = int(self.total_distance / self.checkpoint_interval)
        self.setpoints_local = []
        
        for i in range(num_checkpoints + 1):
            if i == num_checkpoints:
                checkpoint = self.end_pos_local.clone()
            else:
                checkpoint = self.start_pos_local + self.line_direction * (i * self.checkpoint_interval)
            self.setpoints_local.append(checkpoint)
        
        print(f"[TRAJECTORY] 生成了 {len(self.setpoints_local)} 个轨迹点（10cm间隔）")
    
    def get_current_setpoint_local(self) -> torch.Tensor:
        """获取当前设置点"""
        indices = torch.clamp(self.current_setpoint_idx, 0, self.num_setpoints - 1)
        return self.setpoints_tensor[indices]
    
    def update_setpoint(self, current_pos_local: torch.Tensor) -> torch.Tensor:
        """更新设置点"""
        current_indices = self.current_setpoint_idx
        indices_clamped = torch.clamp(current_indices, 0, self.num_setpoints - 1)
        current_setpoints = self.setpoints_tensor[indices_clamped]
        
        distances = torch.norm(current_pos_local - current_setpoints, dim=-1)
        should_update = (distances < self.switch_threshold) & (current_indices < self.num_setpoints - 1)
        
        self.current_setpoint_idx[should_update] += 1
        return should_update
    
    def get_progress(self, current_pos_local: torch.Tensor) -> torch.Tensor:
        """计算沿轨迹的进度（0到1）"""
        vec_to_current = current_pos_local - self.start_pos_local.unsqueeze(0)
        progress_distance = torch.sum(vec_to_current * self.line_direction.unsqueeze(0), dim=-1)
        progress_distance = torch.clamp(progress_distance, 0, self.total_distance)
        return progress_distance / self.total_distance
    
    def get_deviation(self, current_pos_local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """计算到直线的垂直距离和最近点"""
        vec_to_current = current_pos_local - self.start_pos_local.unsqueeze(0)
        progress_distance = torch.sum(vec_to_current * self.line_direction.unsqueeze(0), dim=-1)
        progress_distance = torch.clamp(progress_distance, 0, self.total_distance)
        
        # 最近点（投影点）
        closest_points = self.start_pos_local.unsqueeze(0) + progress_distance.unsqueeze(-1) * self.line_direction.unsqueeze(0)
        
        # 垂直距离
        deviations = torch.norm(current_pos_local - closest_points, dim=-1)
        
        return deviations, closest_points
    
    def is_final_setpoint_reached(self, current_pos_local: torch.Tensor) -> torch.Tensor:
        """检查是否到达终点"""
        distances_to_final = torch.norm(current_pos_local - self.end_pos_local.unsqueeze(0), dim=-1)
        return distances_to_final < self.switch_threshold
    
    def reset_trajectory(self, env_ids: torch.Tensor = None):
        """重置轨迹"""
        if env_ids is None:
            self.current_setpoint_idx.fill_(0)
        else:
            self.current_setpoint_idx[env_ids] = 0
    
    def get_trajectory_info(self) -> Dict:
        """获取轨迹信息"""
        return {
            'start_point': self.start_pos_local.cpu().numpy(),
            'end_point': self.end_pos_local.cpu().numpy(),
            'total_distance': self.total_distance,
            'num_checkpoints': self.num_setpoints,
            'checkpoint_interval': self.checkpoint_interval
        }


class RewardLogger:
    """奖励记录器 - 简化版"""
    
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.target_episodes = [1, 50, 100, 150, 200]
        
        self.env_episode_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.env_step_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.env_log_files = {}
        
        self.log_dir = "logs/env_details"
        os.makedirs(self.log_dir, exist_ok=True)
        
        import datetime
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def should_log_episode(self, env_id):
        """判断是否应该记录该环境的当前episode"""
        episode_num = self.env_episode_counts[env_id].item()
        return episode_num in self.target_episodes
    
    def get_or_create_log_file(self, env_id):
        """获取或创建环境的日志文件"""
        if env_id not in self.env_log_files:
            log_file_path = os.path.join(self.log_dir, f"env_{env_id}_{self.timestamp}.txt")
            try:
                self.env_log_files[env_id] = open(log_file_path, 'w')
                self.env_log_files[env_id].write(f"ENV {env_id} REWARD DETAILS\n")
                self.env_log_files[env_id].write("="*80 + "\n")
                self.env_log_files[env_id].write("记录Episode: 1, 50, 100, 150, 200\n")
                self.env_log_files[env_id].write("="*80 + "\n\n")
            except Exception as e:
                print(f"[WARNING] 无法创建环境{env_id}的日志文件: {e}")
                self.env_log_files[env_id] = None
        
        return self.env_log_files[env_id]
    
    def log_step_details(self, step_count, reward_components, robot_weights, human_weights, final_rewards):
        """记录步骤详细信息"""
        for env_id in range(self.num_envs):
            if not self.should_log_episode(env_id):
                continue
                
            log_file = self.get_or_create_log_file(env_id)
            if log_file is None:
                continue
            
            try:
                episode_num = self.env_episode_counts[env_id].item()
                env_step = self.env_step_counts[env_id].item()
                
                log_file.write(f"\n[Episode {episode_num} - Step {env_step}]\n")
                log_file.write("-" * 50 + "\n")
                
                # 总奖励
                robot_total = final_rewards["robot"][env_id].item()
                human_total = final_rewards["human"][env_id].item()
                
                # Robot奖励详情
                log_file.write(f"ROBOT (总奖励: {robot_total:.4f}):\n")
                log_file.write(f"  trajectory_reward: {reward_components['trajectory_reward'][env_id]:.4f} * {robot_weights['trajectory_tracking']}\n")
                log_file.write(f"  progress_reward: {reward_components['progress_reward'][env_id]:.4f} * {robot_weights['progress']}\n")
                log_file.write(f"  velocity_reward: {reward_components['velocity_reward'][env_id]:.4f} * {robot_weights['velocity']}\n")
                log_file.write(f"  cbf_reward: {reward_components['cbf_reward'][env_id]:.4f} * {robot_weights['obstacle_cbf']}\n")
                log_file.write(f"  crossing_reward: {reward_components['crossing_reward'][env_id]:.4f} * {robot_weights['crossing']}\n")
                log_file.write(f"  robot_force_penalty: {reward_components['robot_force_penalty'][env_id]:.4f} * {robot_weights['force_efficiency']}\n")
                log_file.write(f"  human_force_awareness: {reward_components['human_force_penalty'][env_id]:.4f} * {robot_weights['human_awareness']}\n")
                
                # Human奖励详情
                log_file.write(f"HUMAN (总奖励: {human_total:.4f}):\n")
                log_file.write(f"  trajectory_reward: {reward_components['trajectory_reward'][env_id]:.4f} * {human_weights['trajectory_tracking']}\n")
                log_file.write(f"  progress_reward: {reward_components['progress_reward'][env_id]:.4f} * {human_weights['progress']}\n")
                log_file.write(f"  velocity_reward: {reward_components['velocity_reward'][env_id]:.4f} * {human_weights['velocity']}\n")
                log_file.write(f"  cbf_reward: {reward_components['cbf_reward'][env_id]:.4f} * {human_weights['obstacle_cbf']}\n")
                log_file.write(f"  crossing_reward: {reward_components['crossing_reward'][env_id]:.4f} * {human_weights['crossing']}\n")
                log_file.write(f"  human_force_penalty: {reward_components['human_force_penalty'][env_id]:.4f} * {human_weights['force_efficiency']}\n")
                log_file.write(f"  robot_force_awareness: {reward_components['robot_force_penalty'][env_id]:.4f} * {human_weights['robot_awareness']}\n")
                
                # 附加信息
                log_file.write(f"EXTRA:\n")
                log_file.write(f"  deviation: {reward_components['deviation'][env_id]:.4f}m\n")
                log_file.write(f"  progress: {reward_components['progress_ratio'][env_id]:.2%}\n")
                log_file.write(f"  completion_reward: {reward_components['completion_reward'][env_id]:.4f}\n")
                
                log_file.flush()
                
            except Exception as e:
                print(f"[WARNING] 记录环境{env_id}详情失败: {e}")
    
    def log_termination(self, env_id, reasons):
        """记录终止原因"""
        if not self.should_log_episode(env_id):
            return
            
        log_file = self.get_or_create_log_file(env_id)
        if log_file is None:
            return
        
        try:
            episode_num = self.env_episode_counts[env_id].item()
            log_file.write(f"\n[TERMINATION] Episode {episode_num}: {', '.join(reasons)}\n")
            log_file.write("="*80 + "\n\n")
            log_file.flush()
        except Exception as e:
            print(f"[WARNING] 记录环境{env_id}终止原因失败: {e}")
    
    def on_episode_end(self, env_ids):
        """Episode结束时更新计数"""
        self.env_episode_counts[env_ids] += 1
        self.env_step_counts[env_ids] = 0
    
    def on_step(self, env_ids=None):
        """每步更新步数计数"""
        if env_ids is None:
            self.env_step_counts += 1
        else:
            self.env_step_counts[env_ids] += 1
    
    def close_all_files(self):
        """关闭所有文件"""
        for env_id, log_file in self.env_log_files.items():
            if log_file:
                try:
                    log_file.close()
                except:
                    pass


class SurgicalDirectMARLEnv(DirectMARLEnv):
    """人机协作手术MARL环境"""
    
    cfg: SurgicalDirectMARLEnvCfg
    
    def __init__(self, cfg: SurgicalDirectMARLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.params = self._load_training_params()
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self._load_yaml_parameters()
        
        self.env_base_positions = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.trajectory_manager = TrajectoryManager(
            device=self.device,
            params=self.params,
            num_envs=self.num_envs,
            env_base_positions=self.env_base_positions
        )
        
        self.reward_logger = RewardLogger(self.num_envs, self.device)
        
        self.agent_actions = {
            agent: torch.zeros(self.num_envs, 3, device=self.device)
            for agent in self.cfg.possible_agents
        }
        
        # 状态缓存
        self.stylus_pos_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.stylus_vel_t1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.joint_pos_t1 = torch.zeros(self.num_envs, 6, device=self.device)
        self.joint_vel_t1 = torch.zeros(self.num_envs, 6, device=self.device)
        
        self.human_forces_t = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_forces_t = torch.zeros(self.num_envs, 3, device=self.device)
        self.safety_distances_t = torch.ones(self.num_envs, device=self.device) * 0.01
        self.is_violating_t = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        self.fixed_end_joints = torch.tensor([
            self.params['initial_conditions']['joint_positions']['yaw'],
            self.params['initial_conditions']['joint_positions']['pitch'],
            self.params['initial_conditions']['joint_positions']['roll']
        ], device=self.device, dtype=torch.float32)
        
        self.stylus_body_idx = None
        self.constraint_checker = CompleteConstraintChecker(self.device, self.collision_threshold)
        
        self.step_count = 0
        
        self.last_constraint_results = None
        self.reward_components = {}
        
        # Gymnasium兼容性
        self.action_space = gym.spaces.Dict({
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.action_spaces[agent],), dtype=np.float32)
            for agent in self.cfg.possible_agents
        })
        
        self.observation_space = gym.spaces.Dict({
            agent: gym.spaces.Box(low=-10.0, high=10.0, shape=(self.cfg.observation_spaces[agent],), dtype=np.float32)
            for agent in self.cfg.possible_agents
        })
        
    def _load_training_params(self) -> dict:
        """加载训练参数"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "agents", "training_params.yaml")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_yaml_parameters(self):
        """加载YAML参数"""
        constraints = self.params['constraints']
        self.min_z_pos = constraints['min_z_position']
        self.max_robot_force = constraints['max_robot_force']
        self.max_human_force = constraints['max_human_force']
        
        joint_limits = constraints['joint_limits']
        self.joint_lower_limits = torch.tensor([
            joint_limits['waist'][0], joint_limits['shoulder'][0], joint_limits['elbow'][0],
            joint_limits['yaw'][0], joint_limits['pitch'][0], joint_limits['roll'][0]
        ], device=self.device, dtype=torch.float32)
        
        self.joint_upper_limits = torch.tensor([
            joint_limits['waist'][1], joint_limits['shoulder'][1], joint_limits['elbow'][1],
            joint_limits['yaw'][1], joint_limits['pitch'][1], joint_limits['roll'][1]
        ], device=self.device, dtype=torch.float32)
        
        self.safety_margin = self.params['reward_parameters']['cbf_parameters']['safety_margin']
        self.constraint_center = torch.tensor(
            self.params['constraint_geometry']['center'], 
            device=self.device, dtype=torch.float32
        )
        self.collision_threshold = self.params['constraint_geometry']['collision_threshold']
        self.cbf_gamma = self.params['reward_parameters']['cbf_parameters']['gamma']
        self.cbf_epsilon = self.params['reward_parameters']['cbf_parameters']['epsilon']
        
        term_config = self.params.get('termination_conditions', {})
        self.enable_z_termination = term_config.get('z_below_zero', True)
        self.enable_edge_termination = term_config.get('edge_collision', True)
        self.safety_distance_threshold = term_config.get('safety_distance_threshold', 0.001)  # 修改为0.001
        
        print(f"[INFO] Episode长度由cfg.episode_length_s控制: {self.cfg.episode_length_s}s")
        
    def _setup_scene(self):
        """设置场景"""
        self._omni_robot = Articulation(self.cfg.phantom_omni)
        self.scene.articulations["phantom_omni"] = self._omni_robot
        
        self._constraint = RigidObject(self.cfg.constraint)
        self.scene.rigid_objects["constraint"] = self._constraint
        
        self.scene.clone_environments(copy_from_source=False)
        
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    def _setup_post_scene_creation(self):
        """场景创建后设置"""
        super()._setup_post_scene_creation()
        self._initialize_body_indices()
        
        if hasattr(self, '_omni_robot'):
            self.env_base_positions = self._omni_robot.data.root_link_pos_w.clone()
            self.trajectory_manager.env_base_positions = self.env_base_positions
            
            num_joints = self._omni_robot.num_joints
            zero_stiffness = torch.zeros(self.num_envs, num_joints, device=self.device)
            zero_damping = torch.zeros(self.num_envs, num_joints, device=self.device)
            
            self._omni_robot.write_joint_stiffness_to_sim(zero_stiffness)
            self._omni_robot.write_joint_damping_to_sim(zero_damping)
            
    def _initialize_body_indices(self):
        """初始化body索引"""
        if not hasattr(self._omni_robot, 'body_names'):
            return
        
        search_patterns = ['stylus', 'tip', 'end_effector', 'link6', 'end', 'tool']
        for pattern in search_patterns:
            for i, name in enumerate(self._omni_robot.body_names):
                if pattern in name.lower():
                    self.stylus_body_idx = i
                    return
        
        if len(self._omni_robot.body_names) > 0:
            self.stylus_body_idx = len(self._omni_robot.body_names) - 1
        
    def _pre_physics_step(self, actions: Dict[str, torch.Tensor]) -> None:
        """物理步骤前处理"""
        for agent, action in actions.items():
            if agent in self.cfg.possible_agents:
                if action.dim() == 1:
                    if action.shape[0] == 3:
                        action = action.unsqueeze(0).expand(self.num_envs, -1)
                    else:
                        action = action.unsqueeze(-1).expand(-1, 3)
                
                max_force = self.max_robot_force if agent == "robot" else self.max_human_force
                self.agent_actions[agent] = torch.clamp(action, -max_force, max_force)
        
        self.robot_forces_t = self.agent_actions["robot"]
        self.human_forces_t = self.agent_actions["human"]
        
        # 应用外力
        if self.stylus_body_idx is not None:
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
        
        # 固定末端关节
        joint_pos = self._omni_robot.data.joint_pos.clone()
        joint_vel = self._omni_robot.data.joint_vel.clone()
        
        joint_pos = torch.clamp(joint_pos, self.joint_lower_limits, self.joint_upper_limits)
        joint_pos[:, 3:6] = self.fixed_end_joints.unsqueeze(0).expand(self.num_envs, -1)
        joint_vel[:, 3:6] = 0.0
        
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel)
        
    def _apply_action(self) -> None:
        """应用动作并更新状态缓存"""
        self._omni_robot.write_data_to_sim()
        
        self.step_count += 1
        self.reward_logger.on_step()
        
        # 更新状态缓存
        self.stylus_pos_t1 = self._get_stylus_position()
        self.stylus_vel_t1 = self._get_stylus_velocity()
        
        joint_pos = self._omni_robot.data.joint_pos
        joint_vel = self._omni_robot.data.joint_vel
        
        if joint_pos.shape[-1] < 6:
            padding = torch.zeros(self.num_envs, 6 - joint_pos.shape[-1], device=self.device)
            self.joint_pos_t1 = torch.cat([joint_pos, padding], dim=-1)
            self.joint_vel_t1 = torch.cat([joint_vel, padding], dim=-1)
        else:
            self.joint_pos_t1 = joint_pos[..., :6]
            self.joint_vel_t1 = joint_vel[..., :6]
        
        # 更新约束状态
        current_base_positions = self._omni_robot.data.root_link_pos_w
        self.last_constraint_results = self.constraint_checker.analyze_constraint_state_batch(
            self.stylus_pos_t1, current_base_positions
        )
        self.safety_distances_t = self.last_constraint_results['distances_constraint']
        self.is_violating_t = self.last_constraint_results['is_overlapping']

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """获取观测"""
        stylus_vel_constrained = self.stylus_vel_t1
        
        constraint_distances = torch.stack([
            torch.norm(self.stylus_pos_t1 - self.constraint_center.unsqueeze(0).expand(self.num_envs, -1), dim=-1),
            self.safety_distances_t,
            torch.abs(self.stylus_pos_t1[:, 2] - self.constraint_center[2])
        ], dim=-1)
        
        obs = torch.cat([
            self.stylus_pos_t1,
            stylus_vel_constrained,
            self.joint_pos_t1,
            self.joint_vel_t1,
            constraint_distances,
        ], dim=-1)
        
        observations = {}
        for agent in self.cfg.possible_agents:
            observations[agent] = torch.clamp(obs, -20.0, 20.0)
            
        return observations
        
    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        """计算奖励 - 改进的障碍物跨越系统"""
        # 1. 轨迹偏差奖励
        deviations, closest_points = self.trajectory_manager.get_deviation(self.stylus_pos_t1)
        
        trajectory_reward = torch.where(
            deviations < 0.01,
            torch.ones_like(deviations),
            torch.where(
                deviations < 0.025,
                1.0 - (deviations - 0.01) / 0.015,
                -10.0 * (deviations - 0.025)
            )
        )
        
        # 2. 进度奖励
        progress_ratio = self.trajectory_manager.get_progress(self.stylus_pos_t1)
        progress_reward = progress_ratio * 5.0
        
        # 3. 速度奖励
        velocity_along_line = torch.abs(
            torch.sum(self.stylus_vel_t1 * self.trajectory_manager.line_direction.unsqueeze(0), dim=-1)
        )
        target_velocity = 0.025  # 2.5cm/s
        velocity_error = torch.abs(velocity_along_line - target_velocity)
        velocity_reward = torch.exp(-velocity_error * 20.0)
        
        # 4. 改进的CBF障碍物奖励
        obstacle_y_pos = 0.0  # 障碍物在y=0位置
        crossing_zone = torch.abs(self.stylus_pos_t1[:, 1] - obstacle_y_pos) < 0.05  # 5cm跨越区
        
        cbf_reward = torch.where(
            crossing_zone,  # 在跨越区域
            torch.where(
                self.safety_distances_t < 0.001,  # 非常近，强惩罚
                torch.full_like(self.safety_distances_t, -30.0),
                torch.where(
                    self.safety_distances_t < 0.01,  # 0.1-1cm，中等惩罚
                    torch.full_like(self.safety_distances_t, -5.0),
                    torch.full_like(self.safety_distances_t, 0.2)  # >1cm，小奖励
                )
            ),
            # 不在跨越区域，正常CBF
            torch.where(
                self.safety_distances_t < 0.01,
                -10.0 * (0.01 - self.safety_distances_t) / 0.01,
                torch.full_like(self.safety_distances_t, 0.1)
            )
        )
        
        # 5. 添加跨越进度奖励
        near_obstacle = torch.abs(self.stylus_pos_t1[:, 1] - obstacle_y_pos) < 0.1  # 10cm范围
        moving_forward = self.stylus_vel_t1[:, 1] > 0.01  # y方向正速度
        safe_distance = self.safety_distances_t > 0.001  # 保持安全距离
        
        crossing_reward = torch.where(
            near_obstacle & moving_forward & safe_distance,
            torch.full_like(progress_ratio, 2.0),  # 鼓励跨越
            torch.zeros_like(progress_ratio)
        )
        
        # 6. 力惩罚（二次型）
        robot_force_penalty = -50.0 * torch.sum(self.robot_forces_t**2, dim=-1)
        human_force_penalty = -50.0 * torch.sum(self.human_forces_t**2, dim=-1)
        
        # 7. 完成奖励
        distance_to_final = torch.norm(
            self.stylus_pos_t1 - self.trajectory_manager.end_pos_local.unsqueeze(0), 
            dim=-1
        )
        completion_reward = torch.where(
            distance_to_final < 0.01,
            torch.full_like(distance_to_final, 50.0),
            torch.zeros_like(distance_to_final)
        )
        
        # 更新检查点（仅用于进度跟踪）
        self.trajectory_manager.update_setpoint(self.stylus_pos_t1)
        
        # 获取权重
        robot_weights = self.params['reward_parameters']['robot_weights']
        human_weights = self.params['reward_parameters']['human_weights']
        
        # 计算最终奖励
        rewards = {}
        rewards["robot"] = (
            trajectory_reward * robot_weights['trajectory_tracking'] +
            progress_reward * robot_weights['progress'] +
            velocity_reward * robot_weights['velocity'] +
            cbf_reward * robot_weights['obstacle_cbf'] +
            crossing_reward * robot_weights.get('crossing', 1.0) +
            robot_force_penalty * robot_weights['force_efficiency'] +
            human_force_penalty * robot_weights['human_awareness'] +
            completion_reward
        )
        
        rewards["human"] = (
            trajectory_reward * human_weights['trajectory_tracking'] +
            progress_reward * human_weights['progress'] +
            velocity_reward * human_weights['velocity'] +
            cbf_reward * human_weights['obstacle_cbf'] +
            crossing_reward * human_weights.get('crossing', 0.8) +
            human_force_penalty * human_weights['force_efficiency'] +
            robot_force_penalty * human_weights['robot_awareness'] +
            completion_reward
        )
        
        # 存储组件
        self.reward_components = {
            'trajectory_reward': trajectory_reward,
            'progress_reward': progress_reward,
            'velocity_reward': velocity_reward,
            'cbf_reward': cbf_reward,
            'crossing_reward': crossing_reward,
            'robot_force_penalty': robot_force_penalty,
            'human_force_penalty': human_force_penalty,
            'completion_reward': completion_reward,
            'deviation': deviations,
            'progress_ratio': progress_ratio,
            'distance_to_setpoint': torch.norm(self.stylus_pos_t1 - self.trajectory_manager.get_current_setpoint_local(), dim=-1),
            'distance_to_final': distance_to_final
        }
        
        # 记录详细信息
        self.reward_logger.log_step_details(
            step_count=self.step_count, 
            reward_components=self.reward_components, 
            robot_weights=robot_weights, 
            human_weights=human_weights,
            final_rewards=rewards
        )
        
        self.extras["log"] = {
            "robot_reward": rewards["robot"].mean().item(),
            "human_reward": rewards["human"].mean().item(),
            "deviation": deviations.mean().item(),
            "progress": progress_ratio.mean().item(),
            "safety_distance": self.safety_distances_t.mean().item(),
            "crossing_reward": crossing_reward.mean().item(),
        }
        
        return rewards
        
    def _get_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """确定终止条件"""
        
        # Z轴低于0终止
        z_below_zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.enable_z_termination:
            z_below_zero = self.stylus_pos_t1[:, 2] < self.min_z_pos
        
        # 边缘碰撞终止 - 使用0.001
        edge_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.enable_edge_termination:
            edge_collision = self.safety_distances_t < self.safety_distance_threshold
        
        # 到达终点
        final_reached = self.trajectory_manager.is_final_setpoint_reached(self.stylus_pos_t1)
        
        # 合并终止条件
        terminated_condition = z_below_zero | edge_collision | final_reached

        # 时间截断
        truncated_condition = self.episode_length_buf >= self.max_episode_length - 1
        
        # 记录终止原因
        if terminated_condition.any():
            terminated_envs = torch.where(terminated_condition)[0]
            for env_id in terminated_envs:
                env_id_item = env_id.item()
                reasons = []
                if z_below_zero[env_id_item]:
                    reasons.append("Z轴低于0")
                if edge_collision[env_id_item]:
                    reasons.append("边缘碰撞")
                if final_reached[env_id_item]:
                    reasons.append("到达终点")
                
                if reasons:
                    self.reward_logger.log_termination(env_id_item, reasons)
        
        if truncated_condition.any():
            truncated_envs = torch.where(truncated_condition)[0]
            for env_id in truncated_envs:
                self.reward_logger.log_termination(env_id.item(), ["时间截断"])
        
        terminated = {agent: terminated_condition for agent in self.cfg.possible_agents}
        truncated = {agent: truncated_condition for agent in self.cfg.possible_agents}
        
        return terminated, truncated
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """重置环境"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        super()._reset_idx(env_ids)
        
        self.reward_logger.on_episode_end(env_ids)
        
        if self.stylus_body_idx is None:
            self._initialize_body_indices()
        
        num_resets = len(env_ids)
        
        # 使用稳定的初始配置
        joint_pos = torch.zeros((num_resets, 6), device=self.device)
        joint_pos[:, 0] = -0.96
        joint_pos[:, 1] = 0.0
        joint_pos[:, 2] = 1.0
        joint_pos[:, 3] = 0.0
        joint_pos[:, 4] = 2.0944
        joint_pos[:, 5] = 0.0
        
        joint_vel = torch.zeros((num_resets, 6), device=self.device)
        
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # 重置状态
        for agent in self.cfg.possible_agents:
            self.agent_actions[agent][env_ids] = 0.0
        
        self.human_forces_t[env_ids] = 0.0
        self.robot_forces_t[env_ids] = 0.0
        self.safety_distances_t[env_ids] = 0.01
        self.is_violating_t[env_ids] = False
        
        self.trajectory_manager.reset_trajectory(env_ids)
        
    def _get_stylus_position(self):
        """获取stylus位置"""
        if self.stylus_body_idx is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        
        base_pos = self._omni_robot.data.root_link_pos_w
        ee_pos = self._omni_robot.data.body_link_pos_w[:, self.stylus_body_idx, :]
        return ee_pos - base_pos
    
    def _get_stylus_velocity(self):
        """获取stylus速度"""
        if self.stylus_body_idx is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        
        return self._omni_robot.data.body_link_lin_vel_w[:, self.stylus_body_idx, :]
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'reward_logger'):
            self.reward_logger.close_all_files()
    
    # 公共接口方法
    def get_trajectory_info(self) -> Dict:
        return self.trajectory_manager.get_trajectory_info()
    
    def get_constraint_state(self, env_ids: Optional[List[int]] = None) -> Dict:
        if self.last_constraint_results is None:
            return {}
        
        if env_ids is None:
            return self.last_constraint_results
        
        return {
            'distances_constraint': self.last_constraint_results['distances_constraint'][env_ids],
            'closest_points': self.last_constraint_results['closest_points'][env_ids],
            'normal_vectors': self.last_constraint_results['normal_vectors'][env_ids],
            'is_overlapping': self.last_constraint_results['is_overlapping'][env_ids],
            'is_inside': self.last_constraint_results['is_inside'][env_ids]
        }
    
    def get_reward_details(self, env_ids: Optional[List[int]] = None) -> Dict:
        if not self.reward_components:
            return {}
        
        if env_ids is None:
            return self.reward_components
        
        return {
            key: value[env_ids] if torch.is_tensor(value) else value
            for key, value in self.reward_components.items()
        }