# surgical_direct_marl_env.py - 优化版，简化RewardLogger

from __future__ import annotations

import torch
import numpy as np
import yaml
import os
import gymnasium as gym
from typing import Any, Dict, List, Optional
from collections import defaultdict

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
        
        current_base_positions = self._omni_robot.data.root_link_pos_w if hasattr(self, '_omni_robot') else env_base_positions #备用
        
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
        self.switch_threshold = traj.get('switch_threshold', 0.01)      # 1cm
    
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
    
    def get_trajectory_info(self) -> Dict:
        """获取轨迹信息"""
        return {
            'start_point': self.start_pos_local.cpu().numpy(),
            'end_point': self.end_pos_local.cpu().numpy(),
            'total_distance': self.total_distance
        }

class RewardLogger:
    """优化版奖励记录器 - 只在milestone时记录详细数据"""

    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        
        # Episode计数（已完成的episode数，从0开始）
        self.episode_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        # 基本统计数据（始终记录）
        self.basic_stats = {
            env_id: {
                'total_episodes': 0,
                'total_steps': 0,
                'total_reward': 0.0,
                'completed_episodes': 0,
                'collision_episodes': 0,
            } for env_id in range(num_envs)
        }
        
        # 当前episode的基本数据（始终记录）
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
        
        # 详细数据（只在milestone episode时记录）
        self.milestone_detailed_data = {}
        
        # milestone配置
        self.milestones = []
        self.milestone_performances = {}
        self.milestone_completion_status = {}  # 追踪每个milestone的完成情况
        
        # 文本日志（可选）
        self.enable_text_logging = False
        self.log_dir = None
        self.env_log_files = {}
        
        # 兼容性别名
        self.env_episode_counts = self.episode_count
        self.env_step_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        # 新增：用于存储回调函数的变量
        self.topk_update_callback = None

    # 新增：设置回调函数的方法
    def set_topk_update_callback(self, callback_fn):
        """Sets a callback function to be triggered at milestones for Top-K updates."""
        self.topk_update_callback = callback_fn

    def configure_logging(self, params):
        """配置日志"""
        logging_config = params.get('logging', {})
        self.enable_text_logging = logging_config.get('enable_text_logging', False)
        
        # 从YAML读取milestone配置
        training_monitor = params.get('training_monitor', {})
        yaml_milestones = training_monitor.get('milestone_episodes', None)
        
        if yaml_milestones:
            self.milestones = yaml_milestones
            print(f"[INFO] 使用YAML配置的Milestones: {self.milestones}")
        else:
            self.milestones = [2, 10, 20, 30, 50, 100]
            print(f"[INFO] 使用默认Milestones: {self.milestones}")
        
        # 初始化milestone数据结构
        self.milestone_performances = {m: {} for m in self.milestones}
        self.milestone_completion_status = {m: {'completed_envs': 0, 'reported': False} for m in self.milestones}
        
        if self.enable_text_logging:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = f"logs/env_details/{timestamp}"
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"[INFO] 文本日志已启用，只记录milestone episodes，保存到: {self.log_dir}")
        else:
            print(f"[INFO] 文本日志已禁用")
        
        print(f"[INFO] RewardLogger初始化: {self.num_envs}个环境")
        print(f"[INFO] 将在以下Episodes进行性能评估: {self.milestones}")
    
    def is_milestone_episode(self, episode_num):
        """检查是否为milestone episode"""
        return episode_num in self.milestones
    
    def on_step(self, env_ids=None):
        """每步更新步数"""
        if env_ids is None:
            for env_id in range(self.num_envs):
                self.current_episode_basic[env_id]['steps'] += 1
                self.env_step_counts[env_id] += 1
        else:
            if torch.is_tensor(env_ids):
                env_ids_list = env_ids.cpu().numpy().tolist()
            elif isinstance(env_ids, int):
                env_ids_list = [env_ids]
            else:
                env_ids_list = env_ids
                
            for env_id in env_ids_list:
                self.current_episode_basic[env_id]['steps'] += 1
                self.env_step_counts[env_id] += 1
    
    def update_step_metrics(self, env_id, reward_components, safety_distance, rewards):
        """更新step指标 - 优化版"""
        current_episode = self.episode_count[env_id].item() + 1
        
        # 更新基本数据（始终进行）
        robot_reward = rewards["robot"][env_id].item() if "robot" in rewards else 0
        human_reward = rewards["human"][env_id].item() if "human" in rewards else 0
        total_reward = robot_reward + human_reward
        
        basic = self.current_episode_basic[env_id]
        basic['total_reward'] += total_reward
        
        progress = reward_components['progress_ratio'][env_id].item()
        basic['final_progress'] = progress
        
        safety_dist = safety_distance[env_id].item()
        basic['min_safety_distance'] = min(basic['min_safety_distance'], safety_dist)
        
        if reward_components['completion_reward'][env_id].item() > 0:
            basic['completed'] = True
            
        if safety_dist < 0.001:
            basic['collision'] = True
        
        # 只在milestone episode时记录详细数据
        if self.is_milestone_episode(current_episode):
            if current_episode not in self.milestone_detailed_data:
                self.milestone_detailed_data[current_episode] = {}
            
            if env_id not in self.milestone_detailed_data[current_episode]:
                self.milestone_detailed_data[current_episode][env_id] = {
                    'rewards': [],
                    'deviations': [],
                    'safety_distances': [],
                    'progress_ratios': []
                }
            
            detailed = self.milestone_detailed_data[current_episode][env_id]
            detailed['rewards'].append(total_reward)
            detailed['deviations'].append(reward_components['deviation'][env_id].item())
            detailed['safety_distances'].append(safety_dist)
            detailed['progress_ratios'].append(progress)
            
            # 文本日志（只在milestone episode时）
            if self.enable_text_logging:
                self._log_step_to_file(env_id, current_episode, reward_components, rewards, safety_distance)
    
    def _log_step_to_file(self, env_id, episode_num, reward_components, rewards, safety_distance):
        """记录详细步骤信息到文件（只在milestone episode时）"""
        if env_id not in self.env_log_files:
            log_file_path = os.path.join(self.log_dir, f"env_{env_id}_episode_{episode_num}.txt")
            self.env_log_files[env_id] = open(log_file_path, 'w')
            self.env_log_files[env_id].write(f"Environment {env_id} - Episode {episode_num} (MILESTONE)\n")
            self.env_log_files[env_id].write("="*60 + "\n")
        
        log_file = self.env_log_files[env_id]
        step = self.current_episode_basic[env_id]['steps']
        
        if step <= 10 or step % 50 == 0:
            log_file.write(f"\n[Step {step}]\n")
            log_file.write("-" * 40 + "\n")
            
            robot_reward = rewards["robot"][env_id].item() if "robot" in rewards else 0
            human_reward = rewards["human"][env_id].item() if "human" in rewards else 0
            
            log_file.write(f"Robot reward: {robot_reward:.4f}\n")
            log_file.write(f"Human reward: {human_reward:.4f}\n")
            log_file.write(f"Total reward: {robot_reward + human_reward:.4f}\n")
            
            for key, value in reward_components.items():
                if torch.is_tensor(value):
                    log_file.write(f"{key}: {value[env_id].item():.4f}\n")
            
            log_file.write(f"Safety distance: {safety_distance[env_id]:.4f}m\n")
            log_file.flush()
    
    def calculate_performance_score(self, env_id, basic_data=None, detailed_data=None):
        """计算性能分数"""
        if basic_data is None:
            basic_data = self.current_episode_basic[env_id]
        
        if basic_data['steps'] == 0:
            return 0.0
        
        # 1. 完成分数 (40分)
        completion_score = 40.0 if basic_data['completed'] else 0.0
        
        # 2. 进度分数 (20分)
        progress_score = 20.0 * min(1.0, basic_data['final_progress'])
        
        # 3. 轨迹精度 (20分) - 如果有详细数据则使用，否则估算
        trajectory_score = 0.0
        if detailed_data and 'deviations' in detailed_data and detailed_data['deviations']:
            avg_deviation = np.mean(detailed_data['deviations'])
            trajectory_score = 20.0 * max(0, min(1.0, (0.05 - avg_deviation) / 0.04))
        else:
            # 基于完成情况估算
            trajectory_score = 10.0 if basic_data['completed'] else 5.0
        
        # 4. 安全性 (20分)
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
    
    def on_episode_end(self, env_ids):
        """Episode结束处理 - 优化版"""
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor([env_ids] if isinstance(env_ids, int) else env_ids, device=self.device)
        
        for env_id in env_ids:
            env_id_item = env_id.item()
            current_episode_num = self.episode_count[env_id_item].item() + 1
            
            basic = self.current_episode_basic[env_id_item]
            
            # 更新基本统计
            stats = self.basic_stats[env_id_item]
            stats['total_episodes'] += 1
            stats['total_steps'] += basic['steps']
            stats['total_reward'] += basic['total_reward']
            if basic['completed']:
                stats['completed_episodes'] += 1
            if basic['collision']:
                stats['collision_episodes'] += 1
            
            # 只在milestone episode时输出详细日志和计算分数
            if self.is_milestone_episode(current_episode_num):
                detailed_data = None
                if current_episode_num in self.milestone_detailed_data and env_id_item in self.milestone_detailed_data[current_episode_num]:
                    detailed_data = self.milestone_detailed_data[current_episode_num][env_id_item]
                
                score = self.calculate_performance_score(env_id_item, basic, detailed_data)
                
                # 输出milestone日志
                print(f"[MILESTONE {current_episode_num}] Env {env_id_item}: "
                      f"Steps={basic['steps']}, Progress={basic['final_progress']:.1%}, "
                      f"Completed={basic['completed']}, Collision={basic['collision']}, "
                      f"Score={score:.2f}/100")
                
                # 存储milestone性能数据
                self.milestone_performances[current_episode_num][env_id_item] = {
                    'score': score,
                    'completed': basic['completed'],
                    'steps': basic['steps'],
                    'collision': basic['collision'],
                    'final_progress': basic['final_progress'],
                    'avg_reward': basic['total_reward'] / basic['steps'] if basic['steps'] > 0 else 0.0
                }
                
                # 更新milestone完成状态
                self.milestone_completion_status[current_episode_num]['completed_envs'] += 1
                
                # 检查是否所有环境都达到此milestone
                self._check_milestone_completion(current_episode_num)
                
                # 关闭文本日志文件
                if self.enable_text_logging and env_id_item in self.env_log_files:
                    self.env_log_files[env_id_item].close()
                    del self.env_log_files[env_id_item]
            
            # 更新episode计数和重置
            self.episode_count[env_id_item] += 1
            self.env_step_counts[env_id_item] = 0
            
            # 重置当前episode数据
            self.current_episode_basic[env_id_item] = {
                'steps': 0,
                'total_reward': 0.0,
                'final_progress': 0.0,
                'completed': False,
                'collision': False,
                'min_safety_distance': float('inf'),
            }
    
    def _check_milestone_completion(self, milestone):
        """检查milestone完成情况并报告"""
        status = self.milestone_completion_status[milestone]
        
        # 如果所有环境都达到了这个milestone且还未报告
        if status['completed_envs'] >= self.num_envs and not status['reported']:
            # 修改：触发回调
            if self.topk_update_callback:
                print(f"[CALLBACK] Triggering Top-K update for milestone {milestone}...")
                self.topk_update_callback(milestone)

            self._report_milestone_summary(milestone)
            status['reported'] = True
    
    def _report_milestone_summary(self, milestone):
        """输出milestone汇总报告"""
        performances = self.milestone_performances[milestone]
        
        if len(performances) != self.num_envs:
            print(f"[WARNING] Milestone {milestone}: 只有{len(performances)}/{self.num_envs}个环境数据")
            return
        
        scores = [p['score'] for p in performances.values()]
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        completion_rate = sum(1 for p in performances.values() if p['completed']) / len(performances)
        avg_steps = np.mean([p['steps'] for p in performances.values()])
        collision_rate = sum(1 for p in performances.values() if p['collision']) / len(performances)
        avg_progress = np.mean([p['final_progress'] for p in performances.values()])
        
        print(f"\n{'='*60}")
        print(f"[MILESTONE {milestone} COMPLETE] 所有环境已达到")
        print(f"{'='*60}")
        print(f"  平均分数: {avg_score:.2f} ± {std_score:.2f} / 100")
        print(f"  完成率: {completion_rate:.1%}")
        print(f"  碰撞率: {collision_rate:.1%}")
        print(f"  平均进度: {avg_progress:.1%}")
        print(f"  平均步数: {avg_steps:.1f}")
        print(f"{'='*60}\n")
    
    def get_final_evaluation(self, env_id, target_episodes):
        """获取最终评估分数"""
        # 使用最近的milestone分数
        recent_milestones = [m for m in self.milestones if m <= target_episodes][-3:]
        scores = []
        
        for milestone in recent_milestones:
            if milestone in self.milestone_performances:
                if env_id in self.milestone_performances[milestone]:
                    score = self.milestone_performances[milestone][env_id]['score']
                    scores.append(score)
        
        if scores:
            final_score = np.mean(scores)
        else:
            # 基于基本统计估算
            stats = self.basic_stats[env_id]
            if stats['total_episodes'] > 0:
                completion_rate = stats['completed_episodes'] / stats['total_episodes']
                collision_rate = stats['collision_episodes'] / stats['total_episodes']
                final_score = (completion_rate * 60) + (1 - collision_rate) * 40
            else:
                final_score = 0.0
        
        return np.clip(final_score, 0, 100)
    
    def get_next_milestone_progress(self):
        """获取下一个milestone进度"""
        min_episodes = self.episode_count.min().item()
        for milestone in self.milestones:
            if milestone > min_episodes:
                reached = (self.episode_count >= milestone).sum().item()
                return milestone, reached, self.num_envs - reached
        return None, 0, 0
    
    def close_all_files(self):
        """关闭所有日志文件"""
        for f in self.env_log_files.values():
            if f and not f.closed:
                f.close()
        self.env_log_files.clear()

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
        self.safety_distance_threshold = term_config.get('safety_distance_threshold', 0.001)
        
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
        """计算奖励"""
        self.reward_logger.on_step()
        
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
        velocity_reward = torch.exp(-velocity_along_line * 20.0)
        
        # 4. CBF障碍物奖励
        cbf_reward = torch.where(
            self.safety_distances_t < 0.001,
            torch.full_like(self.safety_distances_t, -500.0),
            torch.where(
                self.safety_distances_t < 0.008,
                torch.full_like(self.safety_distances_t, -200.0),
                0.5 + 10.0 * torch.clamp(self.safety_distances_t, max=0.05)
            )
        )
        
        # 6. 力惩罚
        robot_force_penalty = -50.0 * torch.sum(self.robot_forces_t**2, dim=-1)
        human_force_penalty = -50.0 * torch.sum(self.human_forces_t**2, dim=-1)
        
        # 7. Z轴软约束
        z_penalty = torch.where(
            self.stylus_pos_t1[:, 2] < 0.0,
            -500.0 * torch.abs(self.stylus_pos_t1[:, 2]),
            torch.zeros_like(self.stylus_pos_t1[:, 2])
        )
        
        # 8. 完成奖励
        distance_to_final = torch.norm(
            self.stylus_pos_t1 - self.trajectory_manager.end_pos_local.unsqueeze(0), 
            dim=-1
        )
        completion_reward = torch.where(
            distance_to_final < 0.01,
            torch.full_like(distance_to_final, 50.0),
            torch.zeros_like(distance_to_final)
        )
        
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
            robot_force_penalty * robot_weights['force_efficiency'] +
            human_force_penalty * robot_weights['human_awareness'] +
            z_penalty +
            completion_reward
        )
        
        rewards["human"] = (
            trajectory_reward * human_weights['trajectory_tracking'] +
            progress_reward * human_weights['progress'] +
            velocity_reward * human_weights['velocity'] +
            cbf_reward * human_weights['obstacle_cbf'] +
            human_force_penalty * human_weights['force_efficiency'] +
            robot_force_penalty * human_weights['robot_awareness'] +
            z_penalty +
            completion_reward
        )
        
        # 存储组件
        self.reward_components = {
            'trajectory_reward': trajectory_reward,
            'progress_reward': progress_reward,
            'velocity_reward': velocity_reward,
            'cbf_reward': cbf_reward,
            'robot_force_penalty': robot_force_penalty,
            'human_force_penalty': human_force_penalty,
            'z_penalty': z_penalty,
            'completion_reward': completion_reward,
            'deviation': deviations,
            'progress_ratio': progress_ratio,
            'distance_to_final': distance_to_final
        }
        
        # 更新每个环境的指标（优化版 - 根据是否为milestone决定详细程度）
        for env_id in range(self.num_envs):
            self.reward_logger.update_step_metrics(
                env_id, 
                self.reward_components, 
                self.safety_distances_t,
                rewards
            )
        
        self.extras["log"] = {
            "robot_reward": rewards["robot"].mean().item(),
            "human_reward": rewards["human"].mean().item(),
            "deviation": deviations.mean().item(),
            "progress": progress_ratio.mean().item(),
            "safety_distance": self.safety_distances_t.mean().item(),
            "z_penalty": z_penalty.mean().item(),
        }
        
        return rewards
        
    def _get_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """确定终止条件"""
        
        # Z轴低于0终止
        z_below_zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.enable_z_termination:
            z_below_zero = self.stylus_pos_t1[:, 2] < self.min_z_pos
        
        # 边缘碰撞终止
        edge_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.enable_edge_termination:
            edge_collision = self.safety_distances_t < self.safety_distance_threshold
        
        # 到达终点
        final_reached = self.trajectory_manager.is_final_setpoint_reached(self.stylus_pos_t1)
        
        # 合并终止条件
        terminated_condition = z_below_zero | edge_collision | final_reached

        # 时间截断 - IsaacLab会管理每个环境的独立episode_length_buf
        truncated_condition = self.episode_length_buf >= self.max_episode_length - 1
        
        terminated = {agent: terminated_condition for agent in self.cfg.possible_agents}
        truncated = {agent: truncated_condition for agent in self.cfg.possible_agents}
        
        return terminated, truncated
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """重置环境"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 只在episode实际运行后才记录结束
        if torch.is_tensor(env_ids):
            env_ids_list = env_ids.cpu().numpy().tolist()
        else:
            env_ids_list = [env_ids] if isinstance(env_ids, int) else env_ids
        
        # 过滤出实际运行过的环境
        valid_env_ids = []
        for env_id in env_ids_list:
            if self.reward_logger.current_episode_basic[env_id]['steps'] > 0:
                valid_env_ids.append(env_id)
        
        # 只记录实际运行过的episodes
        if valid_env_ids:
            self.reward_logger.on_episode_end(torch.tensor(valid_env_ids, device=self.device))
        
        # 调用父类重置
        super()._reset_idx(env_ids)
        
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