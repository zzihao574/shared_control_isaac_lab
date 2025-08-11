# surgical_direct_marl_env.py - 修复多环境版本

from __future__ import annotations

import torch
import numpy as np
import yaml
import os
import gymnasium as gym
from typing import Any, Dict

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.math import sample_uniform, quat_rotate_inverse

from .surgical_direct_marl_env_cfg import SurgicalDirectMARLEnvCfg


class CompleteConstraintChecker:
    """完整的约束状态检测类 - 支持多环境批量处理"""
    
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
            
        # 用于调试
        self.debug_constraint_paths = set()
        self.debug_count = 0
    
    def analyze_constraint_state_batch(self, stylus_positions: torch.Tensor, env_base_positions: torch.Tensor):
        """批量分析约束状态 - 使用实时base位置"""
        num_envs = stylus_positions.shape[0]
        
        # 获取实时的环境base位置
        current_base_positions = self._omni_robot.data.root_link_pos_w if hasattr(self, '_omni_robot') else env_base_positions
        
        # 初始化结果 - 默认为安全状态（距离远，无重叠）
        batch_results = {
            'distances_constraint': torch.ones(num_envs, device=self.device) * 0.2,  # 默认20厘米安全距离
            'closest_points': torch.zeros(num_envs, 3, device=self.device),
            'normal_vectors': torch.ones(num_envs, 3, device=self.device),
            'is_overlapping': torch.zeros(num_envs, dtype=torch.bool, device=self.device),  # 默认不重叠
            'is_inside': torch.zeros(num_envs, dtype=torch.bool, device=self.device)  # 默认不在内部
        }
        
        if self.physics_attachment_interface is None or self.physics_scene_query_interface is None:
            if self.debug_count < 3:
                print("[WARNING] Physics interfaces not available, using default safe values")
            return batch_results
        
        # 逐环境处理
        for env_id in range(num_envs):
            # 计算世界坐标系中的stylus位置
            stylus_world_pos = stylus_positions[env_id] + current_base_positions[env_id]
            
            # 基于单环境路径 /World/Constraint/mesh 构建多环境路径
            # 修复：使用 env_0, env_1 而不是 env_000, env_001
            constraint_path = f"/World/envs/env_{env_id}/Constraint/mesh"
            
            # 调试信息
            if self.debug_count < 3:
                print(f"[CONSTRAINT DEBUG] Env {env_id}:")
                print(f"  Stylus local pos: {stylus_positions[env_id].cpu().numpy()}")
                print(f"  Current base pos: {current_base_positions[env_id].cpu().numpy()}")
                print(f"  Stylus world pos: {stylus_world_pos.cpu().numpy()}")
                print(f"  Constraint path: {constraint_path}")
                
                # 计算到约束中心的预期距离（用于验证）
                constraint_center_local = torch.tensor([0.14, 0.0, 0.0], device=self.device)
                constraint_center_world = constraint_center_local + current_base_positions[env_id]
                expected_distance = torch.norm(stylus_world_pos - constraint_center_world).item()
                print(f"  Expected distance to constraint center: {expected_distance:.6f}m")
            
            result = None
            try:
                result = self._analyze_single_constraint(stylus_world_pos, constraint_path)
                if result is not None:
                    # 验证结果的合理性
                    if result['distance'] > 0 and not (result['closest_point'] == np.array([0., 0., 0.])).all():
                        if constraint_path not in self.debug_constraint_paths:
                            print(f"[SUCCESS] Found constraint for env {env_id} at: {constraint_path}")
                            self.debug_constraint_paths.add(constraint_path)
                    else:
                        if self.debug_count < 3:
                            print(f"[DEBUG] Invalid result from {constraint_path}: distance={result['distance']}, closest_point={result['closest_point']}")
                        result = None
            except Exception as e:
                if self.debug_count < 3:
                    print(f"[DEBUG] Constraint path {constraint_path} failed: {e}")
            
            if result is not None:
                # 物理接口调用成功
                batch_results['distances_constraint'][env_id] = result['distance']
                batch_results['closest_points'][env_id] = torch.tensor(result['closest_point'], device=self.device)
                batch_results['normal_vectors'][env_id] = torch.tensor(result['normal_vector'], device=self.device)
                batch_results['is_overlapping'][env_id] = result['is_overlapping']
                batch_results['is_inside'][env_id] = result['is_inside']
            else:
                # 物理接口调用失败，保持默认安全值
                if self.debug_count < 3:
                    print(f"[WARNING] Constraint detection failed for env {env_id}, using default safe values")
            
            # 调试信息
            if self.debug_count < 3 and env_id <= 1:  # 只显示前2个环境
                print(f"[CONSTRAINT RESULT] Env {env_id}:")
                print(f"  Distance: {batch_results['distances_constraint'][env_id].item():.6f}m")
                print(f"  Closest point world: {batch_results['closest_points'][env_id].cpu().numpy()}")
                closest_local = batch_results['closest_points'][env_id] - current_base_positions[env_id]
                print(f"  Closest point local: {closest_local.cpu().numpy()}")
                print(f"  Overlapping: {batch_results['is_overlapping'][env_id].item()}")
                print(f"  Inside: {batch_results['is_inside'][env_id].item()}")
        
        self.debug_count += 1
        return batch_results
    
    def _analyze_single_constraint(self, stylus_position: torch.Tensor, constraint_path: str):
        """分析单个环境的约束状态 - 添加详细调试"""
        try:
            from carb._carb import Float3
            
            pos = stylus_position.cpu().numpy()
            current_point = Float3(float(pos[0]), float(pos[1]), float(pos[2]))
            
            if self.debug_count < 3:
                print(f"[PHYSICS DEBUG] Calling get_closest_points:")
                print(f"  Input point: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
                print(f"  Constraint path: {constraint_path}")
            
            # 1. 获取最近点
            result = self.physics_attachment_interface.get_closest_points([current_point], constraint_path)
            
            if self.debug_count < 3:
                print(f"  get_closest_points result keys: {result.keys() if result else 'None'}")
                if result and 'closest_points' in result:
                    print(f"  closest_points length: {len(result['closest_points'])}")
                    if result['closest_points']:
                        closest_pt = result['closest_points'][0]
                        print(f"  Raw closest point: [{closest_pt.x:.6f}, {closest_pt.y:.6f}, {closest_pt.z:.6f}]")
            
            if not (result and 'closest_points' in result and result['closest_points']):
                if self.debug_count < 3:
                    print(f"[PHYSICS DEBUG] get_closest_points failed or returned empty")
                return None
            
            closest_pt = result['closest_points'][0]
            closest_pos = np.array([closest_pt.x, closest_pt.y, closest_pt.z])
            
            # 2. 计算距离
            distance = float(np.linalg.norm(pos - closest_pos))
            
            if self.debug_count < 3:
                print(f"  Computed distance: {distance:.6f}")
                print(f"  Distance should be close to expected ~0.2m for current test case")
            
            # 3. 检查重叠：从stylus向最近点发射射线
            direction_to_closest = Float3(
                closest_pt.x - pos[0], 
                closest_pt.y - pos[1], 
                closest_pt.z - pos[2]
            )
            
            raycast_result = self.physics_scene_query_interface.raycast_closest(
                current_point, direction_to_closest, 10000
            )
            
            if self.debug_count < 3:
                print(f"  Raycast result keys: {raycast_result.keys() if raycast_result else 'None'}")
            
            # 判断重叠（按照源代码逻辑）
            is_overlapping = False
            if (('faceIndex' not in raycast_result) and distance < 0.5) or \
               ('faceIndex' in raycast_result and raycast_result.get('faceIndex', -1) == 0 and distance < 0.5):
                is_overlapping = True
            
            # 4. 检查是否在内部（使用水平射线检测）
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
            
            is_inside_1 = self.physics_scene_query_interface.raycast_any(
                current_point, direction_is_inside_1, 10000
            )
            is_inside_2 = self.physics_scene_query_interface.raycast_any(
                current_point, direction_is_inside_2, 10000
            )
            
            is_inside = bool(is_inside_1) and bool(is_inside_2)
            
            # 5. 获取法向量
            normal_vector = np.array([1.0, 0.0, 0.0])  # 默认法向量
            
            if is_overlapping:
                # 重叠时使用镜像点方法
                mirror_point_pos = 2 * closest_pos - pos
                mirror_point = Float3(float(mirror_point_pos[0]), float(mirror_point_pos[1]), float(mirror_point_pos[2]))
                
                direction_mirror_to_current = Float3(
                    pos[0] - mirror_point_pos[0],
                    pos[1] - mirror_point_pos[1], 
                    pos[2] - mirror_point_pos[2]
                )
                
                mirror_raycast = self.physics_scene_query_interface.raycast_closest(
                    mirror_point, direction_mirror_to_current, 10000
                )
                
                if mirror_raycast and 'normal' in mirror_raycast:
                    normal_carb = mirror_raycast['normal']
                    if is_inside:
                        normal_vector = np.array([normal_carb.x, normal_carb.y, normal_carb.z])
                    else:
                        normal_vector = -np.array([normal_carb.x, normal_carb.y, normal_carb.z])
            else:
                # 不重叠时直接使用射线结果
                if raycast_result and 'normal' in raycast_result:
                    normal_carb = raycast_result['normal']
                    if is_inside:
                        normal_vector = np.array([normal_carb.x, normal_carb.y, normal_carb.z])
                    else:
                        normal_vector = -np.array([normal_carb.x, normal_carb.y, normal_carb.z])
            
            if self.debug_count < 3:
                print(f"[PHYSICS DEBUG] Final result:")
                print(f"  Distance: {distance:.6f}")
                print(f"  Closest point: {closest_pos}")
                print(f"  Is overlapping: {is_overlapping}")
                print(f"  Is inside: {is_inside}")
                print(f"  Normal vector: {normal_vector}")
            
            return {
                'distance': distance,
                'closest_point': closest_pos,
                'normal_vector': normal_vector,
                'is_overlapping': is_overlapping,
                'is_inside': is_inside
            }
                
        except Exception as e:
            print(f"[ERROR] Constraint state analysis failed for {constraint_path}: {e}")
            import traceback
            traceback.print_exc()
            return None


class TrajectoryManager:
    """轨迹管理器"""
    
    def __init__(self, device: torch.device, params: dict, num_envs: int, env_base_positions: torch.Tensor):
        self.device = device
        self.num_envs = num_envs
        self.env_base_positions = env_base_positions
        
        traj = params['trajectory']
        self.start_pos_local = torch.tensor(traj['start_point'], device=device, dtype=torch.float32)
        self.end_pos_local = torch.tensor(traj['end_point'], device=device, dtype=torch.float32)
        self.setpoint_interval = traj['setpoint_interval']
        self.switch_threshold = traj['switch_threshold']
        
        self._generate_setpoints(params)
        self.current_setpoint_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.setpoints_tensor = torch.stack(self.setpoints_local)
        self.num_setpoints = len(self.setpoints_local)
        
    def _generate_setpoints(self, params):
        """生成轨迹设置点"""
        constraint = params['constraint_geometry']
        y_pos_range = constraint['y_range_positive']
        y_neg_range = constraint['y_range_negative']
        
        total_distance = torch.norm(self.end_pos_local - self.start_pos_local).item()
        num_setpoints = int(total_distance / self.setpoint_interval) + 1
        
        self.setpoints_local = []
        direction = (self.end_pos_local - self.start_pos_local) / torch.norm(self.end_pos_local - self.start_pos_local)
        
        for i in range(num_setpoints + 1):
            setpoint = self.end_pos_local.clone() if i == num_setpoints else self.start_pos_local + direction * (i * self.setpoint_interval)
            
            y_coord = setpoint[1].item()
            in_constraint = (y_pos_range[0] <= y_coord <= y_pos_range[1]) or (y_neg_range[0] <= y_coord <= y_neg_range[1])
            
            if not in_constraint:
                self.setpoints_local.append(setpoint)
        
        print(f"[TRAJECTORY] 生成了 {len(self.setpoints_local)} 个轨迹点")
        print(f"[TRAJECTORY] 起点: {self.start_pos_local.cpu().numpy()}")
        print(f"[TRAJECTORY] 终点: {self.end_pos_local.cpu().numpy()}")
        
    def get_current_setpoint_local(self) -> torch.Tensor:
        """获取当前设置点（局部坐标）"""
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
        
    def reset_trajectory(self, env_ids: torch.Tensor = None):
        """重置轨迹"""
        if env_ids is None:
            self.current_setpoint_idx.fill_(0)
        else:
            self.current_setpoint_idx[env_ids] = 0
        
    def is_final_setpoint_reached(self, current_pos_local: torch.Tensor) -> torch.Tensor:
        """检查是否到达终点"""
        at_final = (self.current_setpoint_idx >= self.num_setpoints - 1)
        final_setpoint = self.setpoints_local[-1]
        distances_to_final = torch.norm(current_pos_local - final_setpoint.unsqueeze(0), dim=-1)
        return at_final & (distances_to_final < self.switch_threshold)


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
        self.safety_distances_t = torch.ones(self.num_envs, device=self.device) * 0.01  # 1厘米初始安全距离
        self.is_violating_t = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        self.fixed_end_joints = torch.tensor([
            self.params['initial_conditions']['joint_positions']['yaw'],
            self.params['initial_conditions']['joint_positions']['pitch'],
            self.params['initial_conditions']['joint_positions']['roll']
        ], device=self.device, dtype=torch.float32)
        
        self.stylus_body_idx = None
        self.constraint_checker = CompleteConstraintChecker(self.device, self.collision_threshold)
        
        self.debug_step_count = 0
        self.debug_interval = 1
        
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
        self.max_cartesian_vel = constraints['max_cartesian_velocity']
        self.min_z_pos = constraints['min_z_position']
        self.max_robot_force = constraints['max_robot_force']
        self.max_human_force = constraints['max_human_force']
        
        print(f"[PARAMS] 最大力: robot={self.max_robot_force}, human={self.max_human_force}")
        print(f"[PARAMS] 最大速度: {self.max_cartesian_vel}, 最小Z: {self.min_z_pos}")
        
        joint_limits = constraints['joint_limits']
        self.joint_lower_limits = torch.tensor([
            joint_limits['waist'][0], joint_limits['shoulder'][0], joint_limits['elbow'][0],
            joint_limits['yaw'][0], joint_limits['pitch'][0], joint_limits['roll'][0]
        ], device=self.device, dtype=torch.float32)
        
        self.joint_upper_limits = torch.tensor([
            joint_limits['waist'][1], joint_limits['shoulder'][1], joint_limits['elbow'][1],
            joint_limits['yaw'][1], joint_limits['pitch'][1], joint_limits['roll'][1]
        ], device=self.device, dtype=torch.float32)
        
        # 约束相关参数 - 详细说明
        self.safety_margin = self.params['reward_parameters']['cbf_parameters']['safety_margin']  # CBF安全边距(米)
        self.constraint_center = torch.tensor(
            self.params['constraint_geometry']['center'], 
            device=self.device, dtype=torch.float32
        )  # 约束中心位置(局部坐标)
        self.collision_threshold = self.params['constraint_geometry']['collision_threshold']  # 碰撞检测阈值
        self.cbf_gamma = self.params['reward_parameters']['cbf_parameters']['gamma']  # CBF gamma参数
        self.cbf_epsilon = self.params['reward_parameters']['cbf_parameters']['epsilon']  # CBF epsilon参数
        
        print(f"[CONSTRAINT PARAMS] 约束相关参数:")
        print(f"  安全边距: {self.safety_margin}m")
        print(f"  约束中心: {self.constraint_center.cpu().numpy()}")
        print(f"  碰撞阈值: {self.collision_threshold}")
        print(f"  CBF gamma: {self.cbf_gamma}, epsilon: {self.cbf_epsilon}")
        
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
            
            # 强制覆盖USD文件中的关节刚度和阻尼
            num_joints = self._omni_robot.num_joints
            zero_stiffness = torch.zeros(self.num_envs, num_joints, device=self.device)
            zero_damping = torch.zeros(self.num_envs, num_joints, device=self.device)
            
            self._omni_robot.write_joint_stiffness_to_sim(zero_stiffness)
            self._omni_robot.write_joint_damping_to_sim(zero_damping)
            
            print(f"[OVERRIDE] 强制设置所有关节刚度和阻尼为0")
            print(f"[OVERRIDE] 关节数量: {num_joints}")
            print(f"[OVERRIDE] 环境数量: {self.num_envs}")
            print(f"[OVERRIDE] 环境基础位置形状: {self.env_base_positions.shape}")
            
        print(f"[SETUP] Stylus body index: {self.stylus_body_idx}")
        
    def _initialize_body_indices(self):
        """初始化body索引"""
        if not hasattr(self._omni_robot, 'body_names'):
            print("[WARNING] No body names found!")
            return
        
        search_patterns = ['stylus', 'tip', 'end_effector', 'link6', 'end', 'tool']
        for pattern in search_patterns:
            for i, name in enumerate(self._omni_robot.body_names):
                if pattern in name.lower():
                    self.stylus_body_idx = i
                    print(f"[DEBUG] Found stylus body: {name} at index {i}")
                    return
        
        if len(self._omni_robot.body_names) > 0:
            self.stylus_body_idx = len(self._omni_robot.body_names) - 1
            print(f"[DEBUG] Using last body as stylus: index {self.stylus_body_idx}")
        
    def _pre_physics_step(self, actions: Dict[str, torch.Tensor]) -> None:
        """物理步骤前处理 - 修复多环境版本"""
        self.debug_step_count += 1
        
        # 验证输入动作的维度
        for agent, action in actions.items():
            if agent in self.cfg.possible_agents:
                # 确保动作有正确的维度 [num_envs, 3]
                if action.dim() == 1:
                    # 如果是1D，需要扩展到所有环境
                    if action.shape[0] == 3:
                        # [3] -> [num_envs, 3]
                        action = action.unsqueeze(0).expand(self.num_envs, -1)
                    else:
                        # [num_envs] -> [num_envs, 1] -> [num_envs, 3] (假设是单一力值)
                        action = action.unsqueeze(-1).expand(-1, 3)
                elif action.dim() == 2:
                    # 应该是正确的 [num_envs, 3]
                    if action.shape[0] != self.num_envs:
                        print(f"[WARNING] Action shape mismatch for {agent}: {action.shape}, expected [{self.num_envs}, 3]")
                        # 尝试修复
                        if action.shape[0] == 1 and action.shape[1] == 3:
                            # [1, 3] -> [num_envs, 3]
                            action = action.expand(self.num_envs, -1)
                        else:
                            print(f"[ERROR] Cannot fix action shape for {agent}: {action.shape}")
                            action = torch.zeros(self.num_envs, 3, device=self.device)
                
                # 限制力的大小
                max_force = self.max_robot_force if agent == "robot" else self.max_human_force
                self.agent_actions[agent] = torch.clamp(action, -max_force, max_force)
        
        self.robot_forces_t = self.agent_actions["robot"]
        self.human_forces_t = self.agent_actions["human"]
        
        # 调试信息 - 避免重复打印
        if not hasattr(self, '_last_pre_physics_step'):
            self._last_pre_physics_step = -1
            
        if self.debug_step_count <= 5 and self.debug_step_count != self._last_pre_physics_step:
            print(f"\n========== Step {self.debug_step_count} ==========")
            print(f"[ACTOR] Actor outputs:")
            for agent_id in self.cfg.possible_agents:
                if agent_id in actions:
                    print(f"  {agent_id} input shape: {actions[agent_id].shape}")
                    print(f"  {agent_id} input[0]: {actions[agent_id][0].cpu().numpy() if actions[agent_id].dim() > 0 else actions[agent_id].cpu().numpy()}")
            
            print(f"[FORCES] Force processing:")
            print(f"  Robot max_force: {self.max_robot_force}")
            print(f"  Human max_force: {self.max_human_force}")
            print(f"  Robot clamped shape: {self.robot_forces_t.shape}")
            print(f"  Human clamped shape: {self.human_forces_t.shape}")
            print(f"  Robot clamped[0]: {self.robot_forces_t[0].cpu().numpy()}")
            print(f"  Human clamped[0]: {self.human_forces_t[0].cpu().numpy()}")
            print(f"  Total[0]: {(self.robot_forces_t[0] + self.human_forces_t[0]).cpu().numpy()}")
            print(f"  Total magnitude[0]: {torch.norm(self.robot_forces_t[0] + self.human_forces_t[0]).item():.6f}")
            
            # 获取并打印START状态
            step_start_pos = self._get_stylus_position()
            step_start_vel = self._get_stylus_velocity()
            print(f"[STEP {self.debug_step_count} START] Pos[0]: {step_start_pos[0].cpu().numpy()}, Vel[0]: {step_start_vel[0].cpu().numpy()}, |V|: {torch.norm(step_start_vel[0]).item():.4f}")
            
            self._last_pre_physics_step = self.debug_step_count
        
        # 应用外力 - 修复多环境版本
        if self.stylus_body_idx is not None:
            total_forces = self.robot_forces_t + self.human_forces_t  # [num_envs, 3]
            stylus_quat = self._omni_robot.data.body_link_quat_w[:, self.stylus_body_idx, :]  # [num_envs, 4]
            
            # 确保维度匹配
            if total_forces.shape[0] != stylus_quat.shape[0]:
                print(f"[ERROR] Dimension mismatch: total_forces {total_forces.shape}, stylus_quat {stylus_quat.shape}")
                return
            
            forces_local = quat_rotate_inverse(stylus_quat, total_forces)  # [num_envs, 3]
            
            if self.debug_step_count <= 5 and self.debug_step_count == self._last_pre_physics_step:
                print(f"[PHYSICS] Force application:")
                print(f"  Stylus quat shape: {stylus_quat.shape}")
                print(f"  Total forces shape: {total_forces.shape}")
                print(f"  Forces local shape: {forces_local.shape}")
                print(f"  World forces[0]: {total_forces[0].cpu().numpy()}")
                print(f"  Local forces[0]: {forces_local[0].cpu().numpy()}")
                print(f"  Stylus quaternion[0]: {stylus_quat[0].cpu().numpy()}")
                
                # 验证转换精度
                from isaaclab.utils.math import quat_rotate
                forces_back = quat_rotate(stylus_quat, forces_local)
                print(f"  反转换[0]: {forces_back[0].cpu().numpy()}")
                print(f"  转换误差[0]: {torch.norm(forces_back[0] - total_forces[0]).item():.8f}")
            
            # 施加外力 - forces_local已经是[num_envs, 3]，需要添加body维度
            forces_with_body_dim = forces_local.unsqueeze(1)  # [num_envs, 1, 3]
            torques_with_body_dim = torch.zeros_like(forces_with_body_dim)  # [num_envs, 1, 3]
            
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
        # 防止重复调用
        if not hasattr(self, '_apply_action_called_step'):
            self._apply_action_called_step = -1
        
        if self._apply_action_called_step == self.debug_step_count:
            return  # 已经在这个step调用过了
        
        self._apply_action_called_step = self.debug_step_count
        
        # 必须调用write_data_to_sim()将关节状态写入仿真器
        self._omni_robot.write_data_to_sim()
        
        # 更新状态缓存 - 获取仿真后的真实状态
        self.stylus_pos_t1 = self._get_stylus_position()
        self.stylus_vel_t1 = self._get_stylus_velocity()
        
        # 只在step结束时打印END（仿真后）
        if self.debug_step_count <= 5:
            print(f"[STEP {self.debug_step_count} END]   Pos[0]: {self.stylus_pos_t1[0].cpu().numpy()}, Vel[0]: {self.stylus_vel_t1[0].cpu().numpy()}, |V|: {torch.norm(self.stylus_vel_t1[0]).item():.4f}")
            
        # 处理关节数据
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
        constraint_results = self.constraint_checker.analyze_constraint_state_batch(
            self.stylus_pos_t1, current_base_positions
        )
        self.safety_distances_t = constraint_results['distances_constraint']
        self.is_violating_t = constraint_results['is_overlapping']

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """获取观测"""
        stylus_vel_constrained = torch.clamp(self.stylus_vel_t1, -self.max_cartesian_vel, self.max_cartesian_vel)
        
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
            observations[agent] = torch.clamp(obs, -10.0, 10.0)
            
        return observations
        
    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        """计算奖励
        
        距离参数说明:
        - distances_constraint: stylus到constraint表面的实际距离(米)
        - safety_margin: CBF安全边距，通常0.01m(1厘米) 
        - safety_distances_t: 当前步的安全距离，用于CBF计算
        
        CBF逻辑:
        当 distance < safety_margin 时，CBF给予负奖励
        当 distance >= safety_margin 时，CBF奖励较小
        """
        # 轨迹跟踪
        self.trajectory_manager.update_setpoint(self.stylus_pos_t1)
        current_setpoints = self.trajectory_manager.get_current_setpoint_local()
        
        pos_error = self.stylus_pos_t1 - current_setpoints
        position_tracking = -torch.sum(pos_error**2, dim=-1)
        velocity_regulation = -torch.sum(self.stylus_vel_t1**2, dim=-1)
        
        # CBF障碍函数
        s = torch.clamp(self.safety_distances_t, min=self.cbf_epsilon)
        gamma_s = self.cbf_gamma * s
        cbf_values = -torch.log(torch.clamp(gamma_s / (gamma_s + 1.0), min=self.cbf_epsilon, max=1.0 - self.cbf_epsilon))
        
        # 惩罚项
        z_violation = torch.where(
            self.stylus_pos_t1[:, 2] < self.min_z_pos,
            torch.full_like(self.stylus_pos_t1[:, 2], -500.0),
            torch.zeros_like(self.stylus_pos_t1[:, 2])
        )
        
        collision_penalty = self.is_violating_t.float() * self.params['reward_parameters']['collision_penalty'] * 0.01  # 降低碰撞惩罚
        
        # 完成奖励
        final_setpoint = self.trajectory_manager.setpoints_local[-1].unsqueeze(0).expand(self.num_envs, -1)
        distance_to_final = torch.norm(self.stylus_pos_t1 - final_setpoint, dim=-1)
        completion_reward = torch.where(
            distance_to_final < self.params['reward_parameters']['completion_threshold'],
            torch.full_like(distance_to_final, self.params['reward_parameters']['completion_reward']),
            torch.zeros_like(distance_to_final)
        )
        
        # 力冲突
        dot_product = torch.sum(self.human_forces_t * self.robot_forces_t, dim=-1)
        human_norm = torch.norm(self.human_forces_t, dim=-1)
        robot_norm = torch.norm(self.robot_forces_t, dim=-1)
        cos_angle = dot_product / (human_norm * robot_norm + 1e-6)
        
        force_magnitude = human_norm + robot_norm
        conflict_threshold = self.params['reward_parameters']['collaboration_parameters']['force_conflict_threshold']
        force_conflict = torch.where(
            (cos_angle < -0.5) & (force_magnitude > conflict_threshold),
            cos_angle * self.params['reward_parameters']['collaboration_parameters']['conflict_penalty_scale'],
            torch.zeros_like(cos_angle)
        )
        
        # 计算奖励
        rewards = {}
        
        robot_weights = self.params['reward_parameters']['robot_weights']
        robot_control_penalty = -torch.sum(self.robot_forces_t**2, dim=-1)
        rewards["robot"] = (
            position_tracking * robot_weights['position_tracking'] +
            velocity_regulation * robot_weights['velocity_regulation'] +
            cbf_values * robot_weights['obstacle_distance'] +
            robot_control_penalty * robot_weights['control_input'] +
            force_conflict + z_violation + collision_penalty + completion_reward
        )
        
        human_weights = self.params['reward_parameters']['human_weights']
        human_force_penalty = -torch.sum(self.human_forces_t**2, dim=-1)
        rewards["human"] = (
            position_tracking * human_weights['position_tracking'] +
            velocity_regulation * human_weights['velocity_regulation'] +
            cbf_values * human_weights['obstacle_distance'] +
            human_force_penalty * human_weights['force_input'] +
            force_conflict + z_violation + collision_penalty + completion_reward
        )
        
        # 存储奖励组件用于外部访问
        self.reward_components = {
            'position_tracking': position_tracking,
            'velocity_regulation': velocity_regulation,
            'cbf_values': cbf_values,
            'robot_control_penalty': robot_control_penalty,
            'human_force_penalty': human_force_penalty,
            'force_conflict': force_conflict,
            'z_violation': z_violation,
            'collision_penalty': collision_penalty,
            'completion_reward': completion_reward
        }
        
        # 调试信息
        if self.debug_step_count <= 5:
            print(f"[REWARDS] Breakdown (env 0):")
            print(f"  Position tracking: {position_tracking[0].item():.4f}")
            print(f"  Velocity regulation: {velocity_regulation[0].item():.4f}")
            print(f"  CBF value: {cbf_values[0].item():.4f}")
            print(f"  Robot control penalty: {robot_control_penalty[0].item():.4f}")
            print(f"  Human force penalty: {human_force_penalty[0].item():.4f}")
            print(f"  TOTAL Robot: {rewards['robot'][0].item():.4f}")
            print(f"  TOTAL Human: {rewards['human'][0].item():.4f}")
        
        self.extras["log"] = {
            "robot_reward": rewards["robot"].mean().item(),
            "human_reward": rewards["human"].mean().item(),
            "safety_distance": self.safety_distances_t.mean().item(),
        }
        
        return rewards
        
    def _get_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """确定终止条件"""
        # 暂时禁用约束违反终止条件，用于调试
        constraint_violated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # 暂时禁用
        fell_out = self.stylus_pos_t1[:, 2] < self.min_z_pos
        vel_exceeded = torch.any(torch.abs(self.stylus_vel_t1) > self.max_cartesian_vel, dim=1)
        
        joint_violated = torch.any(
            (self.joint_pos_t1[:, :3] < self.joint_lower_limits[:3] - 0.01) |
            (self.joint_pos_t1[:, :3] > self.joint_upper_limits[:3] + 0.01),
            dim=1
        )
        
        final_reached = self.trajectory_manager.is_final_setpoint_reached(self.stylus_pos_t1)
        
        # 调试信息
        if self.debug_step_count <= 5:
            print(f"[TERMINATION] Conditions at step {self.debug_step_count} (env 0):")
            print(f"  Stylus Z position: {self.stylus_pos_t1[0, 2].item():.4f} (min: {self.min_z_pos})")
            print(f"  Constraint violated: {self.is_violating_t[0].item()} (disabled for debugging)")
            print(f"  Fell out: {fell_out[0].item()}")
            print(f"  Velocity: {self.stylus_vel_t1[0].cpu().numpy()} (max: {self.max_cartesian_vel})")
            print(f"  Velocity exceeded: {vel_exceeded[0].item()}")
            print(f"  Joint positions: {self.joint_pos_t1[0, :3].cpu().numpy()}")
            print(f"  Joint violated: {joint_violated[0].item()}")
            print(f"  Final reached: {final_reached[0].item()}")
            print(f"  Episode length: {self.episode_length_buf[0].item()}/{self.max_episode_length}")
        
        terminated_condition = constraint_violated | fell_out | vel_exceeded | joint_violated | final_reached
        truncated_condition = self.episode_length_buf >= self.max_episode_length - 1
        
        if self.debug_step_count <= 5 and terminated_condition[0]:
            print(f"[TERMINATION] Episode terminated at step {self.debug_step_count}!")
        
        terminated = {agent: terminated_condition for agent in self.cfg.possible_agents}
        truncated = {agent: truncated_condition for agent in self.cfg.possible_agents}
        
        return terminated, truncated
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """重置环境"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        super()._reset_idx(env_ids)
        
        if self.stylus_body_idx is None:
            self._initialize_body_indices()
        
        num_resets = len(env_ids)
        
        # 使用稳定的初始配置
        joint_pos = torch.zeros((num_resets, 6), device=self.device)
        joint_pos[:, 0] = -0.96   # waist
        joint_pos[:, 1] = 0.0     # shoulder  
        joint_pos[:, 2] = 1.0     # elbow
        joint_pos[:, 3] = 0.0     # yaw
        joint_pos[:, 4] = 2.0944  # pitch
        joint_pos[:, 5] = 0.0     # roll
        
        joint_vel = torch.zeros((num_resets, 6), device=self.device)
        
        self._omni_robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # 重置状态
        for agent in self.cfg.possible_agents:
            self.agent_actions[agent][env_ids] = 0.0
        
        self.human_forces_t[env_ids] = 0.0
        self.robot_forces_t[env_ids] = 0.0
        self.safety_distances_t[env_ids] = 0.01  # 重置为1厘米安全距离
        self.is_violating_t[env_ids] = False
        
        self.trajectory_manager.reset_trajectory(env_ids)
        
        # 重置调试计数器
        if 0 in env_ids:
            self.debug_step_count = 0
            print(f"\n[RESET] Environment reset for env_ids: {env_ids.cpu().numpy()}")
            print(f"[RESET] Stable initial joint positions: {joint_pos[0].cpu().numpy()}")
        
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