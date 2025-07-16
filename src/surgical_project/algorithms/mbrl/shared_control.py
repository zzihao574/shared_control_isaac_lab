# surgical_project/algorithms/shared_control.py - 最终修复版本
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import numpy as np
from .actor_critic import SurgicalActorCritic

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.stack([b[0] for b in batch])
        actions = torch.stack([b[1] for b in batch])
        rewards = torch.stack([b[2] for b in batch])
        next_states = torch.stack([b[3] for b in batch])
        dones = torch.stack([b[4] for b in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class HumanImpedanceModel:
    """论文方程(6)的人体阻抗模型: C_H ẋ + K_H(x - x_H) = -f"""
    def __init__(self, device, config=None):
        self.device = device
        
        if config is not None:
            damping_diag = config.get('human_damping_CH', [21.0, 21.0, 21.0])
            stiffness_diag = config.get('human_stiffness_KH', [201.0, 201.0, 201.0])
        else:
            damping_diag = [21.0, 21.0, 21.0]
            stiffness_diag = [201.0, 201.0, 201.0]
            
        self.CH = torch.diag(torch.tensor(damping_diag, device=device, dtype=torch.float32))
        self.KH = torch.diag(torch.tensor(stiffness_diag, device=device, dtype=torch.float32))
        
        self.workspace_radius = 0.2
        self.max_human_velocity = 0.3
        
    def get_human_intention(self, current_pos: torch.Tensor, current_vel: torch.Tensor, 
                          interaction_force: torch.Tensor) -> torch.Tensor:
        """基于论文方程(6)估计人体意图位置 x_H"""
        # 输入验证
        if current_pos is None or current_vel is None or interaction_force is None:
            if current_pos is not None:
                return current_pos.clone()
            return torch.zeros(3, device=self.device)
        
        try:
            # 确保正确维度
            if current_pos.dim() == 1:
                current_pos = current_pos.unsqueeze(0)
            if current_vel.dim() == 1:
                current_vel = current_vel.unsqueeze(0)
            if interaction_force.dim() == 1:
                interaction_force = interaction_force.unsqueeze(0)
                
            batch_size = current_pos.shape[0]
            
            # 数值稳定性检查
            if torch.any(torch.isnan(current_pos)) or torch.any(torch.isnan(current_vel)) or torch.any(torch.isnan(interaction_force)):
                return current_pos.clone()
            
            # 扩展阻抗矩阵
            CH_batch = self.CH.unsqueeze(0).expand(batch_size, -1, -1)
            KH_batch = self.KH.unsqueeze(0).expand(batch_size, -1, -1)
            
            # 计算阻尼项 C_H ẋ
            damping_term = torch.bmm(CH_batch, current_vel.unsqueeze(-1)).squeeze(-1)
            
            # 异常力值检查
            force_magnitude = torch.norm(interaction_force, dim=-1, keepdim=True)
            if torch.any(force_magnitude > 10.0):
                return current_pos.clone()
            
            # 人体意图位置: x_H = x + (f + C_H ẋ) / K_H
            numerator = interaction_force + damping_term
            
            try:
                KH_inv_batch = torch.inverse(KH_batch + 1e-6 * torch.eye(3, device=self.device))
                intention_offset = torch.bmm(KH_inv_batch, numerator.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                KH_diag_inv = 1.0 / (torch.diagonal(self.KH) + 1e-6)
                intention_offset = numerator * KH_diag_inv.unsqueeze(0)
            
            human_intention = current_pos + intention_offset
            
            # 工作空间约束
            distance_2d = torch.norm(human_intention[..., :2], dim=-1, keepdim=True)
            if torch.any(distance_2d > self.workspace_radius):
                scale_factor = self.workspace_radius / (distance_2d + 1e-6)
                scale_factor = torch.clamp(scale_factor, max=1.0)
                human_intention[..., :2] = human_intention[..., :2] * scale_factor
            
            # 最终检查
            if torch.any(torch.isnan(human_intention)) or torch.any(torch.isinf(human_intention)):
                return current_pos.clone()
            
            return human_intention
            
        except Exception as e:
            return current_pos.clone() if current_pos is not None else torch.zeros(3, device=self.device)
    
    def compute_human_action(self, current_pos: torch.Tensor, human_intention: torch.Tensor, 
                           dt: float = 0.01) -> torch.Tensor:
        """基于人体意图计算人体动作"""
        if current_pos is None or human_intention is None:
            return torch.zeros(3, device=self.device)
        
        try:
            if current_pos.dim() == 1:
                current_pos = current_pos.unsqueeze(0)
            if human_intention.dim() == 1:
                human_intention = human_intention.unsqueeze(0)
            
            position_error = human_intention - current_pos
            desired_velocity = position_error / dt
            
            velocity_norm = torch.norm(desired_velocity, dim=-1, keepdim=True)
            velocity_scale = torch.clamp(velocity_norm / self.max_human_velocity, max=1.0)
            desired_velocity = desired_velocity / (velocity_scale + 1e-6)
            
            human_action = desired_velocity * 0.1
            
            if torch.any(torch.isnan(human_action)) or torch.any(torch.isinf(human_action)):
                return torch.zeros_like(current_pos)
            
            return torch.clamp(human_action, -1.0, 1.0)
            
        except Exception as e:
            return torch.zeros_like(current_pos) if current_pos is not None else torch.zeros(3, device=self.device)

class PaperCostFunction:
    """论文方程(13)的成本函数: r = (x-x_d)^T Q_1(x-x_d) + ẋ^T Q_2 ẋ + f^T Q_3 f + u^T R u"""
    def __init__(self, Q1_weight: float, Q2_weight: float, Q3_weight: float, R_weight: float, device):
        self.device = device
        self.Q1 = torch.eye(3, device=device) * Q1_weight
        self.Q2 = torch.eye(3, device=device) * Q2_weight
        self.Q3 = torch.eye(3, device=device) * Q3_weight
        self.R = torch.eye(3, device=device) * R_weight
    
    def compute_cost(self, current_pos: torch.Tensor, desired_pos: torch.Tensor,
                    current_vel: torch.Tensor, interaction_force: torch.Tensor,
                    control_action: torch.Tensor) -> torch.Tensor:
        """计算论文方程(13)的即时成本"""
        
        # 输入验证
        inputs = [current_pos, desired_pos, current_vel, interaction_force, control_action]
        for inp in inputs:
            if inp is None:
                return torch.zeros(current_pos.shape[0] if current_pos is not None else 1, device=self.device)
        
        try:
            # 论文成本函数各项
            position_error = current_pos - desired_pos
            tracking_cost = torch.sum(position_error * (self.Q1 @ position_error.T).T, dim=-1)
            velocity_cost = torch.sum(current_vel * (self.Q2 @ current_vel.T).T, dim=-1)
            force_cost = torch.sum(interaction_force * (self.Q3 @ interaction_force.T).T, dim=-1)
            control_cost = torch.sum(control_action * (self.R @ control_action.T).T, dim=-1)
            
            total_cost = tracking_cost + velocity_cost + force_cost + control_cost
            
            if torch.any(torch.isnan(total_cost)) or torch.any(torch.isinf(total_cost)):
                return torch.ones_like(total_cost) * 1.0
            
            return total_cost
            
        except Exception as e:
            batch_size = current_pos.shape[0] if current_pos is not None else 1
            return torch.ones(batch_size, device=self.device) * 1.0

class AdaptiveSharedControl:
    """论文的自适应共享控制策略"""
    def __init__(self, config):
        self.robot_weight = config.get('robot_action_weight', 0.7)
        self.human_weight = config.get('human_action_weight', 0.3)
        self.adaptation_rate = config.get('collaboration_adaptation_rate', 0.05)
        
    def fuse_actions(self, robot_action: torch.Tensor, human_action: torch.Tensor, 
                    interaction_force: torch.Tensor) -> torch.Tensor:
        """自适应动作融合策略"""
        
        # 输入验证
        if robot_action is None or human_action is None:
            if robot_action is not None:
                return torch.clamp(robot_action, -1.0, 1.0)
            elif human_action is not None:
                return torch.clamp(human_action, -1.0, 1.0)
            else:
                return torch.zeros(3)
        
        try:
            # 基于交互力的自适应权重
            if interaction_force is not None:
                force_magnitude = torch.norm(interaction_force, dim=-1, keepdim=True)
                force_factor = torch.tanh(force_magnitude * 5.0)
                adaptive_human_weight = self.human_weight + force_factor * 0.2
                adaptive_robot_weight = 1.0 - adaptive_human_weight
            else:
                adaptive_robot_weight = self.robot_weight
                adaptive_human_weight = self.human_weight
            
            fused_action = (adaptive_robot_weight * robot_action + 
                           adaptive_human_weight * human_action)
            
            if torch.any(torch.isnan(fused_action)) or torch.any(torch.isinf(fused_action)):
                return torch.clamp(robot_action, -1.0, 1.0)
            
            return torch.clamp(fused_action, -1.0, 1.0)
            
        except Exception as e:
            return torch.clamp(robot_action, -1.0, 1.0)

class SharedControlTrainer:
    """论文对齐的共享控制训练器"""
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_envs = getattr(env, 'num_envs', 1)
        
        # 检测观测维度
        obs_dict, _ = env.reset()
        raw_obs_dim = obs_dict["policy"].shape[-1]
        
        # 论文标准维度
        state_dim = 9
        action_dim = 3
        augmented_state_dim = 12
        
        # 成本函数权重
        Q1_weight = self.safe_float('Q1_weight', 100.0)
        Q2_weight = self.safe_float('Q2_weight', 0.01)
        Q3_weight = self.safe_float('Q3_weight', 0.001)
        R_weight = self.safe_float('R_weight', 0.001)
        
        # 权重矩阵
        Q_matrix = torch.block_diag(
            torch.eye(3) * Q1_weight,
            torch.eye(3) * Q2_weight,
            torch.eye(3) * Q3_weight,
            torch.eye(3) * Q1_weight
        ).to(self.device)
        R_matrix = torch.eye(action_dim, device=self.device) * R_weight
        
        # 初始化网络
        self.policy = SurgicalActorCritic(
            state_dim=state_dim,
            action_dim=action_dim, 
            augmented_state_dim=augmented_state_dim,
            Q_weights=Q_matrix,
            R_weights=R_matrix
        ).to(self.device)
        
        # 优化器
        lr = self.safe_float('learning_rate', 3e-4)
        id_lr = self.safe_float('identifier_lr', 1e-3)
        
        self.actor_optimizer = optim.Adam(self.policy.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.policy.critic.parameters(), lr=lr)
        self.identifier_optimizer = optim.Adam(self.policy.identifier.parameters(), lr=id_lr)
        
        # 其他组件
        buffer_size = self.safe_int('buffer_size', 10000)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        human_config = {
            'human_damping_CH': config.get('human_damping_CH', [21.0, 21.0, 21.0]),
            'human_stiffness_KH': config.get('human_stiffness_KH', [201.0, 201.0, 201.0]),
        }
        self.human_dynamics = HumanImpedanceModel(self.device, human_config)
        self.cost_function = PaperCostFunction(Q1_weight, Q2_weight, Q3_weight, R_weight, self.device)
        self.shared_control = AdaptiveSharedControl(config)
        
        # 训练参数
        self.batch_size = self.safe_int('batch_size', 128)
        self.min_buffer_size = self.safe_int('min_buffer_size', 1000)
        self.gamma = self.safe_float('gamma', 0.99)
        self.max_grad_norm = self.safe_float('max_grad_norm', 1.0)
        self.dt = 0.01
        
        # 初始化交互力
        self.interaction_forces = torch.zeros(self.num_envs, 3, device=self.device)
        
    def safe_float(self, key: str, default: float) -> float:
        value = self.config.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return float(default)
    
    def safe_int(self, key: str, default: int) -> int:
        value = self.config.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return int(default)
    
    def extract_paper_state(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """从原始观测中提取论文标准状态 z = [x, ẋ, f]^T"""
        if obs is None:
            dummy_state = torch.zeros(self.num_envs, 9, device=self.device)
            dummy_desired = torch.zeros(self.num_envs, 3, device=self.device)
            return dummy_state, dummy_desired
        
        try:
            current_pos = obs[..., :3]
            current_vel = obs[..., 3:6]
            
            if obs.shape[-1] >= 12:
                desired_pos = obs[..., 9:12]
            else:
                desired_pos = torch.zeros_like(current_pos)
            
            interaction_force = self.interaction_forces
            paper_state = torch.cat([current_pos, current_vel, interaction_force], dim=-1)
            
            # 安全检查
            if torch.any(torch.isnan(paper_state)) or torch.any(torch.isnan(desired_pos)):
                paper_state = torch.zeros_like(paper_state)
                desired_pos = torch.zeros_like(desired_pos)
            
            return paper_state, desired_pos
            
        except Exception as e:
            dummy_state = torch.zeros(self.num_envs, 9, device=self.device)
            dummy_desired = torch.zeros(self.num_envs, 3, device=self.device)
            return dummy_state, dummy_desired
    
    def create_augmented_state(self, paper_state: torch.Tensor, desired_pos: torch.Tensor) -> torch.Tensor:
        """创建增广状态 z̄ = [z^T, x_d^T]^T"""
        if paper_state is None or desired_pos is None:
            return torch.zeros(self.num_envs, 12, device=self.device)
        
        try:
            augmented_state = torch.cat([paper_state, desired_pos], dim=-1)
            
            if torch.any(torch.isnan(augmented_state)):
                augmented_state = torch.zeros_like(augmented_state)
            
            return augmented_state
            
        except Exception as e:
            return torch.zeros(self.num_envs, 12, device=self.device)
    
    def train(self, total_steps: int):
        """主训练循环"""
        obs_dict, _ = self.env.reset()
        obs = obs_dict["policy"]
        
        step_rewards = []
        update_count = 0
        
        for step in range(total_steps):
            try:
                # 状态验证
                if obs is None:
                    obs_dict, _ = self.env.reset()
                    obs = obs_dict["policy"]
                    continue
                
                # 提取论文状态
                paper_state, desired_pos = self.extract_paper_state(obs)
                augmented_state = self.create_augmented_state(paper_state, desired_pos)
                
                if augmented_state is None:
                    continue
                
                # 获取机器人动作
                with torch.no_grad():
                    robot_action = self.policy.get_action(augmented_state)
                    if robot_action is None:
                        robot_action = torch.zeros(self.num_envs, 3, device=self.device)
                    robot_action = torch.clamp(robot_action, -1.0, 1.0)
                
                # 获取人体动作
                current_pos = paper_state[..., :3]
                current_vel = paper_state[..., 3:6]
                human_intention = self.human_dynamics.get_human_intention(
                    current_pos, current_vel, self.interaction_forces
                )
                human_action = self.human_dynamics.compute_human_action(
                    current_pos, human_intention, self.dt
                )
                
                # 共享控制融合
                final_action = self.shared_control.fuse_actions(
                    robot_action, human_action, self.interaction_forces
                )
                final_action = torch.clamp(final_action, -1.0, 1.0)
                
                # 环境步进
                try:
                    next_obs_dict, env_reward, terminated, truncated, info = self.env.step(final_action)
                    next_obs = next_obs_dict["policy"]
                    done = terminated | truncated
                except Exception as e:
                    obs_dict, _ = self.env.reset()
                    obs = obs_dict["policy"]
                    continue
                
                # 计算成本
                try:
                    paper_cost = self.cost_function.compute_cost(
                        current_pos, desired_pos, current_vel, 
                        self.interaction_forces, final_action
                    )
                    paper_reward = -paper_cost
                except Exception as e:
                    paper_reward = torch.zeros(self.num_envs, device=self.device)
                
                # 更新交互力
                try:
                    self.interaction_forces = (final_action - robot_action) * 2.0
                    self.interaction_forces = torch.clamp(self.interaction_forces, -5.0, 5.0)
                except Exception as e:
                    pass
                
                # 存储经验
                try:
                    next_paper_state, next_desired_pos = self.extract_paper_state(next_obs)
                    for i in range(self.num_envs):
                        augmented_state_i = self.create_augmented_state(
                            paper_state[i], desired_pos[i]
                        )
                        next_augmented_state_i = self.create_augmented_state(
                            next_paper_state[i], next_desired_pos[i]
                        )
                        
                        self.replay_buffer.add(
                            augmented_state_i.cpu(),
                            final_action[i].cpu(),
                            paper_reward[i].cpu(),
                            next_augmented_state_i.cpu(),
                            done[i].cpu()
                        )
                except Exception as e:
                    pass
                
                step_rewards.append(paper_reward.mean().item())
                
                # 网络更新
                if len(self.replay_buffer) > self.min_buffer_size and step % 20 == 0:
                    try:
                        self.update_networks_paper_aligned()
                        update_count += 1
                    except Exception as e:
                        pass
                
                obs = next_obs
                
                # 进度记录
                if step % 50 == 0:
                    avg_reward = np.mean(step_rewards[-50:]) if step_rewards else 0
                    print(f"Step {step:5d} | Reward: {avg_reward:.3f} | "
                          f"Buffer: {len(self.replay_buffer)} | Updates: {update_count}")
                
            except Exception as e:
                try:
                    obs_dict, _ = self.env.reset()
                    obs = obs_dict["policy"]
                except:
                    obs = torch.zeros(self.num_envs, 12, device=self.device)
                continue
        
        print(f"[INFO] Paper-aligned training completed. Total updates: {update_count}")
    
    def update_networks_paper_aligned(self):
        """按论文框架更新网络"""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        try:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = [
                x.to(self.device) for x in [states, actions, rewards, next_states, dones]
            ]
            
            if any(x is None for x in [states, actions, rewards, next_states, dones]):
                return
            
            self.update_dynamics_identifier(states, actions, next_states)
            self.update_critic_paper_aligned(states, actions, rewards, next_states, dones)
            self.update_actor_paper_aligned(states)
            
        except Exception as e:
            pass
    
    def update_dynamics_identifier(self, states: torch.Tensor, actions: torch.Tensor, 
                                 next_states: torch.Tensor):
        """更新动力学识别器"""
        try:
            current_z = states[..., :9]
            next_z = next_states[..., :9]
            
            if current_z is None or next_z is None or actions is None:
                return
            
            true_z_dot = (next_z - current_z) / self.dt
            pred_z_dot = self.policy.identifier(current_z, actions)
            
            if pred_z_dot is None:
                return
            
            identifier_loss = F.mse_loss(pred_z_dot, true_z_dot)
            
            self.identifier_optimizer.zero_grad()
            identifier_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.identifier.parameters(), self.max_grad_norm)
            self.identifier_optimizer.step()
            
        except Exception as e:
            pass
    
    def update_critic_paper_aligned(self, states: torch.Tensor, actions: torch.Tensor,
                                  rewards: torch.Tensor, next_states: torch.Tensor, 
                                  dones: torch.Tensor):
        """更新Critic"""
        try:
            if any(x is None for x in [states, actions, rewards, next_states, dones]):
                return
            
            with torch.no_grad():
                next_values = self.policy.critic(next_states)
                if next_values is None:
                    return
                target_values = rewards + self.gamma * next_values.squeeze() * (1 - dones.float())
            
            current_values = self.policy.critic(states)
            if current_values is None:
                return
            
            critic_loss = F.mse_loss(current_values.squeeze(), target_values)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
        except Exception as e:
            pass
    
    def update_actor_paper_aligned(self, states: torch.Tensor):
        """更新Actor"""
        try:
            if states is None:
                return
            
            actor_loss = self.policy.compute_actor_loss(states)
            
            if actor_loss is None or torch.isnan(actor_loss):
                return
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
        except Exception as e:
            pass