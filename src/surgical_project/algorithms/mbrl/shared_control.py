# 论文对齐的手术机器人人机共享控制算法 - 重构版本，支持完整YAML配置
import torch
import yaml
import os
from typing import Dict, Any, Tuple
from pathlib import Path

from .actor_critic import SurgicalActorCritic
from ..utils import (
    ReplayBuffer, PaperCostFunction, HumanImpedanceModel, 
    AdaptiveSharedControl, OffPolicyTrainer, ControlBarrierFunction,
    extract_paper_state, create_augmented_state
)


class SharedControlTrainer:
    """论文对齐的共享控制训练器 - 重构版本，支持完整配置"""
    def __init__(self, env, agent_cfg: Dict[str, Any], log_dir: str = None):
        self.env = env
        self.device = torch.device(agent_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.num_envs = getattr(env, 'num_envs', 1)
        self.log_dir = log_dir or "logs"
        
        # 直接使用agent_cfg，不再加载额外配置文件
        self.config = agent_cfg
        
        # 从配置中获取网络架构参数
        network_cfg = self.config.get('network', {})
        
        # 论文标准维度
        state_dim = self.config.get('state_space_dim', 9)
        action_dim = self.config.get('action_space_dim', 3)
        augmented_state_dim = self.config.get('augmented_state_dim', 12)
        
        # 初始化网络
        self.policy = SurgicalActorCritic(
            state_dim=state_dim,
            action_dim=action_dim, 
            augmented_state_dim=augmented_state_dim,
            network_cfg=network_cfg  # 传递网络配置
        ).to(self.device)
        
        # 初始化组件 - 从配置获取所有参数
        buffer_size = self.config.get('buffer_size', 10000)
        self.replay_buffer = ReplayBuffer(buffer_size, self.device)
        
        # 成本函数 - 使用配置参数
        self.cost_function = PaperCostFunction(
            Q1_weight=self.config.get('Q1_weight', 100.0),
            Q2_weight=self.config.get('Q2_weight', 0.01),
            Q3_weight=self.config.get('Q3_weight', 0.001),
            R_weight=self.config.get('R_weight', 0.001),
            cbf_weight=self.config.get('cbf_weight', 10.0),
            device=self.device
        )
        
        # 人体阻抗模型 - 使用配置参数
        self.human_dynamics = HumanImpedanceModel(
            device=self.device,
            damping_diag=self.config.get('human_damping_CH', [21.0, 21.0, 21.0]),
            stiffness_diag=self.config.get('human_stiffness_KH', [201.0, 201.0, 201.0])
        )
        
        # 共享控制 - 使用配置参数
        self.shared_control = AdaptiveSharedControl(
            robot_weight=self.config.get('robot_action_weight', 0.7),
            human_weight=self.config.get('human_action_weight', 0.3)
        )
        
        # Off-Policy训练器
        self.trainer = OffPolicyTrainer(self.policy, self.config, self.device)
        
        # CBF约束管理器 - 使用配置参数
        self.cbf = ControlBarrierFunction(
            gamma=self.config.get('cbf_gamma', 1.0),
            safety_margin=self.config.get('safety_margin', 0.002),
            device=self.device
        )
        
        # 初始化交互力
        self.interaction_forces = torch.zeros(self.num_envs, 3, device=self.device)
        
        # 人类平衡点 - 从配置获取
        equilibrium_points_cfg = self.config.get('equilibrium_points', [
            [0.0, 0.15, 0.03],  # 默认第一个平衡点
            [0.2, 0.15, 0.03]   # 默认第二个平衡点
        ])
        self.equilibrium_points = torch.tensor(
            equilibrium_points_cfg, device=self.device, dtype=torch.float32
        )
        
        # 训练参数
        self.save_frequency = self.config.get('save_frequency', 100)
        self.eval_frequency = self.config.get('eval_frequency', 50)
        self.log_frequency = self.config.get('log_frequency', 50)
        
        print(f"[INFO] SharedControlTrainer initialized:")
        print(f"  - Device: {self.device}")
        print(f"  - Network parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
        print(f"  - CBF gamma: {self.config.get('cbf_gamma', 1.0)}")
        print(f"  - Human equilibrium points: {self.equilibrium_points.cpu().numpy().tolist()}")
        print(f"  - Save frequency: {self.save_frequency}")
        print(f"  - Evaluation frequency: {self.eval_frequency}")
        
    def get_current_equilibrium_point(self, target_index: int) -> torch.Tensor:
        """根据当前目标索引获取对应的人类平衡点"""
        if target_index >= len(self.equilibrium_points):
            target_index = len(self.equilibrium_points) - 1
        return self.equilibrium_points[target_index]
    
    def train(self, total_steps: int):
        """主训练循环 - 增强版本，支持更多配置功能"""
        obs_dict, _ = self.env.reset()
        obs = obs_dict["policy"]
        
        step_rewards = []
        update_count = 0
        min_buffer_size = self.config.get('min_buffer_size', 1000)
        update_frequency = self.config.get('update_frequency', 20)
        
        print(f"[INFO] Starting training for {total_steps} steps")
        print(f"[INFO] Minimum buffer size: {min_buffer_size}")
        print(f"[INFO] Update frequency: {update_frequency}")
        
        for step in range(total_steps):
            try:
                # 提取论文状态
                paper_state, desired_pos = extract_paper_state(obs, self.interaction_forces)
                augmented_state = create_augmented_state(paper_state, desired_pos)
                
                # 获取机器人动作 - 使用配置的探索噪声
                with torch.no_grad():
                    robot_action = self.policy.get_action(
                        augmented_state, 
                        deterministic=False,
                        exploration_noise=self.config.get('exploration_noise', 0.01)
                    )
                    robot_action = torch.clamp(robot_action, -1.0, 1.0)
                
                # 获取当前目标索引
                # ===== 安全获取 trajectory_manager ====
                # 先尝试直接访问
                if hasattr(self.env, 'trajectory_manager'):
                    tm = self.env.trajectory_manager
                # 如果被封装在 wrapper 里，则 unwrap 一层
                elif hasattr(self.env, 'env') and hasattr(self.env.env, 'trajectory_manager'):
                    tm = self.env.env.trajectory_manager
                else:
                    tm = None

                if tm is not None:
                    current_target_index = tm.current_target_index
                else:
                    current_target_index = 0
                # ======================================
                current_equilibrium = self.get_current_equilibrium_point(current_target_index)
                
                # 计算人类动作
                current_pos = paper_state[..., :3]
                human_action = self.human_dynamics.compute_human_action(
                    current_pos, 
                    current_equilibrium.unsqueeze(0).expand(self.num_envs, -1),
                    dt=self.config.get('dt', 0.01)
                )
                
                # 共享控制融合
                final_action = self.shared_control.fuse_actions(
                    robot_action, human_action, self.interaction_forces
                )
                final_action = torch.clamp(final_action, -1.0, 1.0)
                
                # 环境步进
                next_obs_dict, env_reward, terminated, truncated, info = self.env.step(final_action)
                next_obs = next_obs_dict["policy"]
                done = terminated | truncated
                
                # 计算成本
                current_pos = paper_state[..., :3]
                current_vel = paper_state[..., 3:6]
                
                # 获取CBF值
                cbf_values = None
                if hasattr(self.env, 'safety_distances'):
                    cbf_values = self.cbf.compute_cbf_value(self.env.safety_distances)
                
                paper_cost = self.cost_function.compute_cost(
                    current_pos, desired_pos, current_vel, 
                    self.interaction_forces, final_action, cbf_values
                )
                paper_reward = -paper_cost
                
                # 更新交互力
                self.interaction_forces = (final_action - robot_action) * self.config.get('interaction_force_scale', 2.0)
                self.interaction_forces = torch.clamp(
                    self.interaction_forces, 
                    -self.config.get('max_interaction_force', 5.0), 
                    self.config.get('max_interaction_force', 5.0)
                )
                
                # 存储经验
                next_paper_state, next_desired_pos = extract_paper_state(next_obs, self.interaction_forces)
                for i in range(self.num_envs):
                    augmented_state_i = create_augmented_state(paper_state[i], desired_pos[i])
                    next_augmented_state_i = create_augmented_state(
                        next_paper_state[i], next_desired_pos[i]
                    )
                    
                    self.replay_buffer.add(
                        augmented_state_i,
                        final_action[i],
                        paper_reward[i],
                        next_augmented_state_i,
                        done[i]
                    )
                
                step_rewards.append(paper_reward.mean().item())
                
                # 网络更新
                if len(self.replay_buffer) > min_buffer_size and step % update_frequency == 0:
                    self.trainer.update_networks(self.replay_buffer)
                    update_count += 1
                
                obs = next_obs
                
                # 进度记录
                if step % self.log_frequency == 0:
                    avg_reward = sum(step_rewards[-self.log_frequency:]) / min(self.log_frequency, len(step_rewards))
                    print(f"Step {step:5d} | Reward: {avg_reward:.3f} | "
                          f"Buffer: {len(self.replay_buffer)} | Updates: {update_count}")
                    
                    # 详细日志
                    if hasattr(self.env, 'extras') and 'log' in self.env.extras:
                        log_info = self.env.extras['log']
                        print(f"  Target: {current_target_index} | "
                              f"Progress: {log_info.get('trajectory_progress', 0):.2f} | "
                              f"Safety: {log_info.get('safety_distance', 0):.4f}")
                
                # 定期保存检查点
                if step > 0 and step % self.save_frequency == 0:
                    self._save_checkpoint(step)
                
                # 定期评估
                if step > 0 and step % self.eval_frequency == 0:
                    self._run_evaluation(step)
                
            except Exception as e:
                print(f"[ERROR] Training step {step} failed: {e}")
                obs_dict, _ = self.env.reset()
                obs = obs_dict["policy"]
                continue
        
        print(f"[INFO] Training completed. Total updates: {update_count}")
        print(f"[INFO] Final average reward: {sum(step_rewards[-100:]) / min(100, len(step_rewards)):.3f}")
    
    def _save_checkpoint(self, step: int):
        """保存检查点"""
        if not self.log_dir:
            return
            
        checkpoint_path = os.path.join(self.log_dir, "checkpoints", f"checkpoint_step_{step}.pth")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        try:
            checkpoint_data = {
                'policy_state_dict': self.policy.state_dict(),
                'actor_optimizer_state_dict': self.trainer.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.trainer.critic_optimizer.state_dict(),
                'identifier_optimizer_state_dict': self.trainer.identifier_optimizer.state_dict(),
                'training_step': step,
                'config': self.config,
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            print(f"[INFO] Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}")
    
    def _run_evaluation(self, step: int):
        """运行评估"""
        eval_episodes = self.config.get('eval_episodes', 3)
        max_eval_steps = self.config.get('max_eval_steps', 500)
        
        eval_rewards = []
        print(f"[INFO] Running evaluation at step {step}...")
        
        for eval_ep in range(eval_episodes):
            obs_dict, _ = self.env.reset()
            ep_reward = 0
            eval_step_count = 0
            
            while eval_step_count < max_eval_steps:
                obs = obs_dict["policy"]
                
                # 使用确定性策略评估
                paper_state, desired_pos = extract_paper_state(obs, self.interaction_forces)
                augmented_state = create_augmented_state(paper_state, desired_pos)
                
                with torch.no_grad():
                    action = self.policy.get_action(augmented_state, deterministic=True)
                
                obs_dict, reward, terminated, truncated, info = self.env.step(action)
                ep_reward += reward.mean().item()
                eval_step_count += 1
                
                if (terminated | truncated).any():
                    break
            
            eval_rewards.append(ep_reward)
        
        avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
        print(f"[INFO] Evaluation complete. Average reward: {avg_eval_reward:.3f}")
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'config': self.config,
        }, path)
        print(f"[INFO] Model saved to {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"[INFO] Model loaded from {path}")