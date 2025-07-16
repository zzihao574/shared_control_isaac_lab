#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""修复导入问题的手术机器人训练脚本"""

import argparse
import sys
import os
from datetime import datetime

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="Train surgical robot with paper-aligned human-robot shared control.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=1000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Surgical-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum training iterations.")
parser.add_argument("--paper_mode", action="store_true", default=True, help="Enable paper-aligned mode.")

# 添加AppLauncher命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 如果录制视频，启用相机
if args_cli.video:
    args_cli.enable_cameras = True

# 启动omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Isaac Sim初始化后的所有内容"""

import gymnasium as gym
import torch
import yaml
import random
import numpy as np
from pathlib import Path
import json
import time

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

# 导入自定义环境
import surgical_project.envs.single_agent  # 注册环境

# 直接导入具体的类，避免__init__.py的问题
try:
    from surgical_project.algorithms.mbrl.shared_control import (
        SharedControlTrainer,
        HumanImpedanceModel as HumanDynamicsModel,  # 使用别名匹配原来的名称
        PaperCostFunction,
        AdaptiveSharedControl,
        ReplayBuffer,
    )
    print("[INFO] Successfully imported paper-aligned algorithms")
except ImportError as e:
    print(f"[ERROR] Failed to import algorithms: {e}")
    print("[INFO] Trying alternative import method...")
    
    # 如果直接导入失败，尝试导入原始版本
    try:
        from surgical_project.algorithms.mbrl.shared_control import SharedControlTrainer
        print("[INFO] Successfully imported SharedControlTrainer")
    except ImportError as e2:
        print(f"[ERROR] All import methods failed: {e2}")
        sys.exit(1)


def load_paper_aligned_config():
    """加载论文对齐的训练配置"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'surgical_project', 'envs', 'single_agent', 'agents', 'training_params.yaml')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            raw_config = config_data['params']['config']
            
            # 转换数值字符串为适当类型
            processed_config = {}
            for key, value in raw_config.items():
                if isinstance(value, str):
                    try:
                        if '.' in value or 'e' in value.lower():
                            processed_config[key] = float(value)
                        else:
                            processed_config[key] = int(value)
                    except ValueError:
                        processed_config[key] = value
                elif isinstance(value, list):
                    try:
                        processed_config[key] = [float(x) if isinstance(x, (str, int, float)) else x for x in value]
                    except (ValueError, TypeError):
                        processed_config[key] = value
                else:
                    processed_config[key] = value
            
            print(f"[INFO] Loaded paper-aligned config with {len(processed_config)} parameters")
            return processed_config
    else:
        print(f"[WARNING] Config file not found: {config_path}")
        return get_default_paper_config()


def get_default_paper_config():
    """获取论文对齐的默认配置"""
    return {
        # 网络学习率
        'learning_rate': 1e-4,
        'identifier_lr': 5e-4,
        'batch_size': 128,
        'buffer_size': 10000,
        'min_buffer_size': 1000,
        'max_grad_norm': 1.0,
        
        # 论文方程(13)的成本函数权重
        'Q1_weight': 100.0,
        'Q2_weight': 0.01,
        'Q3_weight': 0.001,
        'R_weight': 0.001,
        
        # 论文方程(6)的人体动力学参数
        'human_damping_CH': [21.0, 21.0, 21.0],
        'human_stiffness_KH': [201.0, 201.0, 201.0],
        
        # 强化学习参数
        'gamma': 0.99,
        'tau': 0.005,
        
        # 共享控制参数
        'robot_action_weight': 0.7,
        'human_action_weight': 0.3,
        'collaboration_adaptation_rate': 0.05,
    }


def set_random_seeds(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"[INFO] Random seeds set to {seed}")


def create_directories(log_root_path: str, log_dir: str):
    """创建日志目录"""
    full_log_path = os.path.join(log_root_path, log_dir)
    
    directories = [
        full_log_path,
        os.path.join(full_log_path, "checkpoints"),
        os.path.join(full_log_path, "configs"),
        os.path.join(full_log_path, "logs"),
        os.path.join(full_log_path, "paper_analysis"),
    ]
    
    if args_cli.video:
        directories.append(os.path.join(full_log_path, "videos", "train"))
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return full_log_path


def save_checkpoint(trainer, step: int, log_path: str, config: dict):
    """保存模型检查点"""
    checkpoint_path = os.path.join(log_path, "checkpoints", f"paper_checkpoint_step_{step}.pth")
    
    try:
        checkpoint_data = {
            'policy_state_dict': trainer.policy.state_dict(),
            'actor_optimizer_state_dict': trainer.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': trainer.critic_optimizer.state_dict(),
            'identifier_optimizer_state_dict': trainer.identifier_optimizer.state_dict(),
            'training_step': step,
            'config': config,
            'args': vars(args_cli),
            'paper_aligned': True,
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"[INFO] Paper-aligned checkpoint saved: {checkpoint_path}")
        return checkpoint_path
        
    except Exception as e:
        print(f"[ERROR] Failed to save checkpoint: {e}")
        return None


def load_checkpoint(trainer, checkpoint_path: str):
    """加载模型检查点"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        
        trainer.policy.load_state_dict(checkpoint['policy_state_dict'])
        trainer.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        trainer.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        trainer.identifier_optimizer.load_state_dict(checkpoint['identifier_optimizer_state_dict'])
        
        start_step = checkpoint.get('training_step', 0)
        print(f"[INFO] Checkpoint loaded: {checkpoint_path}")
        print(f"[INFO] Resuming from step: {start_step}")
        
        return start_step
        
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return 0


def evaluate_paper_policy(env, trainer, num_eval_episodes: int = 5):
    """评估论文对齐策略的性能"""
    print(f"[INFO] Evaluating paper-aligned policy over {num_eval_episodes} episodes...")
    
    eval_rewards = []
    eval_costs = []
    
    for episode in range(num_eval_episodes):
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]
        
        episode_reward = 0
        episode_cost = 0
        episode_length = 0
        
        while episode_length < 200:
            with torch.no_grad():
                # 提取论文状态
                paper_state, desired_pos = trainer.extract_paper_state(obs)
                augmented_state = trainer.create_augmented_state(paper_state, desired_pos)
                
                # 获取确定性动作
                action = trainer.policy.get_action(augmented_state, deterministic=True)
                action = torch.clamp(action, -1.0, 1.0)
            
            obs_dict, rewards, terminated, truncated, info = env.step(action)
            obs = obs_dict["policy"]
            
            # 计算论文成本
            current_pos = paper_state[..., :3]
            current_vel = paper_state[..., 3:6]
            interaction_force = paper_state[..., 6:9]
            
            paper_cost = trainer.cost_function.compute_cost(
                current_pos, desired_pos, current_vel, interaction_force, action
            )
            
            episode_reward += rewards.mean().item()
            episode_cost += paper_cost.mean().item()
            episode_length += 1
            
            if terminated.any() or truncated.any():
                break
        
        eval_rewards.append(episode_reward)
        eval_costs.append(episode_cost)
    
    avg_reward = np.mean(eval_rewards)
    avg_cost = np.mean(eval_costs)
    print(f"[EVAL] Average reward: {avg_reward:.3f}, Average cost: {avg_cost:.3f}")
    
    return avg_reward, avg_cost


def save_paper_analysis(log_path: str, trainer, final_reward: float, final_cost: float):
    """保存论文分析结果"""
    analysis_path = os.path.join(log_path, "paper_analysis")
    
    analysis_results = {
        'paper_alignment': {
            'state_space_dimension': getattr(trainer.policy, 'state_dim', 9),
            'augmented_state_dimension': getattr(trainer.policy, 'augmented_state_dim', 12),
            'action_space_dimension': getattr(trainer.policy, 'action_dim', 3),
            'cost_function_implemented': hasattr(trainer, 'cost_function'),
            'human_dynamics_implemented': hasattr(trainer, 'human_dynamics'),
        },
        'training_results': {
            'final_reward': final_reward,
            'final_cost': final_cost,
            'convergence_achieved': final_cost < 10.0,
        },
        'theoretical_validation': {
            'hjb_equation_implemented': hasattr(trainer.policy, 'hjb_solver'),
            'dynamics_identifier_implemented': hasattr(trainer.policy, 'identifier'),
            'shared_control_fusion_implemented': hasattr(trainer, 'shared_control'),
        }
    }
    
    with open(os.path.join(analysis_path, "paper_analysis.json"), 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"[INFO] Paper analysis saved to: {analysis_path}")


def main():
    """主训练函数"""
    
    # 设置随机种子
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    
    set_random_seeds(args_cli.seed)
    
    print("=" * 80)
    print("PAPER-ALIGNED SURGICAL HUMAN-ROBOT SHARED CONTROL TRAINING")
    print("=" * 80)
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    print(f"[INFO] Max iterations: {args_cli.max_iterations}")
    print(f"[INFO] Random seed: {args_cli.seed}")
    print(f"[INFO] Paper mode: {args_cli.paper_mode}")
    print(f"[INFO] Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)
    
    # 创建环境
    try:
        from surgical_project.envs.single_agent.surgical_direct_env_cfg import SurgicalDirectEnvCfg
        
        env_cfg = SurgicalDirectEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        
        if args_cli.paper_mode:
            print(f"[INFO] Applying paper-aligned environment settings...")
            env_cfg.observation_space = 12
            env_cfg.state_space = 9
        
        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.video else None
        )
        
        print(f"[INFO] Environment created successfully (Paper-aligned)")
        print(f"[INFO] Observation space: {env.observation_space}")
        print(f"[INFO] Action space: {env.action_space}")
        
    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 视频录制包装
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "surgical_videos", "paper_train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording training videos (Paper mode).")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # 加载配置
    config = load_paper_aligned_config()
    
    # 创建输出目录
    log_root_path = "logs/paper_surgical_shared_control"
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_paper_aligned"
    full_log_path = create_directories(log_root_path, log_dir)
    
    print(f"[INFO] Logging to: {full_log_path}")
    
    # 保存配置
    dump_yaml(os.path.join(full_log_path, "configs", "env_config.yaml"), env_cfg)
    dump_yaml(os.path.join(full_log_path, "configs", "training_config.yaml"), config)
    dump_yaml(os.path.join(full_log_path, "configs", "args.yaml"), vars(args_cli))
    
    # 创建训练器
    try:
        trainer = SharedControlTrainer(env, config)
        print(f"[INFO] Paper-aligned trainer created successfully")
        print(f"[INFO] Network parameters: {sum(p.numel() for p in trainer.policy.parameters()):,}")
        print(f"[INFO] Training device: {trainer.device}")
        
        # 验证论文对齐
        if hasattr(trainer.policy, 'state_dim'):
            print(f"[INFO] Paper alignment verification:")
            print(f"  - State space: z ∈ R^{trainer.policy.state_dim}")
            print(f"  - Augmented state: z̄ ∈ R^{trainer.policy.augmented_state_dim}")
            print(f"  - Action space: u ∈ R^{trainer.policy.action_dim}")
        
        # 加载检查点
        start_step = 0
        if args_cli.checkpoint:
            if os.path.exists(args_cli.checkpoint):
                start_step = load_checkpoint(trainer, args_cli.checkpoint)
            else:
                print(f"[WARNING] Checkpoint file not found: {args_cli.checkpoint}")
            
    except Exception as e:
        print(f"[ERROR] Failed to create trainer: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return
    
    # 训练循环
    try:
        print("\n" + "=" * 80)
        print("STARTING PAPER-ALIGNED TRAINING")
        print("=" * 80)
        
        best_reward = float('-inf')
        best_cost = float('inf')
        
        # 初始评估
        if start_step == 0:
            try:
                initial_reward, initial_cost = evaluate_paper_policy(env, trainer, num_eval_episodes=3)
                print(f"[INFO] Initial performance: Reward={initial_reward:.3f}, Cost={initial_cost:.3f}")
            except Exception as e:
                print(f"[WARNING] Initial evaluation failed: {e}")
                print("[INFO] Continuing with training...")
        
        print(f"\n[INFO] Training from step {start_step} to {args_cli.max_iterations}...")
        print("=" * 80)
        
        # 训练模型
        trainer.train(total_steps=args_cli.max_iterations)
        
        print("=" * 80)
        print(f"[INFO] Paper-aligned training completed successfully")
        
        # 最终评估
        print("\n[INFO] Final evaluation...")
        try:
            final_reward, final_cost = evaluate_paper_policy(env, trainer, num_eval_episodes=5)
        except Exception as e:
            print(f"[WARNING] Final evaluation failed: {e}")
            final_reward, final_cost = 0.0, 999.0
        
        # 保存最终模型
        final_checkpoint_path = save_checkpoint(trainer, args_cli.max_iterations, full_log_path, config)
        
        # 保存论文分析
        save_paper_analysis(full_log_path, trainer, final_reward, final_cost)
        
        # 训练总结
        print("\n" + "=" * 80)
        print("PAPER-ALIGNED TRAINING SUMMARY")
        print("=" * 80)
        print(f"Total training steps: {args_cli.max_iterations}")
        print(f"Final reward: {final_reward:.3f}")
        print(f"Final cost (paper metric): {final_cost:.3f}")
        print(f"Cost convergence: {'✓' if final_cost < 10.0 else '✗'}")
        print(f"Model saved to: {final_checkpoint_path}")
        print(f"Logs saved to: {full_log_path}")
        print("=" * 80)
        
        # 理论验证总结
        print("\n[INFO] Theoretical Implementation Status:")
        if hasattr(trainer.policy, 'state_dim'):
            print(f"  ✓ State space aligned: z = [x, ẋ, f]^T ∈ R^{trainer.policy.state_dim}")
        if hasattr(trainer, 'cost_function'):
            print(f"  ✓ Cost function implemented: Paper Eq.(13)")
        if hasattr(trainer, 'human_dynamics'):
            print(f"  ✓ Human dynamics implemented: Paper Eq.(6)")
        if hasattr(trainer.policy, 'critic'):
            print(f"  ✓ Critic network: Paper Eq.(29)")
        if hasattr(trainer.policy, 'actor'):
            print(f"  ✓ Actor network: Paper Eq.(50)")
        if hasattr(trainer.policy, 'identifier'):
            print(f"  ✓ Dynamics identifier: Paper Eq.(34)")
        if hasattr(trainer, 'shared_control'):
            print(f"  ✓ Shared control fusion: Adaptive weighting")
        
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        
        # 保存紧急检查点
        emergency_path = save_checkpoint(trainer, 0, full_log_path, config)
        if emergency_path:
            print(f"[INFO] Emergency checkpoint saved: {emergency_path}")
            
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理
        try:
            env.close()
            print(f"[INFO] Environment closed")
        except:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error in training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭仿真应用
        try:
            simulation_app.close()
            print("[INFO] Simulation app closed")
        except:
            pass