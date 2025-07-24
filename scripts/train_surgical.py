#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""论文对齐的手术机器人训练脚本 - 简化版本（无Hydra）"""

import argparse
import sys
import os
import yaml
from datetime import datetime

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from isaaclab.app import AppLauncher

# 命令行参数
parser = argparse.ArgumentParser(description="Train surgical robot with paper-aligned human-robot shared control.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=1000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Surgical-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--max_iterations", type=int, default=5000, help="Maximum training iterations.")
parser.add_argument("--config", type=str, default=None, help="Path to configuration file.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Isaac Sim初始化后的所有内容"""

import gymnasium as gym
import torch
import random
import numpy as np
from pathlib import Path

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml, dump_pickle

# 导入自定义环境和算法
import surgical_project.envs.single_agent
from surgical_project.algorithms.mbrl.shared_control import SharedControlTrainer


def load_config(config_path: str = None) -> dict:
    """加载训练配置"""
    # 默认配置
    default_config = {
        'seed': 42,
        'device': 'cuda:0',
        'max_iterations': 5000,
        'learning_rate': 0.0001,
        'identifier_lr': 0.0005,
        'batch_size': 128,
        'buffer_size': 10000,
        'min_buffer_size': 1000,
        'update_frequency': 20,
        'save_frequency': 100,
        'eval_frequency': 50,
        'log_frequency': 50,
        'eval_episodes': 3,
        'max_eval_steps': 500,
        'max_grad_norm': 1.0,
        'gamma': 0.99,
        'tau': 0.005,
        'dt': 0.01,
        
        # 网络架构配置
        'network': {
            'actor': {
                'hidden_dims': [256, 128],
                'activation': 'relu',
                'output_activation': 'tanh'
            },
            'critic': {
                'hidden_dims': [256, 128],
                'activation': 'relu'
            },
            'identifier': {
                'hidden_dims': [128, 128],
                'activation': 'relu'
            },
            'initializer': {
                'name': 'orthogonal',
                'gain': 1.0,
                'identifier_gain': 0.1
            },
            'action_noise_std': 0.01
        },
        
        # 论文方程(13)的成本函数权重
        'Q1_weight': 100.0,
        'Q2_weight': 0.01,
        'Q3_weight': 0.001,
        'R_weight': 0.001,
        
        # 论文方程(6)的人体动力学参数
        'human_damping_CH': [21.0, 21.0, 21.0],
        'human_stiffness_KH': [201.0, 201.0, 201.0],
        
        # 论文共享控制参数
        'robot_action_weight': 0.7,
        'human_action_weight': 0.3,
        
        # CBF约束参数
        'cbf_gamma': 1.0,
        'cbf_weight': 10.0,
        'safety_margin': 0.002,
        
        # 交互力参数
        'interaction_force_scale': 2.0,
        'max_interaction_force': 5.0,
        
        # 探索设置
        'exploration_noise': 0.01,
        
        # 论文状态空间设置
        'state_space_dim': 9,
        'augmented_state_dim': 12,
        'action_space_dim': 3,
        
        # 人类平衡点设置
        'equilibrium_points': [
            [0.0, 0.15, 0.03],
            [0.2, 0.15, 0.03]
        ]
    }
    
    # 尝试加载配置文件
    if config_path is None:
        # 默认配置文件路径
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'src', 'surgical_project', 
                        'envs', 'single_agent', 'agents', 'training_params.yaml'),
            os.path.join(os.path.dirname(__file__), 'config', 'training_params.yaml'),
            os.path.join(os.getcwd(), 'training_params.yaml'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                
            # 递归更新配置
            def update_config(base, override):
                for key, value in override.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        update_config(base[key], value)
                    else:
                        base[key] = value
            
            update_config(default_config, yaml_config)
            print(f"[INFO] Loaded config from: {config_path}")
            
        except Exception as e:
            print(f"[WARNING] Failed to load config from {config_path}: {e}")
            print("[INFO] Using default configuration")
    else:
        print("[INFO] No config file found, using default configuration")
    
    return default_config


def set_random_seeds(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(trainer, step: int, log_path: str):
    """保存模型检查点"""
    checkpoint_path = os.path.join(log_path, "checkpoints", f"checkpoint_step_{step}.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    try:
        checkpoint_data = {
            'policy_state_dict': trainer.policy.state_dict(),
            'actor_optimizer_state_dict': trainer.trainer.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': trainer.trainer.critic_optimizer.state_dict(),
            'identifier_optimizer_state_dict': trainer.trainer.identifier_optimizer.state_dict(),
            'training_step': step,
            'config': trainer.config,
            'args': vars(args_cli),
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"[INFO] Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        print(f"[ERROR] Failed to save checkpoint: {e}")
        return None


def load_checkpoint(trainer, checkpoint_path: str):
    """加载模型检查点"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        trainer.policy.load_state_dict(checkpoint['policy_state_dict'])
        trainer.trainer.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        trainer.trainer.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        trainer.trainer.identifier_optimizer.load_state_dict(checkpoint['identifier_optimizer_state_dict'])
        
        start_step = checkpoint.get('training_step', 0)
        print(f"[INFO] Checkpoint loaded from step: {start_step}")
        return start_step
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return 0


def print_system_info():
    """打印系统信息"""
    print("=" * 80)
    print("PAPER-ALIGNED SURGICAL HUMAN-ROBOT SHARED CONTROL TRAINING")
    print("=" * 80)
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    print(f"[INFO] Max iterations: {args_cli.max_iterations}")
    print(f"[INFO] Random seed: {args_cli.seed}")
    print(f"[INFO] Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"[INFO] PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"[INFO] CUDA version: {torch.version.cuda}")
        print(f"[INFO] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("=" * 80)


def create_environment():
    """创建训练环境"""
    try:
        from surgical_project.envs.single_agent.surgical_direct_env_cfg import SurgicalDirectEnvCfg
        
        env_cfg = SurgicalDirectEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        
        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.video else None
        )
        
        print(f"[INFO] Environment created successfully")
        print(f"[INFO] Observation space: {env.observation_space}")
        print(f"[INFO] Action space: {env.action_space}")
        
        # 测试环境重置
        obs_dict, info = env.reset()
        obs_shape = obs_dict["policy"].shape
        print(f"[INFO] Observation shape: {obs_shape}")
        
        return env, env_cfg
        
    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def setup_video_recording(env):
    """设置视频录制"""
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "surgical_videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        print(f"[INFO] Video recording enabled: every {args_cli.video_interval} steps")
    
    return env


def setup_logging():
    """设置日志目录"""
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "surgical_shared_control"))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    
    return log_dir


def main():
    """主训练函数"""
    
    # 设置随机种子
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    set_random_seeds(args_cli.seed)
    
    # 加载配置
    agent_cfg = load_config(args_cli.config)
    
    # 命令行参数覆盖配置
    agent_cfg['seed'] = args_cli.seed
    agent_cfg['max_iterations'] = args_cli.max_iterations
    
    # 打印系统信息
    print_system_info()
    
    # 创建环境
    env, env_cfg = create_environment()
    if env is None:
        return
    
    # 设置视频录制
    env = setup_video_recording(env)
    
    # 设置日志
    log_dir = setup_logging()
    
    # 保存配置
    try:
        dump_yaml(os.path.join(log_dir, "params", "env_config.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent_config.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env_config.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent_config.pkl"), agent_cfg)
    except Exception as e:
        print(f"[WARNING] Failed to save configs: {e}")
    
    # 创建训练器
    try:
        trainer = SharedControlTrainer(env, agent_cfg, log_dir)
        print(f"[INFO] Trainer created successfully")
        print(f"[INFO] Network parameters: {sum(p.numel() for p in trainer.policy.parameters()):,}")
        print(f"[INFO] Training device: {trainer.device}")
        
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
        print("STARTING TRAINING")
        print("=" * 80)
        
        # 训练前验证
        print("[INFO] Running pre-training validation...")
        obs_dict, _ = env.reset()
        test_action = torch.zeros(args_cli.num_envs, 3)
        obs_dict, reward, terminated, truncated, info = env.step(test_action)
        print(f"[INFO] Environment validation passed")
        print(f"[INFO] Reward shape: {reward.shape}, Obs shape: {obs_dict['policy'].shape}")
        
        # 训练模型
        trainer.train(total_steps=agent_cfg['max_iterations'])
        
        print("=" * 80)
        print(f"[INFO] Training completed successfully")
        
        # 保存最终模型
        final_checkpoint = save_checkpoint(trainer, agent_cfg['max_iterations'], log_dir)
        
        # 保存简化模型（仅网络权重）
        if final_checkpoint:
            simple_model_path = os.path.join(log_dir, "final_model.pth")
            trainer.save_model(simple_model_path)
        
        print(f"[INFO] All outputs saved to: {log_dir}")
        
        # 训练后评估
        print("\n" + "=" * 80)
        print("POST-TRAINING EVALUATION")
        print("=" * 80)
        
        # 运行几个评估episode
        eval_rewards = []
        for eval_ep in range(agent_cfg.get('eval_episodes', 3)):
            obs_dict, _ = env.reset()
            ep_reward = 0
            step_count = 0
            
            while step_count < agent_cfg.get('max_eval_steps', 500):
                obs = obs_dict["policy"]
                
                # 使用训练好的策略
                from surgical_project.algorithms.utils import extract_paper_state, create_augmented_state
                paper_state, desired_pos = extract_paper_state(obs, trainer.interaction_forces)
                augmented_state = create_augmented_state(paper_state, desired_pos)
                
                with torch.no_grad():
                    action = trainer.policy.get_action(augmented_state, deterministic=True)
                
                obs_dict, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward.mean().item()
                step_count += 1
                
                if (terminated | truncated).any():
                    break
            
            eval_rewards.append(ep_reward)
            print(f"[INFO] Evaluation episode {eval_ep + 1}: {ep_reward:.3f} reward in {step_count} steps")
        
        avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
        print(f"[INFO] Average evaluation reward: {avg_eval_reward:.3f}")
        
        # 保存评估结果
        eval_results = {
            'eval_episodes': len(eval_rewards),
            'eval_rewards': eval_rewards,
            'average_reward': avg_eval_reward,
            'training_steps': agent_cfg['max_iterations'],
            'final_config': trainer.config
        }
        
        eval_path = os.path.join(log_dir, "evaluation_results.yaml")
        dump_yaml(eval_path, eval_results)
        
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        save_checkpoint(trainer, 0, log_dir)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.close()
        except:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            simulation_app.close()
        except:
            pass
        print("[INFO] Training script completed")