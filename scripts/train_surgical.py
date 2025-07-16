#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""论文对齐的手术机器人训练脚本"""

import argparse
import sys
import os
from datetime import datetime

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from isaaclab.app import AppLauncher

# 命令行参数
parser = argparse.ArgumentParser(description="Train surgical robot with paper-aligned human-robot shared control.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=1000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Surgical-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum training iterations.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

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

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

# 导入自定义环境和算法
import surgical_project.envs.single_agent
from surgical_project.algorithms.mbrl.shared_control import (
    SharedControlTrainer,
    HumanImpedanceModel as HumanDynamicsModel,
    PaperCostFunction,
    AdaptiveSharedControl,
    ReplayBuffer,
)

def load_paper_config():
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
                        processed_config[key] = float(value) if '.' in value or 'e' in value.lower() else int(value)
                    except ValueError:
                        processed_config[key] = value
                elif isinstance(value, list):
                    try:
                        processed_config[key] = [float(x) if isinstance(x, (str, int, float)) else x for x in value]
                    except (ValueError, TypeError):
                        processed_config[key] = value
                else:
                    processed_config[key] = value
            
            return processed_config
    
    # 默认论文配置
    return {
        'learning_rate': 1e-4,
        'identifier_lr': 5e-4,
        'batch_size': 128,
        'buffer_size': 10000,
        'min_buffer_size': 1000,
        'max_grad_norm': 1.0,
        'Q1_weight': 100.0,  # 论文方程(13)
        'Q2_weight': 0.01,
        'Q3_weight': 0.001,
        'R_weight': 0.001,
        'human_damping_CH': [21.0, 21.0, 21.0],  # 论文方程(6)
        'human_stiffness_KH': [201.0, 201.0, 201.0],
        'gamma': 0.99,
        'tau': 0.005,
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

def save_checkpoint(trainer, step: int, log_path: str, config: dict):
    """保存模型检查点"""
    checkpoint_path = os.path.join(log_path, "checkpoints", f"checkpoint_step_{step}.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    try:
        checkpoint_data = {
            'policy_state_dict': trainer.policy.state_dict(),
            'actor_optimizer_state_dict': trainer.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': trainer.critic_optimizer.state_dict(),
            'identifier_optimizer_state_dict': trainer.identifier_optimizer.state_dict(),
            'training_step': step,
            'config': config,
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
        trainer.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        trainer.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        trainer.identifier_optimizer.load_state_dict(checkpoint['identifier_optimizer_state_dict'])
        
        start_step = checkpoint.get('training_step', 0)
        print(f"[INFO] Checkpoint loaded from step: {start_step}")
        return start_step
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return 0

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
    print(f"[INFO] Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)
    
    # 创建环境
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
        
    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        return
    
    # 视频录制包装
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "surgical_videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # 加载配置和创建输出目录
    config = load_paper_config()
    log_root_path = "logs/surgical_shared_control"
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_log_path = os.path.join(log_root_path, log_dir)
    
    os.makedirs(os.path.join(full_log_path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(full_log_path, "configs"), exist_ok=True)
    
    print(f"[INFO] Logging to: {full_log_path}")
    
    # 保存配置
    dump_yaml(os.path.join(full_log_path, "configs", "env_config.yaml"), env_cfg)
    dump_yaml(os.path.join(full_log_path, "configs", "training_config.yaml"), config)
    dump_yaml(os.path.join(full_log_path, "configs", "args.yaml"), vars(args_cli))
    
    # 创建训练器
    try:
        trainer = SharedControlTrainer(env, config)
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
        env.close()
        return
    
    # 训练循环
    try:
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        
        # 训练模型
        trainer.train(total_steps=args_cli.max_iterations)
        
        print("=" * 80)
        print(f"[INFO] Training completed successfully")
        
        # 保存最终模型
        save_checkpoint(trainer, args_cli.max_iterations, full_log_path, config)
        
        print(f"[INFO] Logs saved to: {full_log_path}")
        
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        save_checkpoint(trainer, 0, full_log_path, config)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
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
    finally:
        try:
            simulation_app.close()
        except:
            pass