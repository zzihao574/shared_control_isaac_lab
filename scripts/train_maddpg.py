#!/usr/bin/env python3

"""
Surgical Robot MADDPG Shared Network Training
FINAL VERSION: Refactored with TrainingRunner and MilestoneEvaluator for better maintainability.
MODIFIED: Removed all try/except from training loop except KeyboardInterrupt
MODIFIED: Simplified environment count detection logic
MODIFIED: Removed try/except from cleanup operations

Features:
- TrainingRunner: 统一训练循环管理
- MilestoneEvaluator: 统一里程碑评估管理  
- 主脚本仅负责装配与启动
- 保持所有原有功能不变
"""

import sys
import os
import torch
import numpy as np
import random
from datetime import datetime
from typing import Dict, Any, Tuple, List
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from isaaclab.app import AppLauncher
from utils.training_helpers import (
    WandBLogger, TrainingConfiguration, TrainingLogger, create_argument_parser, 
    MetricsHub, TopKModelManager, TrainingRunner, MilestoneEvaluator, save_final_shared_networks
)


def create_env_and_config(args, config):
    """创建环境和环境配置"""
    from surgical_project.envs.multi_agent.surgical_direct_marl_env_cfg import SurgicalDirectMARLEnvCfg
    import gymnasium as gym
    import surgical_project.envs.multi_agent
    
    env_cfg = SurgicalDirectMARLEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    
    print(f"[INFO] Environment configuration:")
    print(f"  Number of environments: {env_cfg.scene.num_envs}")
    print(f"  Episode length: {env_cfg.episode_length_s}s")
    print(f"  Decimation: {env_cfg.decimation}")
    print(f"  Possible agents: {env_cfg.possible_agents}")
    print(f"  Action spaces: {env_cfg.action_spaces}")
    print(f"  Observation spaces: {env_cfg.observation_spaces}")
    
    env = gym.make(args.task, cfg=env_cfg)
    
    if hasattr(env, 'max_episode_length'):
        print(f"[INFO] Environment max_episode_length: {env.max_episode_length}")
    
    return env, env_cfg


def create_maddpg_trainer(env, config, args):
    """创建MADDPG训练器"""
    from surgical_project.algorithms.marl.maddpg import MADDPG
    
    device = config.get_compute_device()
    
    # 获取环境数量 - 简化逻辑，优先unwrapped.num_envs，否则env.num_envs
    num_envs = args.num_envs
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'num_envs'):
        num_envs = env.unwrapped.num_envs
    elif hasattr(env, 'num_envs'):
        num_envs = env.num_envs

    maddpg = MADDPG(
        num_envs=num_envs,
        env=env,
        params=config.params,
        device=device
    )
    
    print(f"[INFO] MADDPG trainer created:")
    print(f"  Device: {device}")
    print(f"  Agent IDs: {maddpg.agent_ids}")
    print(f"  Environments: {num_envs}")
    
    return maddpg


def setup_reward_logger(env, config, metrics_hub):
    """配置奖励日志器"""
    # 获取现有的reward_logger
    reward_logger = None
    if hasattr(env, 'reward_logger'):
        reward_logger = env.reward_logger
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'reward_logger'):
        reward_logger = env.unwrapped.reward_logger
    
    if not reward_logger:
        print("[INFO] No existing reward_logger found, will create a new one")
        return None
        
    # 获取环境数量
    num_envs = getattr(env, 'num_envs', getattr(getattr(env, 'unwrapped', env), 'num_envs', 512))
    
    # 创建新的RewardLogger
    from surgical_project.envs.multi_agent.utils import RewardLogger
    milestones = config.params.get('training_monitor', {}).get('milestone_episodes', [])
    
    new_logger = RewardLogger(
        num_envs=num_envs,
        device=config.get_compute_device(),
        metrics_hub=metrics_hub,
        enable_console_logging=config.params.get('logging', {}).get('enable_console_logging', False),
        milestones=milestones
    )
    
    # 替换环境中的logger
    if hasattr(env, 'reward_logger'):
        env.reward_logger = new_logger
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'reward_logger'):
        env.unwrapped.reward_logger = new_logger
    
    # 配置新的logger
    new_logger.configure_logging(config.params)
    
    print(f"[INFO] RewardLogger configured:")
    print(f"  Console logging: {'enabled' if new_logger.step_tracer.enable_console_logging else 'disabled'}")
    print(f"  Milestone management: Legacy compatibility mode")
    
    return new_logger


def create_milestone_callback(evaluator, runner):
    """创建里程碑回调函数"""
    def _on_milestone(milestone):
        """里程碑回调：评估→TopK→跳过episode计数"""
        print(f"[MILESTONE CALLBACK] Processing milestone {milestone}")
        result = evaluator.handle(milestone, runner.global_step)
        if result.get("skip_episode_once", False):
            runner.mark_skip_episode_once()
            print(f"[MILESTONE CALLBACK] Marked skip_episode_once for milestone {milestone}")
    return _on_milestone


class MADDPGTrainer:
    """精简的MADDPG训练器 - 仅负责装配与启动"""
    
    def __init__(self, args):
        self.args = args
        
        print(f"[TRAINER] Initializing MADDPGTrainer...")
        
        # 装配阶段
        self._setup_configuration()
        self._setup_environment()
        self._setup_logging_and_wandb()
        self._setup_training_components()
        self._setup_runners_and_evaluators()
        self._setup_milestone_management()
        
        print(f"[TRAINER] MADDPGTrainer initialized successfully")
        print(f"  Max global steps: {self.max_global_steps}")
        print(f"  Milestone episodes: {self.milestone_episodes}")

    def _setup_configuration(self):
        """设置配置"""
        print(f"[SETUP] Loading configuration from: {self.args.config}")
        self.config = TrainingConfiguration.from_yaml(self.args.config)
        
        # 设置随机种子
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        self.config.params['seed'] = self.args.seed
        
        print(f"[SETUP] Configuration loaded, seed set to: {self.args.seed}")

    def _setup_environment(self):
        """设置环境"""
        print(f"[SETUP] Creating environment: {self.args.task}")
        self.env, self.env_cfg = create_env_and_config(self.args, self.config)
        
        # 注入配置到环境
        actual_env = getattr(self.env, 'unwrapped', self.env)
        if hasattr(actual_env, "params"):
            actual_env.params = self.config.params
            print(f"[SETUP] Injected configuration parameters to environment")
        else:
            print("[WARNING] Environment does not support config injection")

    def _setup_logging_and_wandb(self):
        """设置日志和WandB"""
        # 创建日志目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/maddpg_shared/{timestamp}"
        
        self.logger = TrainingLogger(log_dir)
        self.checkpoint_path = os.path.join(log_dir, f"checkpoint_top{self.args.top_k_models}.pth")
        
        print(f"[SETUP] Log directory created: {log_dir}")
        
        # 初始化WandB - 不再用try/except包裹
        self.wandb_logger = WandBLogger(enabled=self.args.wandb)
        if self.wandb_logger.enabled:
            run_config = {**vars(self.args), **self.config.params}
            run_name = f"maddpg_shared_{self.args.num_envs}envs_{timestamp}"
            self.wandb_logger.initialize_run(run_config, run_name)
            print(f"[SETUP] WandB initialized with run name: {run_name}")
        else:
            print(f"[SETUP] WandB disabled")

    def _setup_training_components(self):
        """设置训练组件"""
        print(f"[SETUP] Setting up training components...")
        
        # 创建MetricsHub
        self.metrics_hub = MetricsHub()
        
        # 连接WandB到MetricsHub
        if self.wandb_logger.enabled:
            self.wandb_logger.attach_metrics_hub(self.metrics_hub)
            print(f"[SETUP] WandB attached to MetricsHub")
        
        # 创建MADDPG训练器
        self.maddpg = create_maddpg_trainer(self.env, self.config, self.args)
        
        # 设置奖励日志器
        self.reward_logger = setup_reward_logger(self.env, self.config, self.metrics_hub)
        
        # 创建TopK管理器
        self.top_k_manager = TopKModelManager(k=self.args.top_k_models, mode="max")
        
        # 设置训练参数
        maddpg_cfg = self.config.params.get('maddpg_config', {})
        cfg_steps = maddpg_cfg.get('max_global_steps', 0) or 0
        cli_steps = getattr(self.args, "max_global_steps", 0) or 0
        self.max_global_steps = int(cli_steps if cli_steps > 0 else cfg_steps)
        if self.max_global_steps <= 0:
            self.max_global_steps = float('inf')
        
        self.milestone_episodes = self.config.params.get('training_monitor', {}).get('milestone_episodes', [])
        
        print(f"[SETUP] Training components configured:")
        print(f"  Max global steps: {self.max_global_steps}")
        print(f"  Top-K models: {self.args.top_k_models}")
        print(f"  Milestone episodes: {self.milestone_episodes}")

    def _setup_runners_and_evaluators(self):
        """设置执行器和评估器"""
        print(f"[SETUP] Creating TrainingRunner and MilestoneEvaluator...")
        
        # 创建TrainingRunner
        self.runner = TrainingRunner(
            env=self.env,
            maddpg=self.maddpg,
            replay=self.maddpg.replay,
            metrics_hub=self.metrics_hub,
            reward_logger=self.reward_logger,
            agent_ids=self.maddpg.agent_ids
        )
        
        # 创建MilestoneEvaluator
        self.evaluator = MilestoneEvaluator(
            env=self.env,
            maddpg=self.maddpg,
            topk_mgr=self.top_k_manager,
            metrics_hub=self.metrics_hub,
            log_dir=self.logger.log_directory,
            agent_ids=self.maddpg.agent_ids
        )
        
        print(f"[SETUP] TrainingRunner and MilestoneEvaluator created successfully")

    def _setup_milestone_management(self):
        """设置里程碑管理"""
        print(f"[SETUP] Setting up milestone management...")
        
        # 创建MilestoneManager（用于兼容性，实际管理由trainer处理）
        from surgical_project.envs.multi_agent.utils import MilestoneManager
        self.milestone_manager = MilestoneManager(self.milestone_episodes)
        
        # 设置回调（实际上不会被调用，真实管理在check_and_trigger_milestone中）
        milestone_callback = create_milestone_callback(self.evaluator, self.runner)
        self.milestone_manager.set_callback(milestone_callback)
        
        # 里程碑跟踪
        self.max_milestone_triggered = 0
        
        print(f"[SETUP] Milestone management configured for {len(self.milestone_episodes)} milestones")

    def check_and_trigger_milestone(self):
        """检查并触发里程碑（主要的里程碑管理逻辑）"""
        if not self.milestone_episodes:
            return
            
        # 找到最高的已跨越里程碑
        candidate = 0
        for milestone in sorted(self.milestone_episodes):
            if self.runner.global_episodes >= milestone:
                candidate = milestone
            else:
                break
                
        # 如果跨越了新阈值则触发评估
        if candidate > self.max_milestone_triggered:
            print(f"[MILESTONE] Crossed threshold: episodes {self.runner.global_episodes} >= milestone {candidate}")
            print(f"[MILESTONE] Triggering evaluation (previous max: {self.max_milestone_triggered})")
            
            # 直接调用evaluator处理 - 不再用try/except包裹
            result = self.evaluator.handle(candidate, self.runner.global_step)
            if result.get("skip_episode_once", False):
                self.runner.mark_skip_episode_once()
            
            self.max_milestone_triggered = candidate
            print(f"[MILESTONE] Updated max_milestone_triggered to {self.max_milestone_triggered}")

    def train(self) -> None:
        """主训练循环"""
        # 记录训练开始
        self.logger.log_training_start(self.args, self.config.params)
        
        print(f"[TRAIN] Starting training with refactored architecture:")
        print(f"  - TrainingRunner handles: rollout→replay→update→log→count")
        print(f"  - MilestoneEvaluator handles: milestone→eval→topk→log")
        print(f"  - MADDPGTrainer handles: milestone triggering and coordination")
        print(f"  - Max steps: {self.max_global_steps}")
        
        try:
            # 初始化WandB数据点
            if self.wandb_logger.enabled:
                initial_stats = {
                    "train/episodes_done": 0,
                    "replay/buffer_size": 0,
                }
                self.metrics_hub.push_update(0, initial_stats)
            
            # 重置环境
            print(f"[TRAIN] Resetting environment...")
            obs_dict, _ = self.env.reset()
            print(f"[TRAIN] Environment reset complete, starting training loop")
            
            # 主训练循环 - 使用TrainingRunner，删除所有try/except
            while self.runner.global_step < self.max_global_steps:
                # 执行一个训练步骤 - 直接调用，失败就抛异常
                obs_dict = self.runner.run_step()
                
                # 里程碑检查 - 直接调用，失败就抛异常
                self.check_and_trigger_milestone()
                
                # 进度报告 - 直接调用，失败就抛异常
                if self.runner.global_step % 2000 == 0:
                    self.logger.log_training_progress(
                        self.runner.global_step, 
                        self.runner.global_episodes, 
                        self.top_k_manager
                    )
                
                # 检查终止条件
                if self.runner.global_step >= self.max_global_steps:
                    print(f"\n[TRAINING LIMIT] Reached max_global_steps={self.max_global_steps}")
                    break
            
            # 训练完成
            print(f"\n[TRAINING COMPLETE]")
            print(f"  Total steps: {self.runner.global_step}")
            print(f"  Total episodes: {self.runner.global_episodes}")
            print(f"  Max milestone triggered: {self.max_milestone_triggered}")
            
            self.logger.log_training_complete(self.top_k_manager)
            
            # 保存最终模型 - 直接调用，失败就抛异常
            save_final_shared_networks(
                log_directory=self.logger.log_directory,
                maddpg=self.maddpg,
                global_step=self.runner.global_step,
                global_episodes=self.runner.global_episodes,
                max_milestone_triggered=self.max_milestone_triggered
            )
            
            # 保存训练结果 - 直接调用，失败就抛异常
            self.logger.save_final_results(
                self.runner.global_step,
                self.runner.global_episodes,
                self.top_k_manager,
                self.config.params,
                self.args
            )
            print(f"[TRAIN] Final results saved successfully")
        
        except KeyboardInterrupt:
            print(f"\nTraining interrupted by user")
        finally:
            # 清理资源 - 直接调用，失败就抛异常
            if self.reward_logger and hasattr(self.reward_logger, 'close_all_files'):
                self.reward_logger.close_all_files()
            self.env.close()
            self.wandb_logger.finalize_run()
            print("[TRAIN] Cleanup completed")
            print("\nShared network training completed")


def main():
    """主入口点"""
    print("="*80)
    print("MADDPG Shared Network Training - Refactored Architecture")
    print("="*80)
    
    # 解析参数 - 直接调用，失败就抛异常
    parser = create_argument_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    
    print(f"[MAIN] Arguments parsed:")
    print(f"  Task: {args_cli.task}")
    print(f"  Environments: {args_cli.num_envs}")
    print(f"  Max steps: {args_cli.max_global_steps}")
    print(f"  WandB: {args_cli.wandb}")
    print(f"  Config: {args_cli.config}")
    
    # 启动Isaac Sim - 直接调用，失败就抛异常
    print(f"[MAIN] Launching Isaac Sim...")
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    
    try:
        # 创建并启动训练器 - 直接调用，失败就抛异常
        print(f"[MAIN] Creating MADDPGTrainer...")
        trainer = MADDPGTrainer(args_cli)
        
        print(f"[MAIN] Starting training...")
        trainer.train()
        
        print(f"[MAIN] Training completed successfully")
        
    finally:
        print(f"[MAIN] Closing Isaac Sim...")
        simulation_app.close()


if __name__ == "__main__":
    main()