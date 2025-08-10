#!/usr/bin/env python3
"""MADDPG训练脚本 - 完整修复版本"""

import argparse
import sys
import os
import yaml
import numpy as np
import torch
import random
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train surgical MARL with MADDPG")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default="Isaac-Surgical-MARL-Direct-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_episodes", type=int, default=100)
parser.add_argument("--config", type=str, 
                   default="/home/zzh/workspace/surgical_robot_project/src/surgical_project/envs/multi_agent/agents/training_params.yaml")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

print("[INFO] 启动Isaac Sim...")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[INFO] Isaac Sim启动完成")

import gymnasium as gym
print("[INFO] 导入环境模块...")
import surgical_project.envs.multi_agent
from surgical_project.envs.multi_agent.surgical_direct_marl_env_cfg import SurgicalDirectMARLEnvCfg

# 导入MADDPG相关模块
print("[INFO] 导入MADDPG模块...")
from surgical_project.algorithms.marl.maddpg import MADDPG


def main():
    try:
        # 设置种子
        torch.manual_seed(args_cli.seed)
        np.random.seed(args_cli.seed)
        random.seed(args_cli.seed)
        
        # 加载配置
        print(f"[INFO] 加载配置: {args_cli.config}")
        with open(args_cli.config, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
        
        # 创建环境
        print(f"[INFO] 创建 {args_cli.num_envs} 个并行环境...")
        env_cfg = SurgicalDirectMARLEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        
        env = gym.make(args_cli.task, cfg=env_cfg)
        print("[INFO] 环境创建成功")
        
        # 更新仿真器
        print("[INFO] 更新仿真器...")
        simulation_app.update()
        
        # 创建MADDPG算法
        print("[INFO] 创建MADDPG算法...")
        maddpg = MADDPG(env, params, device='cuda')
        print("[INFO] MADDPG创建成功")
        
        # 测试环境
        print("\n[TEST] 测试环境...")
        obs_dict, _ = env.reset()
        print(f"[TEST] Reset成功，观测形状: {obs_dict['robot'].shape}")
        
        test_actions = {}
        for agent_id in maddpg.agent_ids:
            test_actions[agent_id] = torch.zeros(args_cli.num_envs, 3, device=maddpg.device)
        
        next_obs, rewards, terminated, truncated, info = env.step(test_actions)
        print(f"[TEST] Step成功，奖励: robot={rewards['robot'].mean():.3f}, human={rewards['human'].mean():.3f}")
        
        # 训练循环
        print("\n开始训练...")
        total_steps = 0
        
        for episode in range(args_cli.max_episodes):
            obs_dict, _ = env.reset()
            
            # 重置噪声
            for agent in maddpg.agents.values():
                agent.reset_noise()
            
            episode_rewards = {agent: 0 for agent in maddpg.agent_ids}
            episode_steps = 0
            max_episode_steps = 500
            
            while episode_steps < max_episode_steps:
                # 选择动作 - 修复性能警告
                actions = {}
                for agent_id in maddpg.agent_ids:
                    obs = obs_dict[agent_id]
                    if len(obs.shape) == 1:
                        obs = obs.unsqueeze(0)
                    
                    # 预分配numpy数组避免性能警告
                    obs_np = obs.detach().cpu().numpy()
                    batch_actions = np.zeros((args_cli.num_envs, 3))
                    
                    for env_idx in range(args_cli.num_envs):
                        batch_actions[env_idx] = maddpg.agents[agent_id].select_action(
                            obs_np[env_idx], add_noise=True
                        )
                    
                    # 直接从numpy创建tensor
                    actions[agent_id] = torch.from_numpy(batch_actions).float().to(maddpg.device)
                
                # 环境步进
                next_obs_dict, reward_dict, terminated_dict, truncated_dict, info = env.step(actions)
                
                # 存储经验
                maddpg.store_transition(obs_dict, actions, reward_dict, next_obs_dict, terminated_dict)
                
                # 累积奖励
                for agent in maddpg.agent_ids:
                    episode_rewards[agent] += reward_dict[agent].mean().item()
                
                # 更新网络
                if total_steps % maddpg.update_interval == 0 and len(maddpg.replay_buffer) > maddpg.min_buffer_size:
                    losses = maddpg.update()
                    if losses and episode % 20 == 0:  # 每20个episode打印一次损失
                        print(f"  Losses: {losses}")
                
                obs_dict = next_obs_dict
                total_steps += 1
                episode_steps += 1
                
                # 检查终止条件
                if any(terminated_dict[agent].any() or truncated_dict[agent].any() for agent in maddpg.agent_ids):
                    break
            
            # 打印进度
            if episode % 10 == 0:
                print(f"\nEpisode {episode}/{args_cli.max_episodes}")
                for agent in maddpg.agent_ids:
                    print(f"  {agent}: {episode_rewards[agent]:.2f}")
                print(f"  Buffer size: {len(maddpg.replay_buffer)}")
                print(f"  Total steps: {total_steps}")
            
            # 保存模型
            if episode % 50 == 0 and episode > 0:
                save_path = f"maddpg_checkpoint_episode_{episode}.pt"
                maddpg.save_models(save_path)
                print(f"  Saved model: {save_path}")
        
        print("\n训练完成！")
        
        # 保存最终模型
        final_save_path = "maddpg_final_model.pt"
        maddpg.save_models(final_save_path)
        print(f"最终模型已保存：{final_save_path}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        print("[INFO] 关闭仿真器...")
        simulation_app.close()