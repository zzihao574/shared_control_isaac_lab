#!/usr/bin/env python3

"""
Surgical Robot MADDPG Multi-Environment Parallel Training
Unified dimension management and streamlined configuration
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
    CheckpointManager, WandBLogger, TrainingConfiguration,
    TopKModelManager, TrainingProgressTracker, TrainingLogger,
    create_argument_parser
)

class MADDPGTrainer:
    """Main trainer class for MADDPG algorithm in surgical robot environments."""
    
    def __init__(self, args):
        self.args = args
        self.config = TrainingConfiguration(args.config)
        self.checkpoint_manager = CheckpointManager(args.checkpoint, args.load_strategy)
        self.wandb_logger = WandBLogger(enabled=args.wandb)
        self._setup_environment()
        self._setup_logging()
        self._setup_training_components()
        self.global_step = 0

    def _setup_environment(self):
        """Initialize environment with proper seeding and configuration."""
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        self.config.params['seed'] = self.args.seed
        self.env, self.env_cfg = self._create_environment()
        self._configure_reward_logger()

    def _setup_logging(self):
        """Configure logging directories and WandB integration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/maddpg_parallel/{timestamp}"
        self.logger = TrainingLogger(log_dir)
        self.checkpoint_path = os.path.join(log_dir, f"checkpoint_top{self.args.top_k_models}.pth")
        if self.wandb_logger.enabled:
            run_config = {**vars(self.args), **self.config.params}
            run_name = f"maddpg_{self.args.num_envs}envs_{timestamp}"
            self.wandb_logger.initialize_run(run_config, run_name)

    def _setup_training_components(self):
        """Initialize MADDPG trainer and related training components."""
        from surgical_project.algorithms.marl.maddpg import MADDPG
        device = self.config.get_compute_device()

        self.maddpg_trainer = MADDPG(
            num_envs=self.args.num_envs,
            env=self.env,
            params=self.config.params,
            device=device
        )

        if self.checkpoint_manager.load_checkpoint():
            self.checkpoint_manager.initialize_agents_from_checkpoint(self.maddpg_trainer)
        
        # Create single episode counter in TrainingProgressTracker
        self.progress_tracker = TrainingProgressTracker(self.maddpg_trainer.num_envs, self.args.max_episodes)
        
        # Pass counter reference to reward logger
        reward_logger = self._get_reward_logger()
        if reward_logger:
            reward_logger.set_episode_counter(self.progress_tracker.env_episode_counts)
        
        self.top_k_manager = TopKModelManager(k=self.args.top_k_models)
        self.training_started = False
        self.min_buffer_size = self.maddpg_trainer.min_buffer_size

    def _create_environment(self) -> Tuple[Any, Any]:
        """Create and configure the surgical environment."""
        from surgical_project.envs.multi_agent.surgical_direct_marl_env_cfg import SurgicalDirectMARLEnvCfg
        import gymnasium as gym
        import surgical_project.envs.multi_agent
        
        env_cfg = SurgicalDirectMARLEnvCfg()
        # Override environment count from command line arguments
        env_cfg.scene.num_envs = self.args.num_envs
        env_cfg.seed = self.args.seed
        
        print(f"[INFO] Environment configuration:")
        print(f"  Number of environments: {env_cfg.scene.num_envs}")
        print(f"  Episode length: {env_cfg.episode_length_s}s")
        print(f"  Decimation: {env_cfg.decimation}")
        print(f"  Possible agents: {env_cfg.possible_agents}")
        print(f"  Action spaces: {env_cfg.action_spaces}")
        print(f"  Observation spaces: {env_cfg.observation_spaces}")
        
        env = gym.make(self.args.task, cfg=env_cfg)
        
        if hasattr(env, 'max_episode_length'):
            print(f"[INFO] Environment max_episode_length: {env.max_episode_length}")
        
        return env, env_cfg

    def _get_reward_logger(self):
        """Get reward logger from environment."""
        return getattr(self.env, 'reward_logger', None) or getattr(getattr(self.env, 'unwrapped', None), 'reward_logger', None)

    def _configure_reward_logger(self):
        """Configure reward logger with milestone filtering."""
        reward_logger = self._get_reward_logger()
        if reward_logger:
            # Pass max_episodes to reward logger configuration
            self.config.params['training_monitor']['max_episodes'] = self.args.max_episodes
            
            # Filter milestones to only include those within max_episodes
            original_milestones = self.config.params['training_monitor'].get('milestone_episodes', [])
            filtered_milestones = [m for m in original_milestones if m <= self.args.max_episodes]
            self.config.params['training_monitor']['milestone_episodes'] = filtered_milestones
            
            print(f"[INFO] Filtered milestones for max_episodes={self.args.max_episodes}: {filtered_milestones}")
            
            reward_logger.configure_logging(self.config.params)
            reward_logger.set_topk_update_callback(self.update_topk_at_milestone)

    def _extract_model_state(self, env_id: int) -> Dict[str, Any]:
        """Extract model state dictionaries for all agents in an environment."""
        model_state = {}
        for agent_id in self.maddpg_trainer.agent_ids:
            agent = self.maddpg_trainer.env_agents[env_id][agent_id]
            prefix = f'{agent_id}'
            model_state.update({
                f'{prefix}_actor': agent.actor.state_dict(), 
                f'{prefix}_critic': agent.critic.state_dict(),
                f'{prefix}_actor_target': agent.actor_target.state_dict(), 
                f'{prefix}_critic_target': agent.critic_target.state_dict()
            })
        return model_state

    def update_topk_at_milestone(self, milestone: int) -> None:
        """Update Top-K models when milestone episodes are reached."""
        print(f"[TopK Update] Evaluating all environments at Milestone {milestone}...")
        reward_logger = self._get_reward_logger()
        if not reward_logger or milestone not in reward_logger.milestone_performances:
            print(f"[WARNING] Milestone {milestone} data incomplete, skipping TopK update")
            return
        
        performances = reward_logger.milestone_performances[milestone]
        for env_id, perf_data in performances.items():
            try:
                model_state = self._extract_model_state(env_id)
                self.top_k_manager.update_model(env_id, perf_data['score'], model_state)
            except Exception as e:
                print(f"[WARNING] Env {env_id} model state extraction failed: {e}")
        
        milestone_path = os.path.join(self.logger.log_directory, f"topk_milestone_{milestone}.pth")
        self.top_k_manager.save_checkpoint(milestone_path, self.maddpg_trainer)
        
        self.wandb_logger.log_milestone_completion(self.global_step, milestone, performances)

    def _process_rewards_and_completion(self, active_envs: List[int], terminated: Dict[str, Any], truncated: Dict[str, Any]):
        """Process rewards and handle environment completion detection."""
        reward_logger = self._get_reward_logger()
        if not reward_logger:
            return

        for env_id in active_envs:
            current_steps = reward_logger.episode_tracker.current_episode_basic[env_id]['steps']
            
            # Detect episode end by explicit done flags OR step count reset (Isaac Lab auto-reset)
            env_done = any(terminated[agent][env_id] or truncated[agent][env_id] for agent in self.maddpg_trainer.agent_ids)
            
            # Isaac Lab auto-resets without setting truncated=True
            # Detect reset by step count dropping significantly
            if not hasattr(self, '_prev_steps'):
                self._prev_steps = {}
            
            prev_steps = self._prev_steps.get(env_id, 0)
            episode_reset = prev_steps > current_steps + 100  # Step count dropped significantly
            
            if episode_reset and not env_done:
                env_done = True
            
            self._prev_steps[env_id] = current_steps
            
            if env_done and prev_steps > 0:  # Use prev_steps since current_steps may have reset
                # Process episode metrics and milestones
                reward_logger.on_episode_end(torch.tensor([env_id], device=self.maddpg_trainer.device))
                
                # Increment episode counter
                reached_max = self.progress_tracker.complete_episode(env_id)
                
                if reached_max:
                    final_episode_count = self.progress_tracker.env_episode_counts[env_id]
                    final_performance = reward_logger.get_final_evaluation(env_id, final_episode_count)
                    
                    self.logger.log_environment_completion(
                        env_id, final_episode_count,
                        final_performance, self.progress_tracker.num_completed_envs,
                        self.maddpg_trainer.num_envs
                    )
                    
                    self.maddpg_trainer.disable_environment(env_id)

    def train(self) -> None:
        """Main training loop."""
        self.logger.log_training_start(self.args, self.config.params)
        
        print(f"[INFO] Training with max_episodes={self.args.max_episodes}")
        
        try:
            obs_dict, _ = self.env.reset()
            while not self.progress_tracker.is_training_complete():
                self.global_step += 1
                active_envs = self.progress_tracker.get_active_environments()
                if not active_envs:
                    break

                actions = self.maddpg_trainer.select_actions(obs_dict, active_envs, add_noise=True)
                
                next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                
                self.maddpg_trainer.store_transitions_selective(obs_dict, actions, rewards, next_obs, terminated, active_envs)
                
                can_train = any(
                    len(self.maddpg_trainer.env_replay_buffers[i]) >= self.min_buffer_size 
                    for i in active_envs if i in self.maddpg_trainer.env_replay_buffers
                )

                if can_train:
                    if not self.training_started:
                        self.training_started = True
                        print(f"[Step {self.global_step}] Training Started\n")
                    algorithm_stats = self.maddpg_trainer.update()
                    self.wandb_logger.log_algorithm_statistics(self.global_step, algorithm_stats)

                self._process_rewards_and_completion(active_envs, terminated, truncated)
                
                obs_dict = next_obs

                if self.global_step > 0 and self.global_step % 1000 == 0:
                    stats = self.progress_tracker.get_progress_statistics()
                    self.logger.log_training_progress(self.global_step, stats, self.top_k_manager)
                    self.wandb_logger.log_training_progress(self.global_step, stats, self.top_k_manager)
            
            # Final evaluation of all environments
            print("[TopK Update] Starting final evaluation...")
            reward_logger = self._get_reward_logger()
            if reward_logger:
                for env_id in range(self.maddpg_trainer.num_envs):
                    try:
                        final_episode_count = self.progress_tracker.env_episode_counts[env_id]
                        performance = reward_logger.get_final_evaluation(env_id, final_episode_count)
                        model_state = self._extract_model_state(env_id)
                        self.top_k_manager.update_model(env_id, performance, model_state)

                    except Exception as e:
                        print(f"[WARNING] Env {env_id} final evaluation failed: {e}")

            self.logger.log_training_complete(self.top_k_manager)
            final_path = os.path.join(self.logger.log_directory, "final_top_k_models.pth")
            self.top_k_manager.save_checkpoint(final_path, self.maddpg_trainer)
            self.logger.save_final_results(self.global_step, self.progress_tracker, self.top_k_manager, self.config.params, self.args)
            self.wandb_logger.log_training_completion(self.global_step, self.top_k_manager)
        
        except (KeyboardInterrupt, Exception) as e:
            print(f"\nTraining interrupted or failed: {e}")
            if isinstance(e, Exception):
                import traceback
                traceback.print_exc()
        finally:
            if self._get_reward_logger():
                self._get_reward_logger().close_all_files()
            self.env.close()
            self.wandb_logger.finalize_run()
            print("\nTraining finished")

def main():
    """Main entry point for training script."""
    parser = create_argument_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        trainer = MADDPGTrainer(args_cli)
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()