#!/usr/bin/env python3

"""
Surgical Robot MADDPG Multi-Environment Parallel Training
Unified dimension management with dual protection mechanism
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
    TopKModelManager, UnifiedProgressManager, TrainingLogger,
    create_argument_parser
)

class MADDPGTrainer:
    """Main trainer class for MADDPG algorithm in surgical robot environments with dual protection."""
    
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
        
        # Inject config to avoid env re-reading YAML
        actual_env = getattr(self.env, 'unwrapped', self.env)
        if hasattr(actual_env, "params"):
            actual_env.params = self.config.params
        else:
            print("[WARNING] Environment does not support config injection")
            
        self._configure_reward_logger()

    def _setup_logging(self):
        """Configure logging directories and WandB integration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/maddpg_parallel/{timestamp}"
        self.logger = TrainingLogger(log_dir)
        self.checkpoint_path = os.path.join(log_dir, f"checkpoint_top{self.args.top_k_models}.pth")
        if self.wandb_logger.enabled:
            run_config = {**vars(self.args), **self.config.params}
            run_name = f"maddpg_dual_protection_{self.args.num_envs}envs_{timestamp}"
            self.wandb_logger.initialize_run(run_config, run_name)

    def _setup_training_components(self):
        """Initialize MADDPG trainer and related training components with dual protection."""
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
        
        # Use unified progress manager
        self.progress = UnifiedProgressManager(
            self.maddpg_trainer.num_envs,
            self.args.max_episodes,
            device=device
        )
        
        # Set up training logger reference for progress manager
        self.progress.training_logger = self.logger
        
        # DUAL PROTECTION SETUP: Register buffer clearing callback (First layer protection)
        self.progress.register_closure_callback(self.maddpg_trainer.disable_environment)
        print("[PROTECTION] Dual protection mechanism activated:")
        print("  - First layer: Immediate buffer clearing on environment closure")
        print("  - Second layer: Training filtering by active environments only")
        
        # Pass to env and logger - handle wrapper case
        actual_env = getattr(self.env, 'unwrapped', self.env)
        if hasattr(actual_env, 'set_progress_manager'):
            actual_env.set_progress_manager(self.progress)
        else:
            print("[WARNING] Environment does not support progress manager injection")
        
        reward_logger = self._get_reward_logger()
        if reward_logger:
            # Compatibility for old interface
            reward_logger.set_episode_counter(self.progress.episode_counts)
            # For reading step counts
            reward_logger.set_progress_manager(self.progress)
        
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

    def _print_buffer_status_debug(self):
        """Print buffer status for debugging dual protection mechanism."""
        status = self.maddpg_trainer.get_buffer_status()
        
        if status['cleared_but_not_empty']:
            print(f"[DEBUG] Cleared environments with non-empty buffers: {status['cleared_but_not_empty']}")
        
        if len(status['empty_buffers']) > 0:
            print(f"[DEBUG] Empty buffers count: {len(status['empty_buffers'])}")

    def train(self) -> None:
            """Main training loop with enhanced WandB monitoring system."""
            self.logger.log_training_start(self.args, self.config.params)
            
            print(f"[INFO] Training with max_episodes={self.args.max_episodes}")
            print("[INFO] Enhanced WandB monitoring:")
            print("  - Every step: env0/env1 detailed metrics")
            print("  - Every 10 steps: global reward/behavior aggregations")
            print("  - Every update: algorithm diagnostics (Q-values, TD errors)")
            print("  - Episode end: performance tracking and progress updates")
            print("  - Every 1000 steps: histogram distributions")
            
            try:
                obs_dict, _ = self.env.reset()
                
                # Use new progress manager
                while not self.progress.is_training_complete():
                    self.global_step += 1
                    
                    # Get active environments list
                    active_envs = self.progress.get_active_environments()
                    if not active_envs:
                        break

                    # Single step entry point: only count steps for active environments
                    self.progress.on_step(active_envs)

                    # Only select actions for active environments (no grad; full batch returned)
                    with torch.no_grad():
                        actions = self.maddpg_trainer.select_actions(
                            obs_dict, active_envs, add_noise=True
                        )
                    
                    next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                    
                    # Combine terminated and truncated signals for proper TD target calculation
                    done_any = {aid: (terminated[aid] | truncated[aid]) for aid in terminated.keys()}
                    
                    # Only store transitions for active environments
                    self.maddpg_trainer.store_transitions_selective(obs_dict, actions, rewards, next_obs, done_any, active_envs)
                    
                    # WandB: Log environment statistics (every 10 steps for global aggregation)
                    if hasattr(self.env, 'extras') and self.global_step % 10 == 0:
                        # Add active environment count to extras
                        self.env.extras.setdefault("log", {})["num_active_envs"] = len(active_envs)
                        self.wandb_logger.log_env_statistics(self.global_step, self.env.extras)

                    can_train = any(
                        len(self.maddpg_trainer.env_replay_buffers[i]) >= self.min_buffer_size 
                        for i in active_envs if i in self.maddpg_trainer.env_replay_buffers
                    )

                    if can_train:
                        if not self.training_started:
                            self.training_started = True
                            print(f"[Step {self.global_step}] Training Started with enhanced WandB monitoring\n")
                        
                        # DUAL PROTECTION: Pass active_envs to update method
                        algorithm_stats = self.maddpg_trainer.update(active_envs)
                        
                        # WandB: Log algorithm statistics (every update)
                        if algorithm_stats.get("training/updates", 0) > 0:
                            self.wandb_logger.log_algorithm_statistics(self.global_step, algorithm_stats)

                    # WandB: Episode completion handling
                    if hasattr(self.env, 'extras') and 'episode' in self.env.extras and self.env.extras['episode']:
                        # Log episode statistics with env0/env1 detail + global aggregation
                        self.wandb_logger.log_episode_statistics(self.global_step, self.env.extras['episode'])
                        
                        # Update performance leaderboard and progress tracking
                        self.wandb_logger.update_performance_and_progress(
                            self.global_step, 
                            self.env.extras['episode'], 
                            total_envs=self.maddpg_trainer.num_envs
                        )
                        
                        # Clear episode data after logging
                        self.env.extras['episode'].clear()

                    # Episode settlement is completely handled by env._reset_idx()
                    obs_dict = next_obs

                    # Progress reporting (reduced frequency to avoid log spam)
                    if self.global_step > 0 and self.global_step % 2000 == 0:
                        stats = self.progress.get_progress_statistics()
                        self.logger.log_training_progress(self.global_step, stats, self.top_k_manager)
                        # Keep existing WandB progress logging for compatibility
                        self.wandb_logger.log_training_progress(self.global_step, stats, self.top_k_manager)
                    
                    # Debug buffer status every 5000 steps
                    if self.global_step > 0 and self.global_step % 5000 == 0:
                        self._print_buffer_status_debug()
                    
                    # Milestone progress reporting every 3000 steps (reduced frequency)
                    if self.global_step > 0 and self.global_step % 3000 == 0:
                        reward_logger = self._get_reward_logger()
                        if reward_logger:
                            next_milestone, reached, remaining = reward_logger.get_next_milestone_progress()
                            if next_milestone is not None:
                                print(f"[Step {self.global_step}] Next Milestone {next_milestone}: "
                                    f"{reached}/{self.maddpg_trainer.num_envs} environments reached "
                                    f"({remaining} remaining)")
                            else:
                                print(f"[Step {self.global_step}] All environments completed all milestones!")
                
                # Final evaluation of all environments
                print("[TopK Update] Starting final evaluation...")
                reward_logger = self._get_reward_logger()
                if reward_logger:
                    for env_id in range(self.maddpg_trainer.num_envs):
                        try:
                            final_episode_count = self.progress.env_episode_counts[env_id]
                            performance = reward_logger.get_final_evaluation(env_id, final_episode_count)
                            model_state = self._extract_model_state(env_id)
                            self.top_k_manager.update_model(env_id, performance, model_state)

                        except Exception as e:
                            print(f"[WARNING] Env {env_id} final evaluation failed: {e}")

                # Final buffer status report
                print("\n[FINAL STATUS] Enhanced monitoring summary:")
                status = self.maddpg_trainer.get_buffer_status()
                print(f"  Total environments: {status['total_envs']}")
                print(f"  Cleared environments: {status['cleared_envs']}")
                print(f"  Empty buffers: {len(status['empty_buffers'])}")
                if status['cleared_but_not_empty']:
                    print(f"  WARNING: Cleared but not empty: {status['cleared_but_not_empty']}")

                self.logger.log_training_complete(self.top_k_manager)
                final_path = os.path.join(self.logger.log_directory, "final_top_k_models.pth")
                self.top_k_manager.save_checkpoint(final_path, self.maddpg_trainer)
                self.logger.save_final_results(self.global_step, self.progress, self.top_k_manager, self.config.params, self.args)
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
                print("\nTraining finished with enhanced WandB monitoring")

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