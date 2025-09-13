#!/usr/bin/env python3

"""
Surgical Robot MADDPG Shared Network Training
FINAL VERSION: Complete implementation with shared networks, joint buffer, and in-place evaluation.
MODIFIED: Updated all logging keys to match new WHITELIST structure
MODIFIED: Changed from max_episodes to max_global_steps as primary termination condition

Features:
- Single shared network per agent across all environments
- Joint replay buffer with concatenated observations/actions
- Global step counting (hand-maintained, training only)
- Global episode counting (sum of all environment completions)
- YAML-driven milestone evaluation (in-place, no separate environment)
- Unified step tracking for console logging and WandB
- New logging key structure: train/, model/, replay/, eval/, milestone/
- Primary termination condition: max_global_steps
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
    WandBLogger, TrainingConfiguration, TrainingLogger, create_argument_parser, MetricsHub, TopKModelManager
)

class MADDPGTrainer:
    """Main trainer class for shared network MADDPG algorithm."""
    
    def __init__(self, args):
        self.args = args
        self.config = TrainingConfiguration(args.config)
        self.wandb_logger = WandBLogger(enabled=args.wandb)
        
        # Initialize MetricsHub
        self.metrics = MetricsHub()
        
        self._setup_environment()
        self._setup_logging_and_wandb()
        self._setup_training_components()
        
        # Global step and episode tracking (hand-maintained)
        self.global_step = 0
        self.global_episodes = 0
        
        # Milestone tracking for "cross-threshold" triggering
        self.max_milestone_triggered = 0
        
        # MODIFIED: Setup max_global_steps as primary termination condition
        maddpg_cfg = self.config.params.get('maddpg_config', {})
        cfg_steps = maddpg_cfg.get('max_global_steps', 0) or 0
        cli_steps = getattr(self.args, "max_global_steps", 0) or 0
        self.max_global_steps = int(cli_steps if cli_steps > 0 else cfg_steps)
        if self.max_global_steps <= 0:
            self.max_global_steps = float('inf')
        
        print(f"[INFO] Training limits configured:")
        print(f"  Max global steps: {self.max_global_steps}")
        print(f"  Milestone episodes: {self.config.params.get('training_monitor', {}).get('milestone_episodes', [])}")

    def _get_num_envs_from_env(self):
        """Extract num_envs from environment through wrappers."""
        actual_env = getattr(self.env, 'unwrapped', self.env)
        if hasattr(actual_env, 'num_envs'): 
            return actual_env.num_envs
        if hasattr(actual_env, '_num_envs'): 
            return actual_env._num_envs
        return self.args.num_envs

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

    def _setup_logging_and_wandb(self):
        """Configure logging directories and WandB integration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/maddpg_shared/{timestamp}"
        self.logger = TrainingLogger(log_dir)
        self.checkpoint_path = os.path.join(log_dir, f"checkpoint_top{self.args.top_k_models}.pth")
        
        if self.wandb_logger.enabled:
            run_config = {**vars(self.args), **self.config.params}
            run_name = f"maddpg_shared_{self.args.num_envs}envs_{timestamp}"
            self.wandb_logger.initialize_run(run_config, run_name)
            
            # Attach WandB to MetricsHub
            self.wandb_logger.attach_metrics_hub(self.metrics)

    def _setup_training_components(self):
        """Initialize MADDPG trainer and related training components."""
        from surgical_project.algorithms.marl.maddpg import MADDPG
        device = self.config.get_compute_device()
        
        num_envs = self._get_num_envs_from_env()

        self.maddpg_trainer = MADDPG(
            num_envs=num_envs,
            env=self.env,
            params=self.config.params,
            device=device
        )
        
        # Configure reward logger with simplified architecture
        reward_logger = self._get_reward_logger()
        if reward_logger:
            from surgical_project.envs.multi_agent.utils import RewardLogger
            milestones = self.config.params.get('training_monitor', {}).get('milestone_episodes', [])
            new_logger = RewardLogger(
                num_envs=num_envs,
                device=self.config.get_compute_device(),
                metrics_hub=self.metrics,
                enable_console_logging=self.config.params['logging']['enable_console_logging'],
                milestones=milestones
            )
            
            actual_env = getattr(self.env, 'unwrapped', self.env)
            if hasattr(self.env, 'reward_logger'):
                self.env.reward_logger = new_logger
            elif hasattr(actual_env, 'reward_logger'):
                actual_env.reward_logger = new_logger
            
            reward_logger = self._get_reward_logger()
            if reward_logger:
                reward_logger.configure_logging(self.config.params)
                # NOTE: Milestone callback now handled directly by MADDPGTrainer
                # reward_logger.set_topk_update_callback(self.update_topk_at_milestone)
        
        self.top_k_manager = TopKModelManager(k=self.args.top_k_models)
        self.training_started = False
        self.min_buffer_size = self.maddpg_trainer.min_buffer_size

        print(f"[INFO] Training components initialized:")
        print(f"  MADDPG trainer: {len(self.maddpg_trainer.agent_ids)} agents")
        print(f"  Buffer min size: {self.min_buffer_size}")
        print(f"  Top-K models: {self.args.top_k_models}")
        print(f"  Milestone management: Direct (no RewardLogger callback)")

    def _create_environment(self) -> Tuple[Any, Any]:
        """Create and configure the surgical environment."""
        from surgical_project.envs.multi_agent.surgical_direct_marl_env_cfg import SurgicalDirectMARLEnvCfg
        import gymnasium as gym
        import surgical_project.envs.multi_agent
        
        env_cfg = SurgicalDirectMARLEnvCfg()
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
        """Configure reward logger with MetricsHub injection."""
        reward_logger = self._get_reward_logger()
        if reward_logger:
            num_envs = self._get_num_envs_from_env()
            from surgical_project.envs.multi_agent.utils import RewardLogger
            
            # MODIFIED: Use milestone list directly from YAML without filtering
            milestones = self.config.params.get('training_monitor', {}).get('milestone_episodes', [])
            
            new_logger = RewardLogger(
                num_envs=num_envs,
                device=self.config.get_compute_device(),
                metrics_hub=self.metrics,
                enable_console_logging=self.config.params['logging']['enable_console_logging'],
                milestones=milestones
            )
            
            # Set the new logger in the appropriate place
            actual_env = getattr(self.env, 'unwrapped', self.env)
            if hasattr(self.env, 'reward_logger'):
                self.env.reward_logger = new_logger
            elif hasattr(actual_env, 'reward_logger'):
                actual_env.reward_logger = new_logger
            
            # Configure the new logger
            reward_logger = self._get_reward_logger()
            if reward_logger:
                reward_logger.configure_logging(self.config.params)
                # NOTE: Milestone management now handled directly by MADDPGTrainer

    def _extract_model_state(self) -> Dict[str, Any]:
        """Extract model state dictionaries from shared networks."""
        model_state = {}
        for agent_id in self.maddpg_trainer.agent_ids:
            agent = self.maddpg_trainer.agents[agent_id]
            prefix = f'{agent_id}'
            model_state.update({
                f'{prefix}_actor': agent.actor.state_dict(), 
                f'{prefix}_critic': agent.critic.state_dict(),
                f'{prefix}_actor_target': agent.actor_target.state_dict(), 
                f'{prefix}_critic_target': agent.critic_target.state_dict()
            })
        return model_state

    def evaluate_policy_no_update(self, seed=None):
        """
        In-place policy evaluation (no exploration, no updates, no global_step modification).
        MODIFIED: Single environment (env0) single episode evaluation for efficiency.
        
        Args:
            target_episodes: Number of complete episodes to collect (ignored, forced to 1)
            seed: Optional seed for evaluation reproducibility
            
        Returns:
            Dict with return_avg, num_episodes
        """
        # Force single environment, single episode evaluation
        active_env = 0           # Only evaluate env0
        target_episodes = 1      # Only collect 1 episode
        
        print(f"[EVAL] Starting in-place evaluation (env0 only, 1 episode)...")
        
        # Get actual environment (unwrap if needed)
        env = getattr(self.env, 'unwrapped', self.env)
        
        # Set seed if provided and supported
        if seed is not None and hasattr(env, 'seed'):
            env.seed(seed)
            print(f"[EVAL] Set evaluation seed: {seed}")
        
        # Reset environment for clean evaluation window (handle tuple return)
        obs, _ = env.reset()
        print(f"[EVAL] Environment reset for independent evaluation")
        
        num_envs = self.args.num_envs
        ep_returns = torch.zeros(num_envs, device='cuda' if torch.cuda.is_available() else 'cpu')
        completed_returns = []
        
        # Deterministic evaluation loop (no training modifications)
        with torch.no_grad():
            while len(completed_returns) < target_episodes:
                # Select actions without noise (deterministic policy)
                actions, _ = self.maddpg_trainer.select_actions(obs, add_noise=False)
                
                # MODIFIED: Mask actions - only active_env (env0) gets real actions, others get zeros
                for aid, act in actions.items():
                    # act: [num_envs, act_dim]
                    if act.ndim == 2:
                        masked = torch.zeros_like(act)
                        masked[active_env] = act[active_env]  # Only keep env0's actions
                        actions[aid] = masked
                
                # Environment step (NO store_joint_transitions, NO update, NO global_step increment)
                obs, rewards, terminated, truncated, infos = env.step(actions)
                
                # MODIFIED: Only accumulate rewards from active_env (env0)
                step_rewards = torch.stack([rewards[aid] for aid in self.maddpg_trainer.agent_ids])  # [num_agents, num_envs]
                avg_step_rewards = step_rewards.mean(dim=0)  # [num_envs] - average across agents
                ep_returns[active_env] += avg_step_rewards[active_env]  # Only update env0
                
                # MODIFIED: Only check completion for active_env (env0)
                done_any_dict = {aid: (terminated[aid] | truncated[aid]) for aid in terminated.keys()}
                done_any = None
                for aid in self.maddpg_trainer.agent_ids:
                    d = done_any_dict[aid].to(torch.bool)
                    done_any = d if done_any is None else (done_any | d)
                
                # Check if active_env (env0) completed
                if done_any[active_env]:
                    # Collect env0's episode return
                    ret0 = float(ep_returns[active_env].item())
                    completed_returns.append(ret0)
                    ep_returns[active_env] = 0.0  # Reset for next episode
                    
                    # Stop when we have enough episodes (which is 1)
                    if len(completed_returns) >= target_episodes:
                        break
        
        # Use only the first target_episodes (even if we collected more)
        final_returns = completed_returns[:target_episodes]
        avg_return = sum(final_returns) / max(1, len(final_returns))
        
        print(f"[EVAL] Completed: {len(final_returns)} episodes, Average return: {avg_return:.3f}")
        
        # Reset environment back to training state (handle tuple return)
        _, _ = env.reset()
        print(f"[EVAL] Environment reset back to training mode")
        self._skip_episode_count_once = True
        
        return {
            "return_avg": avg_return,
            "num_episodes": len(final_returns)
        }

    def check_and_trigger_milestone(self):
        """
        Check if we crossed any milestone threshold and trigger evaluation once.
        
        Implements "cross 100 to 101 triggers 100 once" logic:
        - Find the highest milestone <= current global_episodes  
        - If it's higher than max_milestone_triggered, trigger evaluation once
        - Update max_milestone_triggered to prevent duplicate triggers
        """
        milestones = self.config.params.get('training_monitor', {}).get('milestone_episodes', [])
        if not milestones:
            return
            
        # Find the highest milestone we've crossed
        candidate = 0
        for milestone in sorted(milestones):
            if self.global_episodes >= milestone:
                candidate = milestone
            else:
                break
                
        # Trigger evaluation if we crossed a new threshold
        if candidate > self.max_milestone_triggered:
            print(f"[MILESTONE] Crossed threshold: episodes {self.global_episodes} >= milestone {candidate}")
            print(f"[MILESTONE] Triggering evaluation (previous max: {self.max_milestone_triggered})")
            
            self.update_topk_at_milestone(candidate)
            self.max_milestone_triggered = candidate
            
            print(f"[MILESTONE] Updated max_milestone_triggered to {self.max_milestone_triggered}")

    def update_topk_at_milestone(self, milestone: int) -> None:
        """In-place milestone evaluation using current environment."""
        print(f"[MILESTONE {milestone}] Starting in-place policy evaluation...")
        
        # Use current global_step as timestamp (read-only, no modification)
        eval_timestamp = self.global_step
        
        # Perform in-place evaluation (no environment creation)
        eval_result = self.evaluate_policy_no_update(
            seed=4242 + milestone
        )
            
        avg_return = eval_result['return_avg']
        num_episodes = eval_result['num_episodes']
        
        print(f"[MILESTONE {milestone}] Evaluation complete:")
        print(f"  Average return: {avg_return:.3f}")
        print(f"  Episodes evaluated: {num_episodes}")
        print(f"  Timestamp (global_step): {eval_timestamp}")
        
        # Update TopK manager
        model_state = self._extract_model_state()
        self.top_k_manager.update(avg_return, model_state, milestone)
        
        # Save milestone checkpoint
        milestone_path = os.path.join(self.logger.log_directory, f"topk_milestone_{milestone}.pth")
        self.top_k_manager.save_checkpoint(milestone_path, self.maddpg_trainer.agent_ids)
        
        # Push to MetricsHub and WandB using new eval/ and milestone/ keys
        milestone_data = {
            "eval/return_mean": avg_return,
            "eval/num_episodes": num_episodes,
            "milestone/topk_best_score": avg_return,  # For single model evaluation
            "milestone/topk_avg_score": avg_return,   # Same as best for single evaluation
            "milestone/topk_count": 1,                # Single evaluation
            "milestone/latest_completed": milestone,
        }
        
        self.metrics.push_milestone(eval_timestamp, milestone, milestone_data)
        
        print(f"[MILESTONE {milestone}] Results logged and saved, resuming training...")

    def train(self) -> None:
        """Main training loop with shared networks and unified step tracking."""
        self.logger.log_training_start(self.args, self.config.params)
        
        print(f"[INFO] Shared network training with in-place evaluation:")
        print(f"  - Single network per agent shared across all {self.args.num_envs} environments")
        print(f"  - Joint replay buffer with concatenated observations/actions") 
        print(f"  - Global step counting (hand-maintained, training only)")
        print(f"  - Global episode counting (sum of all environment completions)")
        print(f"  - YAML-driven milestone evaluation (in-place, no separate environment)")
        print(f"  - Max steps: {self.max_global_steps}")
        print(f"  - New logging structure: train/, model/, replay/, eval/, milestone/")
        
        try:
            obs_dict, _ = self.env.reset()
            
            # Initial WandB data point using new keys
            if self.wandb_logger.enabled:
                initial_stats = {
                    "train/episodes_done": 0,
                    "replay/buffer_size": 0,
                }
                self.metrics.push_update(self.global_step, initial_stats)
            
            # MODIFIED: Main training loop with step-based termination only
            while self.global_step < self.max_global_steps:
                
                # === TRAINING STEP ===
                # Select actions (shared networks process all environments)
                actions, debug = self.maddpg_trainer.select_actions(obs_dict, add_noise=True)

                # Set debug info for environment before step
                actual_env = getattr(self.env, 'unwrapped', self.env)
                if hasattr(actual_env, 'set_debug_actor_info'):
                    actual_env.set_debug_actor_info(debug)

                # Environment step
                next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                
                # Combine terminated and truncated for proper TD target calculation
                done_any_dict = {aid: (terminated[aid] | truncated[aid]) for aid in terminated.keys()}
                
                # Store joint transitions (training only)
                self.maddpg_trainer.store_joint_transitions(
                    obs_dict, actions, rewards, next_obs, done_any_dict
                )
                
                # Training update
                stats = self.maddpg_trainer.update()
                
                # === STEP AND EPISODE COUNTING ===
                # Hand-maintained global_step increment (training only)
                self.global_step += 1
                
                # Update environment with current global_step for unified console logging
                actual_env = getattr(self.env, 'unwrapped', self.env)
                if hasattr(actual_env, 'set_trainer_global_step'):
                    actual_env.set_trainer_global_step(self.global_step)
                
                # Global episode counting (OR over agents, then sum)
                done_any = None
                for aid in self.maddpg_trainer.agent_ids:
                    d = done_any_dict[aid].to(torch.bool)
                    done_any = d if done_any is None else (done_any | d)
                
                # Skip episode counting once after evaluation to avoid counting eval episodes
                episode_increment = int(done_any.sum().item())
                if getattr(self, "_skip_episode_count_once", False):
                    episode_increment = 0
                    self._skip_episode_count_once = False
                self.global_episodes += episode_increment
                
                # === MILESTONE CHECKING ===
                # YAML-driven milestone triggering (no hardcoded intervals)
                self.check_and_trigger_milestone()
                
                # === LOGGING ===
                # WandB logging (training level) - using new key structure
                if stats and stats.get("training/updates", 0) > 0:
                    # Convert MADDPG stats to new key structure
                    log_data = {
                        "train/actor_loss": stats.get("loss/actor/avg", 0),
                        "train/critic_loss": stats.get("loss/critic/avg", 0),
                        "model/q_mean": stats.get("q_mean/avg", 0),
                        "model/q_std": stats.get("q_std/avg", 0),
                        "model/grad_norm/actor": stats.get("grad_norm/actor/avg", 0),
                        "model/grad_norm/critic": stats.get("grad_norm/critic/avg", 0),
                        "replay/buffer_size": len(self.maddpg_trainer.replay),
                        "train/episodes_done": self.global_episodes,
                        "train/updates": stats.get("training/updates", 0),
                    }

                    # Per-agent action std if available
                    if "action_std/avg" in stats:
                        log_data["train/action_std"] = stats["action_std/avg"]
                    
                    # Individual agent action std if available
                    for aid in self.maddpg_trainer.agent_ids:
                        if aid in stats.get("action_std", {}):
                            log_data[f"train/action_std/{aid}"] = stats["action_std"][aid]
                    
                    # Push to metrics hub with unified global_step
                    self.metrics.push_update(self.global_step, log_data)
                
                # Progress reporting (unified global_step usage)
                if self.global_step % 2000 == 0:
                    self.logger.log_training_progress(self.global_step, self.global_episodes, self.top_k_manager)
                
                # MODIFIED: Only check step-based termination
                if self.global_step >= self.max_global_steps:
                    print(f"\n[TRAINING LIMIT] Reached max_global_steps={self.max_global_steps}")
                    break
                
                obs_dict = next_obs
            
            # Training completion
            print(f"\n[TRAINING COMPLETE]")
            print(f"  Total steps: {self.global_step}")
            print(f"  Total episodes: {self.global_episodes}")
            print(f"  Max milestone triggered: {self.max_milestone_triggered}")
            print(f"[INFO] Shared networks successfully trained across {self.args.num_envs} environments")
            
            self.logger.log_training_complete(self.top_k_manager)
            final_path = os.path.join(self.logger.log_directory, "final_shared_networks.pth")
            
            # Save final shared networks
            final_checkpoint = {
                'params': self.maddpg_trainer.params,
                'agent_ids': self.maddpg_trainer.agent_ids,
                'global_steps_total': self.global_step,
                'episodes_done_total': self.global_episodes,
                'max_milestone_triggered': self.max_milestone_triggered,
                'shared_networks': True,
            }
            for agent_id in self.maddpg_trainer.agent_ids:
                agent = self.maddpg_trainer.agents[agent_id]
                final_checkpoint.update({
                    f'{agent_id}_actor': agent.actor.state_dict(),
                    f'{agent_id}_critic': agent.critic.state_dict(),
                    f'{agent_id}_actor_target': agent.actor_target.state_dict(),
                    f'{agent_id}_critic_target': agent.critic_target.state_dict()
                })
            
            torch.save(final_checkpoint, final_path)
            print(f"[CHECKPOINT] Final shared networks saved: {final_path}")
            
            self.logger.save_final_results(self.global_step, self.global_episodes, self.top_k_manager, self.config.params, self.args)
        
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
            print("\nShared network training completed")

def main():
    """Main entry point for shared network training script."""
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