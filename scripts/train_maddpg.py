#!/usr/bin/env python3

"""
Surgical Robot MADDPG Shared Network Training
Enhanced version with configurable network architecture and exponential noise decay.

Features:
- Configurable network layers, dropout, and orthogonal initialization
- Exponential noise scheduling: fast early decay, slow later convergence
- Unified global step tracking for consistent logging
- TopK model management with milestone evaluation
- WandB integration with exploration metrics
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
    """Create environment and environment configuration."""
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
    """Create MADDPG trainer with configurable networks."""
    from surgical_project.algorithms.marl.maddpg import MADDPG
    
    device = config.get_compute_device()
    
    # Get environment count - simplified logic
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
    """Configure reward logger."""
    # Get existing reward_logger
    reward_logger = None
    if hasattr(env, 'reward_logger'):
        reward_logger = env.reward_logger
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'reward_logger'):
        reward_logger = env.unwrapped.reward_logger
    
    if not reward_logger:
        print("[INFO] No existing reward_logger found, will create a new one")
        return None
        
    # Get environment count
    num_envs = getattr(env, 'num_envs', getattr(getattr(env, 'unwrapped', env), 'num_envs', 512))
    
    # Create new RewardLogger
    from surgical_project.envs.multi_agent.utils import RewardLogger
    milestones = config.params.get('training_monitor', {}).get('milestone_episodes', [])
    
    new_logger = RewardLogger(
        num_envs=num_envs,
        device=config.get_compute_device(),
        metrics_hub=metrics_hub,
        enable_console_logging=config.params.get('logging', {}).get('enable_console_logging', False),
        milestones=milestones
    )
    
    # Replace logger in environment
    if hasattr(env, 'reward_logger'):
        env.reward_logger = new_logger
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'reward_logger'):
        env.unwrapped.reward_logger = new_logger
    
    # Configure new logger
    new_logger.configure_logging(config.params)
    
    print(f"[INFO] RewardLogger configured:")
    print(f"  Console logging: {'enabled' if new_logger.step_tracer.enable_console_logging else 'disabled'}")
    print(f"  Milestone management: Legacy compatibility mode")
    
    return new_logger


def create_milestone_callback(evaluator, runner):
    """Create milestone callback function."""
    def _on_milestone(milestone):
        """Milestone callback: evaluation→TopK→skip episode counting"""
        print(f"[MILESTONE CALLBACK] Processing milestone {milestone}")
        result = evaluator.handle(milestone, runner.global_step)
        if result.get("skip_episode_once", False):
            runner.mark_skip_episode_once()
            print(f"[MILESTONE CALLBACK] Marked skip_episode_once for milestone {milestone}")
    return _on_milestone


class MADDPGTrainer:
    """
    Streamlined MADDPG trainer for configurable networks and noise scheduling.
    
    Features:
    - Configuration-driven network architecture
    - Exponential noise decay scheduling
    - Milestone evaluation with TopK model selection
    - Comprehensive WandB logging with exploration metrics
    """
    
    def __init__(self, args):
        self.args = args
        
        print(f"[TRAINER] Initializing MADDPGTrainer with configurable networks...")
        
        # Setup phase
        self._setup_configuration()
        self._setup_environment()
        self._setup_logging_and_wandb()
        self._setup_training_components()
        self._setup_runners_and_evaluators()
        self._setup_milestone_management()
        
        print(f"[TRAINER] MADDPGTrainer initialized successfully")
        print(f"  Max global steps: {self.max_global_steps}")
        print(f"  Milestone episodes: {self.milestone_episodes}")
        
        # Print network configuration
        self._print_configuration_summary()

    def _setup_configuration(self):
        """Setup configuration."""
        print(f"[SETUP] Loading configuration from: {self.args.config}")
        self.config = TrainingConfiguration.from_yaml(self.args.config)
        
        # Set random seeds
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        self.config.params['seed'] = self.args.seed
        
        print(f"[SETUP] Configuration loaded, seed set to: {self.args.seed}")

    def _setup_environment(self):
        """Setup environment."""
        print(f"[SETUP] Creating environment: {self.args.task}")
        self.env, self.env_cfg = create_env_and_config(self.args, self.config)
        
        # Inject configuration to environment
        actual_env = getattr(self.env, 'unwrapped', self.env)
        if hasattr(actual_env, "params"):
            actual_env.params = self.config.params
            print(f"[SETUP] Injected configuration parameters to environment")
        else:
            print("[WARNING] Environment does not support config injection")

    def _setup_logging_and_wandb(self):
        """Setup logging and WandB."""
        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/maddpg_configurable/{timestamp}"
        
        self.logger = TrainingLogger(log_dir)
        
        print(f"[SETUP] Log directory created: {log_dir}")
        
        # Initialize WandB
        self.wandb_logger = WandBLogger(enabled=self.args.wandb)
        if self.wandb_logger.enabled:
            run_config = {**vars(self.args), **self.config.params}
            run_name = f"maddpg_configurable_{self.args.num_envs}envs_{timestamp}"
            self.wandb_logger.initialize_run(run_config, run_name)
            print(f"[SETUP] WandB initialized with run name: {run_name}")
        else:
            print(f"[SETUP] WandB disabled")

    def _setup_training_components(self):
        """Setup training components."""
        print(f"[SETUP] Setting up training components...")
        
        # Create MetricsHub
        self.metrics_hub = MetricsHub()
        
        # Connect WandB to MetricsHub
        if self.wandb_logger.enabled:
            self.wandb_logger.attach_metrics_hub(self.metrics_hub)
            print(f"[SETUP] WandB attached to MetricsHub")
        
        # Create MADDPG trainer
        self.maddpg = create_maddpg_trainer(self.env, self.config, self.args)
        
        # Setup reward logger
        self.reward_logger = setup_reward_logger(self.env, self.config, self.metrics_hub)
        
        # Create TopK manager
        self.top_k_manager = TopKModelManager(k=self.args.top_k_models, mode="max")
        
        # Setup training parameters
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
        """Setup runners and evaluators."""
        print(f"[SETUP] Creating TrainingRunner and MilestoneEvaluator...")
        
        # Create TrainingRunner with noise scheduling
        self.runner = TrainingRunner(
            env=self.env,
            maddpg=self.maddpg,
            replay=self.maddpg.replay,
            metrics_hub=self.metrics_hub,
            reward_logger=self.reward_logger,
            agent_ids=self.maddpg.agent_ids
        )
        
        # Create MilestoneEvaluator
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
        """Setup milestone management."""
        print(f"[SETUP] Setting up milestone management...")
        
        # Create MilestoneManager for compatibility
        from surgical_project.envs.multi_agent.utils import MilestoneManager
        self.milestone_manager = MilestoneManager(self.milestone_episodes)
        
        # Set callback (actual management handled by check_and_trigger_milestone)
        milestone_callback = create_milestone_callback(self.evaluator, self.runner)
        self.milestone_manager.set_callback(milestone_callback)
        
        # Milestone tracking
        self.max_milestone_triggered = 0
        
        print(f"[SETUP] Milestone management configured for {len(self.milestone_episodes)} milestones")

    def _print_configuration_summary(self):
        """Print configuration summary."""
        networks_cfg = self.config.params.get('networks', {})
        if networks_cfg:
            print(f"[CONFIG] Network Architecture:")
            actor_cfg = networks_cfg.get('actor', {})
            critic_cfg = networks_cfg.get('critic', {})
            print(f"  Actor layers: {actor_cfg.get('hidden_layers', 'default')}")
            print(f"  Critic layers: {critic_cfg.get('hidden_layers', 'default')}")
            print(f"  Dropout: Actor {actor_cfg.get('dropout_p', 0.0)}, Critic {critic_cfg.get('dropout_p', 0.0)}")
            print(f"  Orthogonal init: {actor_cfg.get('orthogonal_init', False)}")
            print(f"  Gains: Hidden {actor_cfg.get('ortho_gain_hidden', 1.0)}, Output {actor_cfg.get('ortho_gain_output', 0.01)}")
            print(f"  Std scale: {actor_cfg.get('std_scale', 1.0)}")
        
        exploration_cfg = self.config.params.get('exploration', {})
        if exploration_cfg:
            print(f"[CONFIG] Exploration Schedule:")
            print(f"  Noise range: {exploration_cfg.get('sigma_start', 0.7)} → {exploration_cfg.get('sigma_end', 0.1)}")
            print(f"  Decay rate: {exploration_cfg.get('decay_k', 6.0)}")

    def check_and_trigger_milestone(self):
        """Check and trigger milestone evaluation."""
        if not self.milestone_episodes:
            return
            
        # Find the highest crossed milestone
        candidate = 0
        for milestone in sorted(self.milestone_episodes):
            if self.runner.global_episodes >= milestone:
                candidate = milestone
            else:
                break
                
        # If crossed new threshold, trigger evaluation
        if candidate > self.max_milestone_triggered:
            print(f"[MILESTONE] Crossed threshold: episodes {self.runner.global_episodes} >= milestone {candidate}")
            print(f"[MILESTONE] Triggering evaluation (previous max: {self.max_milestone_triggered})")
            
            # Direct call to evaluator
            result = self.evaluator.handle(candidate, self.runner.global_step)
            if result.get("skip_episode_once", False):
                self.runner.mark_skip_episode_once()
            
            self.max_milestone_triggered = candidate
            print(f"[MILESTONE] Updated max_milestone_triggered to {self.max_milestone_triggered}")

    def train(self) -> None:
        """Main training loop with configurable networks and noise scheduling."""
        # Record training start
        self.logger.log_training_start(self.args, self.config.params)
        
        print(f"[TRAIN] Starting training with configurable networks and noise scheduling:")
        print(f"  - TrainingRunner: rollout→replay→update→log→count + exponential noise decay")
        print(f"  - MilestoneEvaluator: milestone→eval→topk→log")
        print(f"  - Network layers configurable via YAML")
        print(f"  - Noise schedule: σ_start={self.runner.sigma_start} → σ_end={self.runner.sigma_end}")
        print(f"  - Max steps: {self.max_global_steps}")
        
        try:
            # Initialize WandB data points
            if self.wandb_logger.enabled:
                initial_stats = {
                    "train/episodes_done": 0,
                    "replay/buffer_size": 0,
                    "exploration/noise_scale": self.runner.sigma_start,
                    "exploration/sigma_ratio": 1.0,
                }
                self.metrics_hub.push_update(0, initial_stats)
            
            # Reset environment
            print(f"[TRAIN] Resetting environment...")
            obs_dict, _ = self.env.reset()
            print(f"[TRAIN] Environment reset complete, starting training loop")
            self.runner._current_obs = obs_dict  
            # Main training loop
            while self.runner.global_step < self.max_global_steps:
                # Execute one training step
                obs_dict = self.runner.run_step()
                
                # Milestone checking
                self.check_and_trigger_milestone()
                
                # Progress reporting
                if self.runner.global_step % 2000 == 0:
                    self.logger.log_training_progress(
                        self.runner.global_step, 
                        self.runner.global_episodes, 
                        self.top_k_manager
                    )
                
                # Check termination condition
                if self.runner.global_step >= self.max_global_steps:
                    print(f"\n[TRAINING LIMIT] Reached max_global_steps={self.max_global_steps}")
                    break
            
            # Training complete
            print(f"\n[TRAINING COMPLETE]")
            print(f"  Total steps: {self.runner.global_step}")
            print(f"  Total episodes: {self.runner.global_episodes}")
            print(f"  Max milestone triggered: {self.max_milestone_triggered}")
            print(f"  Final noise scale: {self.runner._calculate_noise_scale():.4f}")
            
            self.logger.log_training_complete(self.top_k_manager)
            
            # Save final model
            save_final_shared_networks(
                log_directory=self.logger.log_directory,
                maddpg=self.maddpg,
                global_step=self.runner.global_step,
                global_episodes=self.runner.global_episodes,
                max_milestone_triggered=self.max_milestone_triggered
            )
            
            # Save training results
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
            # Cleanup resources
            if self.reward_logger and hasattr(self.reward_logger, 'close_all_files'):
                self.reward_logger.close_all_files()
            self.env.close()
            self.wandb_logger.finalize_run()
            print("[TRAIN] Cleanup completed")
            print("\nConfigurable network training completed")


def main():
    """Main entry point."""
    print("="*80)
    print("MADDPG Configurable Network Training")
    print("Enhanced with exponential noise scheduling and network flexibility")
    print("="*80)
    
    # Parse arguments
    parser = create_argument_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    
    print(f"[MAIN] Arguments parsed:")
    print(f"  Task: {args_cli.task}")
    print(f"  Environments: {args_cli.num_envs}")
    print(f"  Max steps: {args_cli.max_global_steps}")
    print(f"  WandB: {args_cli.wandb}")
    print(f"  Config: {args_cli.config}")
    
    # Launch Isaac Sim
    print(f"[MAIN] Launching Isaac Sim...")
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    
    try:
        # Create and start trainer
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