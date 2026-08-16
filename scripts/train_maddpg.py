#!/usr/bin/env python3

"""
MADDPG training for surgical robot with shared networks and per-env scaling.
Features configurable architectures, noise scheduling, and milestone evaluation.
"""

import sys
import os
import copy
import json
import subprocess
import torch
import numpy as np
import yaml
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from isaaclab.app import AppLauncher
from utils.training_helpers_maddpg import (
    WandBLogger,
    TrainingConfiguration,
    create_argument_parser,
    MetricsHub,
    TrainingRunner,
    MilestoneEvaluator,
    resume_from_checkpoint_maddpg,
    save_final_checkpoint_maddpg,
    SeedPlan,
    build_maddpg_wandb_config,
    resolve_startup_seed,
    setup_global_reproducibility,
)


def setup_environment(args, config):
    """Create and configure the surgical robot environment."""
    from surgical_project.envs.multi_agent.surgical_direct_marl_env_cfg import SurgicalDirectMARLEnvCfg
    import gymnasium as gym
    import surgical_project.envs.multi_agent
    
    env_cfg = SurgicalDirectMARLEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.params = config.params
    
    render_mode = getattr(args, "render_mode", None)
    env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)
    return env, env_cfg


def initialize_maddpg_algorithm(env, config, args):
    """Create and initialize MADDPG algorithm with configurable networks."""
    from surgical_project.algorithms.marl.maddpg import MADDPG
    
    device = config.get_compute_device()
    
    # Get environment count
    num_envs = args.num_envs
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'num_envs'):
        num_envs = env.unwrapped.num_envs
    elif hasattr(env, 'num_envs'):
        num_envs = env.num_envs

    maddpg = MADDPG(
        num_envs=num_envs,
        env=env,
        params=config.params,
        seed_plan=SeedPlan(config.params.get("seed", args.seed)),
        device=device
    )
    
    return maddpg


def inject_step_tracer(env, config, num_envs):
    """Inject StepTracer into environment for console logging."""
    actual_env = getattr(env, "unwrapped", env)
    
    from surgical_project.envs.multi_agent.utils import StepTracer
    
    actual_env.step_tracer = StepTracer(
        num_envs=num_envs,
        device=getattr(actual_env, "device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
        enable_console_logging=config.params.get("logging", {}).get("enable_console_logging", False)
    )
    
    print(f"[INFO] StepTracer injected:")
    print(f"  Console logging: {'enabled' if actual_env.step_tracer.enable_console_logging else 'disabled'}")


class MADDPGTrainer:
    """
    MADDPG trainer with shared networks, per-env scaling, and milestone evaluation.
    Supports configurable architectures, noise scheduling, and eval mode control.
    """
    
    def __init__(self, args):
        self.args = args
        
        print(f"[TRAINER] Initializing MADDPGTrainer with persistent generators and per-env scaling...")
        
        # Setup phases
        self._setup_configuration()
        self._setup_environment()
        self._setup_logging_and_wandb()
        self._setup_training_components()
        self._setup_runners_and_evaluators()
        self._resume_if_requested()
        self._setup_milestone_management()
        
        print(f"[TRAINER] MADDPGTrainer initialized successfully")
        print(f"  Max global steps: {self.max_global_steps}")
        print(f"  Milestone episodes: {self.milestone_episodes}")
        
        self._print_configuration_summary()

    def _setup_configuration(self):
        """Load and setup training configuration with per-env scaling."""
        print(f"[SETUP] Loading configuration from: {self.args.config}")
        self.config = TrainingConfiguration.from_yaml(self.args.config)

        # A resumed run must reconstruct the experiment from the checkpoint,
        # not silently from whatever the current default YAML contains.
        if self.args.checkpoint:
            checkpoint = torch.load(
                self.args.checkpoint, map_location="cpu", weights_only=False
            )
            checkpoint_params = checkpoint.get("params")
            if isinstance(checkpoint_params, dict):
                self.config.params = copy.deepcopy(checkpoint_params)
                print("[SETUP] Restored resolved configuration from checkpoint")

        configured_seed = int(self.config.params.get("seed", 42))
        if self.args.checkpoint and self.args.seed is not None and int(self.args.seed) != configured_seed:
            raise ValueError(
                "Cannot resume with a different seed than the checkpoint: "
                f"checkpoint={configured_seed}, CLI={self.args.seed}"
            )
        self.args.seed = configured_seed if self.args.seed is None else int(self.args.seed)

        runtime = self.config.params.get("runtime", {})
        checkpoint_num_envs = int(runtime.get("num_envs", 512))
        if self.args.checkpoint:
            if self.args.num_envs is not None and int(self.args.num_envs) != checkpoint_num_envs:
                raise ValueError(
                    "Cannot resume with a different num_envs than the checkpoint: "
                    f"checkpoint={checkpoint_num_envs}, CLI={self.args.num_envs}"
                )
            self.args.num_envs = checkpoint_num_envs
        elif self.args.num_envs is None:
            self.args.num_envs = 512
        checkpoint_model_type = str(
            self.config.params.get("human_model_type", "learnable")
        )
        if self.args.human_model_type is not None:
            if self.args.checkpoint and self.args.human_model_type != checkpoint_model_type:
                raise ValueError(
                    "Cannot resume a checkpoint with a different human_model_type: "
                    f"checkpoint={checkpoint_model_type}, CLI={self.args.human_model_type}"
                )
            self.config.params["human_model_type"] = self.args.human_model_type
        else:
            self.config.params.setdefault("human_model_type", "learnable")

        maddpg_cfg = self.config.params.setdefault("maddpg_config", {})
        if self.args.max_global_steps > 0:
            maddpg_cfg["max_global_steps"] = int(self.args.max_global_steps)

        self.config.params["seed"] = self.args.seed
        self.config.params["runtime"] = {
            "algorithm": "maddpg",
            "num_envs": int(self.args.num_envs),
            "max_global_steps": int(maddpg_cfg.get("max_global_steps", 200000)),
        }
        
        print(f"[SETUP] Configuration loaded with startup seed: {self.args.seed}")
        
        # Checkpoints already contain the runtime-scaled replay/milestone values.
        if self.args.checkpoint:
            print("[SETUP] Keeping checkpoint runtime scaling unchanged")
        else:
            self._apply_per_env_scaling()

        self._validate_resolved_configuration()

    def _validate_resolved_configuration(self):
        """Fail before environment construction when the experiment is inconsistent."""
        params = self.config.params
        model_type = str(params.get("human_model_type", "learnable"))
        supported = {"learnable", "fixed_impedance", "residual_impedance"}
        if model_type not in supported:
            raise ValueError(
                f"Unsupported human_model_type={model_type!r}; expected {sorted(supported)}"
            )

        impedance = params.get("human_impedance", {})
        for gain_name in ("kp", "kd"):
            gains = impedance.get(gain_name, [0.8, 0.8, 0.8] if gain_name == "kp" else [0.1, 0.1, 0.1])
            if not isinstance(gains, (list, tuple)) or len(gains) != 3:
                raise ValueError(f"human_impedance.{gain_name} must contain three values")
        if float(impedance.get("lookahead_distance", 0.04)) <= 0.0:
            raise ValueError("human_impedance.lookahead_distance must be positive")
        if float(impedance.get("reference_speed", 0.02)) < 0.0:
            raise ValueError("human_impedance.reference_speed must be non-negative")

        constraints = params.get("constraints", {})
        for limit_name in ("max_human_force", "max_robot_force"):
            if float(constraints.get(limit_name, 0.04)) <= 0.0:
                raise ValueError(f"constraints.{limit_name} must be positive")
        max_human_force = float(constraints.get("max_human_force", 0.04))
        impedance_limit = float(impedance.get("max_force", max_human_force))
        residual_limit = float(
            params.get("human_residual", {}).get("max_force", max_human_force)
        )
        if not 0.0 < impedance_limit <= max_human_force:
            raise ValueError(
                "human_impedance.max_force must be in (0, max_human_force]"
            )
        if not 0.0 < residual_limit <= max_human_force:
            raise ValueError(
                "human_residual.max_force must be in (0, max_human_force]"
            )

        force_scaling = params.get("force_scaling", {})
        for agent_name in ("human", "robot"):
            factor_name = f"{agent_name}_factor"
            limit_name = f"max_{agent_name}_force"
            factor = float(force_scaling.get(factor_name, 0.0))
            limit = float(constraints.get(limit_name, 0.04))
            if factor <= 0.0:
                raise ValueError(f"force_scaling.{factor_name} must be positive")
            if not np.isclose(factor * limit, 1.0, rtol=1e-6, atol=1e-6):
                raise ValueError(
                    f"force_scaling.{factor_name} must equal 1/{limit_name}; "
                    f"got {factor} * {limit}"
                )

        obs_factors = params.get("obs_scaling", {}).get("factors", [])
        if len(obs_factors) != 9:
            raise ValueError("obs_scaling.factors must contain 9 values")
        human_factor = float(force_scaling["human_factor"])
        robot_factor = float(force_scaling["robot_factor"])
        if not np.isclose(human_factor, robot_factor, rtol=1e-6, atol=1e-6):
            raise ValueError(
                "The shared observation scaler requires equal human/robot force factors"
            )
        if not np.allclose(obs_factors[-3:], human_factor, rtol=1e-6, atol=1e-6):
            raise ValueError(
                "The final three obs_scaling factors must match force_scaling"
            )

        trajectory = params.get("trajectory", {})
        start = np.asarray(trajectory.get("start_point"), dtype=np.float64)
        end = np.asarray(trajectory.get("end_point"), dtype=np.float64)
        if start.shape != (3,) or end.shape != (3,) or np.allclose(start, end):
            raise ValueError("trajectory start_point/end_point must be distinct 3D points")
        if self.args.num_envs <= 0:
            raise ValueError("num_envs must be positive")

        print(f"[SETUP] Resolved configuration validated ({model_type})")

    def _apply_per_env_scaling(self):
        """
        Scale YAML parameters by number of environments.
        Scales: min_buffer_size, max_replay_buffer_len, milestone_episodes
        """
        num_envs = int(self.args.num_envs)

        maddpg_cfg = self.config.params.get("maddpg_config", {})
        monitor_cfg = self.config.params.get("training_monitor", {})

        base_min_buffer = int(maddpg_cfg.get("min_buffer_size", 3600))
        base_max_buffer = int(maddpg_cfg.get("max_replay_buffer_len", 24000))
        base_milestones = list(monitor_cfg.get("milestone_episodes", []))

        scaled_min_buffer = base_min_buffer * num_envs
        scaled_max_buffer = base_max_buffer * num_envs
        scaled_milestones = [int(m * num_envs) for m in base_milestones]

        maddpg_cfg["min_buffer_size"] = int(scaled_min_buffer)
        maddpg_cfg["max_replay_buffer_len"] = int(scaled_max_buffer)
        self.config.params["maddpg_config"] = maddpg_cfg

        monitor_cfg["milestone_episodes"] = scaled_milestones
        self.config.params["training_monitor"] = monitor_cfg

        print("[SETUP][PER-ENV SCALING]")
        print(f"  num_envs = {num_envs}")
        print(f"  min_buffer_size: {base_min_buffer} → {scaled_min_buffer}")
        print(f"  max_replay_buffer_len: {base_max_buffer} → {scaled_max_buffer}")
        print(f"  milestone_episodes: {base_milestones} → {scaled_milestones}")

    def _setup_environment(self):
        """Create and configure the environment."""
        print(f"[SETUP] Creating environment: {self.args.task}")
        self.env, self.env_cfg = setup_environment(self.args, self.config)

        # Inject StepTracer for console logging
        inject_step_tracer(self.env, self.config, self.args.num_envs)

    def _setup_logging_and_wandb(self):
        """Setup logging directory and WandB integration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        human_model_type = self.config.params.get("human_model_type", "learnable")
        self.log_dir = self.args.run_dir or f"logs/maddpg_dual/{human_model_type}/{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"[SETUP] Log directory created: {self.log_dir}")
        print(f"[SETUP] Checkpoints will be stored in: {self.checkpoint_dir}")

        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                text=True,
            ).strip()
        except Exception:
            git_commit = "unknown"

        self.config.params["git_commit"] = git_commit
        resolved_config_path = os.path.join(self.log_dir, "resolved_config.yaml")
        with open(resolved_config_path, "w", encoding="utf-8") as config_file:
            yaml.safe_dump(
                self.config.params,
                config_file,
                sort_keys=False,
                allow_unicode=True,
            )
        manifest = {
            "algorithm": "maddpg",
            "human_model_type": human_model_type,
            "seed": self.args.seed,
            "num_envs": self.args.num_envs,
            "max_global_steps": self.config.params["maddpg_config"]["max_global_steps"],
            "git_commit": git_commit,
            "resolved_config": resolved_config_path,
        }
        with open(os.path.join(self.log_dir, "run_manifest.json"), "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)

        # Initialize WandB
        self.wandb_logger = WandBLogger(enabled=self.args.wandb)
        if self.wandb_logger.enabled:
            actual_env = getattr(self.env, "unwrapped", self.env)
            human_metadata = actual_env.human_force_controller.wandb_metadata()
            run_config = build_maddpg_wandb_config(
                resolved_config=self.config.params,
                runtime={
                    "seed": self.args.seed,
                    "num_envs": self.args.num_envs,
                    "max_global_steps": self.config.params["maddpg_config"]["max_global_steps"],
                },
                human_metadata=human_metadata,
                git_commit=git_commit,
            )
            self.wandb_run_config = run_config
            run_name = f"maddpg_{human_model_type}_seed{self.args.seed}_{timestamp}"
            self.wandb_logger.initialize_run(run_config, run_name)
            print(f"[SETUP] WandB initialized with run name: {run_name}")
        else:
            print(f"[SETUP] WandB disabled")

    def _setup_training_components(self):
        """Initialize training components and metrics hub."""
        print(f"[SETUP] Setting up training components...")
        
        # Create MetricsHub
        self.metrics_hub = MetricsHub()
        
        # Connect WandB to MetricsHub
        if self.wandb_logger.enabled:
            self.wandb_logger.attach_metrics_hub(self.metrics_hub)
            print(f"[SETUP] WandB attached to MetricsHub")
        
        # Create MADDPG algorithm
        self.maddpg = initialize_maddpg_algorithm(self.env, self.config, self.args)
        
        # Unified max_global_steps logic: CLI takes priority over YAML
        maddpg_cfg = self.config.params.get('maddpg_config', {})
        yaml_max_steps = int(maddpg_cfg.get('max_global_steps', 200000))
        
        # Check if CLI provided a value (0 means not specified)
        if self.args.max_global_steps > 0:
            self.max_global_steps = self.args.max_global_steps
            print(f"[SETUP] Using CLI max_global_steps: {self.max_global_steps}")
        else:
            self.max_global_steps = yaml_max_steps
            print(f"[SETUP] Using YAML max_global_steps: {self.max_global_steps}")
        
        # Ensure we have a valid value
        if self.max_global_steps <= 0:
            self.max_global_steps = 200000  # Default fallback
            print(f"[SETUP] Using default max_global_steps: {self.max_global_steps}")
        
        self.milestone_episodes = self.config.params.get('training_monitor', {}).get('milestone_episodes', [])
        
        print(f"[SETUP] Training components configured:")
        print(f"  Max global steps: {self.max_global_steps}")
        print(f"  Milestone episodes: {self.milestone_episodes}")

    def _setup_runners_and_evaluators(self):
        """Initialize training runner and milestone evaluator."""
        print(f"[SETUP] Creating TrainingRunner and MilestoneEvaluator...")

        # Pass the resolved max_global_steps to TrainingRunner
        self.runner = TrainingRunner(
            env=self.env,
            maddpg=self.maddpg,
            metrics_hub=self.metrics_hub,
            agent_ids=self.maddpg.agent_ids,
            max_global_steps=self.max_global_steps  # Pass the resolved value
        )
        
        # Create MilestoneEvaluator
        self.evaluator = MilestoneEvaluator(
            env=self.env,
            maddpg=self.maddpg,
            metrics_hub=self.metrics_hub,
            log_dir=self.log_dir,
            agent_ids=self.maddpg.agent_ids,
            runner=self.runner,
        )
        
        print(f"[SETUP] TrainingRunner and MilestoneEvaluator created successfully")

    def _resume_if_requested(self):
        """Resume training from checkpoint if --checkpoint supplied."""
        if not getattr(self.args, "checkpoint", None):
            return
        resume_from_checkpoint_maddpg(
            self.args.checkpoint,
            self.maddpg,
            runner=self.runner,
            device=self.config.get_compute_device(),
        )
        if self.wandb_logger.enabled and self.wandb_logger.run is not None:
            self.wandb_logger.run.config.update(
                self.wandb_run_config,
                allow_val_change=True,
            )
            print("[SETUP] WandB config synchronized with resumed resolved config")
        print(f"[SETUP] Resumed from checkpoint: {self.args.checkpoint}")

    def _setup_milestone_management(self):
        """Initialize milestone tracking variables."""
        print(f"[SETUP] Setting up milestone management...")
        self.max_milestone_triggered = 0  # Highest milestone reached
        print(f"[SETUP] Milestone management configured for {len(self.milestone_episodes)} milestones")

    def _print_configuration_summary(self):
        """Print network and exploration configuration summary."""
        networks_cfg = self.config.params.get('networks', {})
        if networks_cfg:
            print(f"[CONFIG] Network Architecture:")
            actor_cfg = networks_cfg.get('actor', {})
            critic_cfg = networks_cfg.get('critic', {})
            print(f"  Actor layers: {actor_cfg.get('hidden_layers', 'default')}")
            print(f"  Critic layers: {critic_cfg.get('hidden_layers', 'default')}")
            print(f"  Orthogonal init: {actor_cfg.get('orthogonal_init', False)}")
            print(f"  Gains: Hidden {actor_cfg.get('ortho_gain_hidden', 1.0)}, Output {actor_cfg.get('ortho_gain_output', 0.01)}")
        
        exploration_cfg = self.config.params.get('exploration', {})
        if exploration_cfg:
            print(f"[CONFIG] Exploration Schedule:")
            print(f"  Noise range: {exploration_cfg.get('sigma_start', 0.7)} → {exploration_cfg.get('sigma_end', 0.1)}")
            print(f"  Decay rate: {exploration_cfg.get('decay_k', 6.0)}")

    def set_eval_mode(self, is_eval: bool):
        """Set evaluation mode for MADDPG (disables buffer/parameter updates)."""
        self.maddpg.set_eval_mode(is_eval)
        mode_str = "EVALUATION" if is_eval else "TRAINING"
        print(f"[TRAINER] Mode set to: {mode_str}")

    def evaluate_milestone_if_due(self):
        """Check and trigger milestone evaluation if threshold crossed."""
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
            
            # Set eval mode before evaluation
            self.set_eval_mode(True)
            
            try:
                # Direct call to evaluator
                result = self.evaluator.run_evaluation(candidate, self.runner.global_step)
                if result.get("skip_episode_once", False):
                    self.runner.mark_skip_episode_once()
            finally:
                # Always restore training mode
                self.set_eval_mode(False)
            
            # Refresh current observations after evaluation
            env = getattr(self.env, "unwrapped", self.env)
            if hasattr(env, "_get_observations"):
                self.runner._current_obs = env._get_observations()
            else:
                obs_dict, _ = self.env.reset()
                self.runner._current_obs = obs_dict
            print(f"[MILESTONE] Refreshed current observations after evaluation")
            
            self.max_milestone_triggered = candidate
            print(f"[MILESTONE] Updated max_milestone_triggered to {self.max_milestone_triggered}")

    def train(self) -> None:
        """Main training loop with milestone evaluation and exit tracking."""
        print(f"[TRAIN] Starting training with persistent generators and noise scheduling:")
        print(f"  - TrainingRunner: rollout→replay→update→log→count + exponential noise decay")
        print(f"  - MilestoneEvaluator: milestone→evaluation→checkpoint")
        print(f"  - Network layers configurable via YAML")
        print(f"  - Noise schedule: σ_start={self.runner.sigma_start} → σ_end={self.runner.sigma_end}")
        print(f"  - Max steps: {self.max_global_steps}")
        print(f"  - Per-env scaling applied: buffer sizes and milestones auto-scaled by {self.args.num_envs}x")
        print(f"  - Persistent generators for reproducible yet diverse per-(agent,env) noise")
        
        reason = "UNKNOWN"  # Track exit reason for debugging
        
        try:
            # Initialize WandB data points
            if self.wandb_logger.enabled:
                initial_stats = {
                    "train/episodes_done": 0,
                    "replay/buffer_size": 0,
                    "exploration/noise_scale": self.runner.sigma_start,
                }
                self.metrics_hub.push_update(0, initial_stats)
            
            # Reset environment
            print(f"[TRAIN] Resetting environment...")
            obs_dict, _ = self.env.reset()
            print(f"[TRAIN] Environment reset complete")
            self.runner._current_obs = obs_dict  
            
            # Main training loop
            while self.runner.global_step < self.max_global_steps:
                # Execute one training step
                obs_dict = self.runner.execute_training_step()
                
                # Milestone checking
                self.evaluate_milestone_if_due()
                
                # Progress reporting
                if self.runner.global_step % 2000 == 0:
                    print(f"[Step {self.runner.global_step}] Episodes so far: {self.runner.global_episodes}")
                
                # Periodic sanity check
                if (self.runner.global_step % 1000) == 0:
                    assert self.runner.global_step <= self.max_global_steps, \
                        f"Step overflow: {self.runner.global_step} > {self.max_global_steps}"
                
                # Check termination condition
                if self.runner.global_step >= self.max_global_steps:
                    print(f"\n[TRAINING LIMIT] Reached max_global_steps={self.max_global_steps}")
                    break
            
            reason = "REACHED_MAX_STEPS"  # Normal completion
            
            # Training complete
            print(f"\n[TRAINING COMPLETE]")
            print(f"  Total steps: {self.runner.global_step}")
            print(f"  Total episodes: {self.runner.global_episodes}")
            print(f"  Max milestone triggered: {self.max_milestone_triggered}")
            print(f"  Final noise scale: {self.runner._calculate_noise_scale():.4f}")
            save_final_checkpoint_maddpg(
                self.maddpg, self.runner, self.checkpoint_dir
            )
            
            print("\n" + "=" * 70, "\nTraining Complete!\n" + "=" * 70)
            print(f"\nResults saved in: {self.log_dir}")
        
        except KeyboardInterrupt:
            reason = "KEYBOARD_INTERRUPT"
            print(f"\nTraining interrupted by user")
            raise  # Re-raise for proper handling
        except Exception as e:
            reason = f"EXCEPTION:{type(e).__name__}:{str(e)[:50]}"
            print(f"\nTraining failed with exception: {e}")
            raise  # Re-raise for proper handling
        finally:
            # ALWAYS print exit information for debugging
            print(f"\n[TRAINING EXIT]")
            print(f"  Reason: {reason}")
            print(f"  Global step: {self.runner.global_step} / {self.max_global_steps}")
            print(f"  Global episodes: {self.runner.global_episodes}")
            print(f"  Max milestone: {self.max_milestone_triggered}")
            
            # Cleanup resources
            self.env.close()
            self.wandb_logger.finalize_run()
            print("[TRAIN] Cleanup completed")
            print("\nPersistent generator training with per-env scaling completed")


def main():
    """Main entry point for MADDPG training."""
    print("="*80)
    print("MADDPG Persistent Generator Training with Per-Env Scaling")
    print("Enhanced with reproducible yet diverse per-(agent,env) noise generation")
    print("="*80)
    
    # Parse arguments
    parser = create_argument_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Resolve and apply the experiment seed before Isaac Sim is launched.
    args_cli.seed = resolve_startup_seed(args_cli)
    setup_global_reproducibility(args_cli.seed, strict_determinism=True)
    
    print(f"[MAIN] Arguments parsed:")
    print(f"  Task: {args_cli.task}")
    print(f"  Environments: {args_cli.num_envs if args_cli.num_envs is not None else 'checkpoint/default'}")
    print(f"  Max steps: {args_cli.max_global_steps if args_cli.max_global_steps > 0 else 'from YAML'}")
    print(f"  WandB: {args_cli.wandb}")
    print(f"  Config: {args_cli.config}")
    print(f"  Per-env scaling: YAML configs will be auto-scaled by {args_cli.num_envs}x")
    
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
