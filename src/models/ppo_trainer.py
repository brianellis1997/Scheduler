"""
PPO Training Script for AEOS Environment

Trains a Proximal Policy Optimization agent on the satellite scheduling task.
"""

import os
import argparse
import numpy as np
import yaml
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym

from ..environment.aeos_env import AEOSEnv


def create_env(config: dict, seed: int = None) -> gym.Env:
    """Create and return AEOS environment"""
    return AEOSEnv(config=config, seed=seed)


def train_ppo(config_path: str = None, output_dir: str = "logs") -> None:
    """
    Train PPO agent on AEOS environment

    Args:
        config_path: Path to configuration file
        output_dir: Directory to save logs and models
    """
    # Load configuration
    config = load_config(config_path)

    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_dir = Path(output_dir) / "models"
    model_dir.mkdir(exist_ok=True)
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("AEOS-RL: PPO Training")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Configuration: {config}")
    print("-" * 60)

    # Create training environment
    train_env = create_env(config["environment"], seed=config.get("seed", 0))
    train_env = DummyVecEnv([lambda: train_env])

    # Create evaluation environment
    eval_env = create_env(config["environment"], seed=config.get("seed", 1))
    eval_env = DummyVecEnv([lambda: eval_env])

    # Optionally wrap both with VecNormalize for observation normalization
    if config.get("normalize_observations", True):
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config["training"].get("checkpoint_freq", 10000),
        save_path=str(model_dir),
        name_prefix="aeos_ppo",
    )

    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=config["training"].get("eval_episodes", 5),
        eval_freq=config["training"].get("eval_freq", 5000),
        log_path=str(log_dir),
        best_model_save_path=str(model_dir),
        deterministic=False,
    )

    # Create PPO agent
    model_kwargs = config["ppo_params"]
    learning_rate = model_kwargs.pop("learning_rate", 3e-4)
    batch_size = model_kwargs.pop("batch_size", 64)
    n_epochs = model_kwargs.pop("n_epochs", 10)
    gamma = model_kwargs.pop("gamma", 0.99)
    gae_lambda = model_kwargs.pop("gae_lambda", 0.95)
    clip_range = model_kwargs.pop("clip_range", 0.2)
    ent_coef = model_kwargs.pop("ent_coef", 0.0)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=config.get("seed", 0),
    )

    print(f"Model: {model}")
    print("-" * 60)

    # Train
    total_timesteps = config["training"].get("total_timesteps", 100000)
    print(f"Training for {total_timesteps:,} timesteps...")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            log_interval=config["training"].get("log_interval", 10),
        )

        # Save final model
        final_model_path = model_dir / "aeos_ppo_final"
        model.save(str(final_model_path))
        print(f"Model saved to {final_model_path}")

        if isinstance(train_env, VecNormalize):
            train_env.save(str(model_dir / "vec_normalize.pkl"))
            print(f"Normalization stats saved")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        train_env.close()
        eval_env.close()


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to configuration file. If None, uses default config.

    Returns:
        Configuration dictionary
    """
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    return {
        "seed": 0,
        "environment": {
            "episode_duration_s": 5400,
            "timestep_s": 60,
            "max_tasks": 20,
            "num_initial_tasks": 10,
            "reward_weights": {
                "task_completion": 1.0,
                "priority_bonus": 0.5,
                "energy_penalty": 0.1,
                "memory_penalty": 0.2,
                "latency_penalty": 0.05,
            },
        },
        "training": {
            "total_timesteps": 100000,
            "checkpoint_freq": 10000,
            "eval_freq": 5000,
            "eval_episodes": 5,
            "log_interval": 10,
        },
        "ppo_params": {
            "learning_rate": 3e-4,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
        },
        "normalize_observations": True,
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train PPO agent on AEOS satellite scheduling task"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs",
        help="Output directory for logs and models",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override timesteps if provided
    if args.timesteps:
        config["training"]["total_timesteps"] = args.timesteps

    # Train
    train_ppo(config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
