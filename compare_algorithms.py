#!/usr/bin/env python3
"""
Algorithm Comparison Script

Compares all baseline algorithms and trained PPO agent on the AEOS environment.
Generates summary report and visualizations.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.environment.aeos_env import AEOSEnv
from src.models.baselines import (
    RandomPolicy, GreedyPolicy, EarliestDeadlineFirstPolicy,
    EnergyAwarePolicy, evaluate_policy
)
from stable_baselines3 import PPO


def run_comparison(num_episodes: int = 10):
    """Run comprehensive algorithm comparison"""

    # Environment configuration
    config = {
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
    }

    results = {}

    # Evaluate baseline algorithms
    baseline_policies = {
        "Random": RandomPolicy,
        "Greedy": GreedyPolicy,
        "EDF": EarliestDeadlineFirstPolicy,
        "Energy-Aware": EnergyAwarePolicy,
    }

    print("=" * 70)
    print("AEOS-RL: Algorithm Comparison")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Episodes per algorithm: {num_episodes}")
    print(f"  Episode duration: {config['episode_duration_s']} seconds")
    print(f"  Initial tasks: {config['num_initial_tasks']}\n")

    print("Evaluating Baseline Algorithms:")
    print("-" * 70)
    print(f"{'Algorithm':<20} {'Mean Reward':>15} {'Std Dev':>15} {'Min':>10} {'Max':>10}")
    print("-" * 70)

    for policy_name, PolicyClass in baseline_policies.items():
        env = AEOSEnv(config=config, seed=42)
        policy = PolicyClass(env)

        # Run episodes and collect detailed stats
        episode_rewards = []
        for _ in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0.0
            terminated = False

            while not terminated:
                action = policy.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

            episode_rewards.append(episode_reward)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        min_reward = np.min(episode_rewards)
        max_reward = np.max(episode_rewards)

        results[policy_name] = {
            "mean": mean_reward,
            "std": std_reward,
            "min": min_reward,
            "max": max_reward,
            "episodes": episode_rewards,
        }

        print(f"{policy_name:<20} {mean_reward:>15.3f} {std_reward:>15.3f} {min_reward:>10.3f} {max_reward:>10.3f}")
        env.close()

    # Evaluate trained PPO agent
    print("-" * 70)
    model_path = project_root / "logs" / "models" / "aeos_ppo_final.zip"

    if model_path.exists():
        print(f"\nEvaluating Trained PPO Agent:")
        print("-" * 70)

        model = PPO.load(str(model_path.with_suffix('')))
        ppo_rewards = []

        for _ in range(num_episodes):
            env = AEOSEnv(config=config, seed=42)
            obs, _ = env.reset()
            episode_reward = 0.0
            terminated = False

            while not terminated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

            ppo_rewards.append(episode_reward)
            env.close()

        ppo_mean = np.mean(ppo_rewards)
        ppo_std = np.std(ppo_rewards)
        ppo_min = np.min(ppo_rewards)
        ppo_max = np.max(ppo_rewards)

        results["PPO (Trained)"] = {
            "mean": ppo_mean,
            "std": ppo_std,
            "min": ppo_min,
            "max": ppo_max,
            "episodes": ppo_rewards,
        }

        print(f"{'PPO (Trained)':<20} {ppo_mean:>15.3f} {ppo_std:>15.3f} {ppo_min:>10.3f} {ppo_max:>10.3f}")
    else:
        print(f"\n⚠ Trained model not found at {model_path}")
        print("Skipping PPO evaluation")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find best algorithm
    best_algo = max(results.items(), key=lambda x: x[1]["mean"])
    print(f"\nBest performing algorithm: {best_algo[0]}")
    print(f"  Mean reward: {best_algo[1]['mean']:.3f} ± {best_algo[1]['std']:.3f}")

    # Comparison to baselines
    if "PPO (Trained)" in results:
        baseline_results = {k: v for k, v in results.items() if k != "PPO (Trained)"}
        if baseline_results:
            best_baseline_name = max(
                baseline_results.items(),
                key=lambda x: x[1]["mean"],
            )[0]
            best_baseline = baseline_results[best_baseline_name]
            improvement = (
                (results["PPO (Trained)"]["mean"] - best_baseline["mean"]) /
                abs(best_baseline["mean"]) * 100
            )
            print(f"\nPPO vs Best Baseline ({best_baseline_name}):")
            print(f"  Improvement: {improvement:+.1f}%")

    # Save results
    results_df = pd.DataFrame({
        algo: {
            "mean": data["mean"],
            "std": data["std"],
            "min": data["min"],
            "max": data["max"],
        }
        for algo, data in results.items()
    }).T

    csv_path = project_root / "comparison_results.csv"
    results_df.to_csv(csv_path)
    print(f"\n✓ Results saved to {csv_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare AEOS-RL algorithms")
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes per algorithm",
    )
    args = parser.parse_args()

    run_comparison(num_episodes=args.episodes)
