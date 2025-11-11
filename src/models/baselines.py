"""
Baseline Algorithms for Satellite Scheduling

Implements simple heuristic baselines for comparison with RL agents:
- Random policy
- Greedy policy (select high-priority tasks)
- Earliest Deadline First (EDF)
"""

import numpy as np
from typing import Tuple
import gymnasium as gym


class RandomPolicy:
    """Random action selection baseline"""

    def __init__(self, env: gym.Env):
        """
        Initialize random policy

        Args:
            env: Gymnasium environment
        """
        self.env = env

    def get_action(self, observation: np.ndarray) -> int:
        """
        Get random action

        Args:
            observation: Current observation (unused)

        Returns:
            Random action from action space
        """
        return self.env.action_space.sample()


class GreedyPolicy:
    """
    Greedy policy: prioritizes high-value tasks

    Selects actions based on task priority and deadline urgency.
    """

    def __init__(self, env: gym.Env):
        """
        Initialize greedy policy

        Args:
            env: Gymnasium environment with task queue
        """
        self.env = env

    def get_action(self, observation: np.ndarray) -> int:
        """
        Get greedy action based on task priority

        Observation format:
        [sat_lat, sat_lon, battery_soc, storage_util, time_norm,
         task0_lat, task0_lon, task0_priority, task0_deadline_urgency, ...]

        Args:
            observation: Current observation

        Returns:
            Action index (1-20 for observing a task, 0 for idle)
        """
        # Extract task features from observation
        base_features = 5
        max_tasks = 20

        # Find highest priority pending task
        best_task_idx = -1
        best_score = -np.inf

        for i in range(max_tasks):
            task_idx = base_features + i * 4

            if task_idx + 3 >= len(observation):
                break

            task_priority = observation[task_idx + 2]
            deadline_urgency = observation[task_idx + 3]

            # Score: combination of priority and deadline urgency
            score = task_priority * 0.7 + deadline_urgency * 0.3

            if score > best_score and score > 0.01:  # Non-zero task
                best_score = score
                best_task_idx = i

        # Check battery and storage constraints
        battery_soc = observation[2]
        storage_util = observation[3]

        if battery_soc < 0.1 or storage_util > 0.9:
            # Cannot observe - try downlink or idle
            if storage_util > 0.5:
                # Try to downlink
                return self.env.action_space.n - 2
            else:
                return 0  # Idle

        if best_task_idx >= 0:
            return best_task_idx + 1  # Actions 1-20 are task observations
        else:
            return 0  # Idle if no good tasks


class EarliestDeadlineFirstPolicy:
    """
    Earliest Deadline First (EDF) policy

    Prioritizes tasks closest to their deadlines.
    """

    def __init__(self, env: gym.Env):
        """
        Initialize EDF policy

        Args:
            env: Gymnasium environment
        """
        self.env = env

    def get_action(self, observation: np.ndarray) -> int:
        """
        Get EDF action based on deadline urgency

        Args:
            observation: Current observation

        Returns:
            Action index for earliest deadline task
        """
        # Extract task features
        base_features = 5
        max_tasks = 20

        # Find task with highest deadline urgency (closest deadline)
        best_task_idx = -1
        best_urgency = -np.inf

        for i in range(max_tasks):
            task_idx = base_features + i * 4

            if task_idx + 3 >= len(observation):
                break

            deadline_urgency = observation[task_idx + 3]

            if deadline_urgency > best_urgency and deadline_urgency > 0.01:
                best_urgency = deadline_urgency
                best_task_idx = i

        # Check constraints
        battery_soc = observation[2]
        storage_util = observation[3]

        if battery_soc < 0.1 or storage_util > 0.9:
            if storage_util > 0.5:
                return self.env.action_space.n - 2  # Downlink
            else:
                return 0  # Idle

        if best_task_idx >= 0:
            return best_task_idx + 1  # Action for this task
        else:
            return 0  # Idle


class EnergyAwarePolicy:
    """
    Energy-aware policy: manages battery to maximize task completion

    Prioritizes charging when battery is low, balances task completion with energy.
    """

    def __init__(self, env: gym.Env):
        """
        Initialize energy-aware policy

        Args:
            env: Gymnasium environment
        """
        self.env = env

    def get_action(self, observation: np.ndarray) -> int:
        """
        Get energy-aware action

        Args:
            observation: Current observation

        Returns:
            Action that balances energy and task completion
        """
        battery_soc = observation[2]
        storage_util = observation[3]

        # If battery critically low, prioritize charging/idling
        if battery_soc < 0.2:
            return 0  # Idle to charge

        # If storage critically full, downlink
        if storage_util > 0.8:
            return self.env.action_space.n - 2  # Downlink

        # Otherwise, use greedy task selection
        greedy_policy = GreedyPolicy(self.env)
        return greedy_policy.get_action(observation)


def evaluate_policy(policy, env: gym.Env, num_episodes: int = 10) -> Tuple[float, float]:
    """
    Evaluate policy performance

    Args:
        policy: Policy object with get_action(observation) method
        env: Gymnasium environment
        num_episodes: Number of episodes to evaluate

    Returns:
        (mean_reward, std_reward) tuple
    """
    episode_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        episode_rewards.append(episode_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def compare_baselines(env_class, env_config: dict, num_episodes: int = 5) -> dict:
    """
    Compare all baseline policies

    Args:
        env_class: Environment class to instantiate
        env_config: Configuration for environment
        num_episodes: Episodes per policy evaluation

    Returns:
        Dictionary of policy_name -> (mean_reward, std_reward)
    """
    results = {}

    policies = {
        "Random": RandomPolicy,
        "Greedy": GreedyPolicy,
        "EDF": EarliestDeadlineFirstPolicy,
        "Energy-Aware": EnergyAwarePolicy,
    }

    for policy_name, PolicyClass in policies.items():
        env = env_class(config=env_config)
        policy = PolicyClass(env)

        mean_reward, std_reward = evaluate_policy(policy, env, num_episodes)
        results[policy_name] = (mean_reward, std_reward)

        print(f"{policy_name:15s}: {mean_reward:8.3f} ± {std_reward:6.3f}")
        env.close()

    return results
