"""
AEOS Environment Implementation

Gymnasium-compatible environment for Agile Earth Observation Satellite scheduling.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional

from ..simulation.satellite import SatelliteModel, SatelliteState
from ..simulation.ground_station import GroundStationManager, GroundStation
from ..simulation.tasks import TaskGenerator, TaskQueue, TaskStatus


class AEOSEnv(gym.Env):
    """
    Agile Earth Observation Satellite (AEOS) Scheduling Environment

    Environment for scheduling observation tasks for a single LEO satellite
    while respecting constraints on power, memory, and ground station visibility.

    State Space:
    - Satellite position (latitude, longitude)
    - Battery state of charge
    - Data storage utilization
    - Task queue (up to 20 tasks with position, priority, deadline)
    - Time in episode

    Action Space (Discrete):
    - 0: Idle (wait)
    - 1-20: Select and observe task (if feasible)
    - 21: Downlink data (if ground station visible)
    - 22: Navigate to next high-priority task
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        """
        Initialize AEOS Environment

        Args:
            config: Configuration dictionary with environment parameters
            seed: Random seed for reproducibility
        """
        self.config = config or {}
        self.np_random = np.random.default_rng(seed)

        # Extract configuration parameters
        self.episode_duration_s = self.config.get("episode_duration_s", 5400)  # 90 min
        self.timestep_s = self.config.get("timestep_s", 60)  # 1 minute steps
        self.max_tasks = self.config.get("max_tasks", 20)
        self.num_initial_tasks = self.config.get("num_initial_tasks", 10)

        # Reward weights for shaping
        self.reward_weights = self.config.get("reward_weights", {
            "task_completion": 1.0,
            "priority_bonus": 0.5,
            "energy_penalty": 0.1,
            "memory_penalty": 0.2,
            "latency_penalty": 0.05,
        })

        # Initialize simulator components
        self.satellite_model = SatelliteModel()
        self.ground_station_manager = GroundStationManager()
        self.task_generator = TaskGenerator(seed=seed)
        self.task_queue = TaskQueue()

        # State variables
        self.satellite_state: Optional[SatelliteState] = None
        self.current_time_s = 0.0
        self.episode_step = 0
        self.episode_tasks = []

        # Define action and observation spaces
        self._setup_spaces()

    def _setup_spaces(self) -> None:
        """Set up action and observation spaces"""
        # Action space: idle, observe task [1-20], downlink, navigate
        self.action_space = spaces.Discrete(self.max_tasks + 3)

        # Observation space: [satellite_lat, satellite_lon, battery_soc, storage_util,
        #                      time_normalized, task_queue_features...]
        # Task queue: (lat, lon, priority, time_until_deadline) * max_tasks
        base_features = 5  # lat, lon, battery, storage, time
        task_features = 4 * self.max_tasks  # (lat, lon, priority, deadline_urgency)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_features + task_features,),
            dtype=np.float32,
        )

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.current_time_s = 0.0
        self.episode_step = 0

        # Initialize satellite at equator, random longitude
        self.satellite_state = SatelliteState(
            epoch=0.0,
            altitude_km=500.0,
            inclination_deg=97.5,
            longitude_deg=self.np_random.uniform(-180, 180),
            latitude_deg=0.0,
            battery_soc=1.0,
        )

        # Generate initial task set
        self.task_queue = TaskQueue()
        initial_tasks = self.task_generator.generate_tasks(
            num_tasks=self.num_initial_tasks,
            current_time_s=self.current_time_s,
            episode_duration_s=self.episode_duration_s,
        )
        self.task_queue.add_tasks(initial_tasks)

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step of the environment dynamics

        Args:
            action: Action index from action space

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.episode_step += 1
        self.current_time_s += self.timestep_s

        # Determine if satellite is sunlit (simplified: half orbit)
        phase = (self.current_time_s % self.satellite_model.orbital_period_s) / \
                self.satellite_model.orbital_period_s
        is_sunlit = phase < (1.0 - self.satellite_model.eclipse_fraction)

        # Propagate satellite dynamics
        self.satellite_state = self.satellite_model.propagate(
            self.satellite_state, self.timestep_s, is_sunlit
        )

        # Check ground station visibility
        visibility = self.ground_station_manager.check_visibility(
            self.satellite_state.latitude_deg,
            self.satellite_state.longitude_deg,
            self.satellite_state.altitude_km,
        )
        has_ground_contact = any(v[1] for v in visibility)

        # Process action
        reward = self._process_action(action, has_ground_contact)

        # Check termination conditions
        terminated = self.current_time_s >= self.episode_duration_s
        truncated = False

        # Add energy penalty for each step
        if self.satellite_state.battery_soc < 0.2:
            reward -= self.reward_weights["energy_penalty"] * 5  # Penalize low battery

        observation = self._get_observation()
        info = self._get_info()

        return observation, float(reward), terminated, truncated, info

    def _process_action(self, action: int, has_ground_contact: bool) -> float:
        """
        Process action and return immediate reward

        Args:
            action: Action index
            has_ground_contact: Whether satellite has ground station contact

        Returns:
            Reward for this action
        """
        reward = 0.0

        if action == 0:
            # Idle action
            pass

        elif action <= self.max_tasks:
            # Observe task
            task_idx = action - 1
            pending_tasks = self.task_queue.get_pending_tasks()

            if task_idx < len(pending_tasks):
                task = pending_tasks[task_idx]

                # Check if we can actually observe this task
                # (Simplified: assume we can if we have battery and storage)
                if self.satellite_state.battery_soc > 0.1 and \
                   self.satellite_state.data_storage_gb < \
                   (0.9 * self.satellite_state.storage_capacity_gb):

                    # Perform observation
                    target_attitude = (45.0, 0.0, 0.0)  # Simplified pointing
                    _, is_pointing = self.satellite_model.slew_to_target(
                        target_attitude,
                        task.target_latitude_deg,
                        task.target_longitude_deg,
                        self.timestep_s,
                    )

                    if is_pointing:
                        # Generate data
                        data_generated = self.satellite_model.observe_target(
                            self.satellite_state, is_pointing, task.required_duration_s
                        )

                        # Update storage
                        self.satellite_state.data_storage_gb += data_generated

                        # Mark task as completed
                        if self.task_queue.complete_task(
                            task.task_id, self.current_time_s
                        ):
                            # Compute reward for completion
                            task_value = self.task_generator.compute_task_value(
                                task, self.current_time_s
                            )
                            reward += task_value * self.reward_weights["task_completion"]

                            # Priority bonus
                            if task.priority > 0.7:
                                reward += self.reward_weights["priority_bonus"]

        elif action == self.max_tasks + 1:
            # Downlink data
            if has_ground_contact and self.satellite_state.data_storage_gb > 0:
                downlinked, remaining = self.ground_station_manager.downlink_data(
                    self.satellite_state.data_storage_gb, self.timestep_s
                )
                self.satellite_state.data_storage_gb = remaining
                reward += downlinked * 0.1  # Small reward for downlinking

        # Storage penalty
        storage_util = self.satellite_state.data_storage_gb / \
                      self.satellite_state.storage_capacity_gb
        if storage_util > 0.8:
            reward -= self.reward_weights["memory_penalty"]

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs = []

        # Normalize position
        obs.append(self.satellite_state.latitude_deg / 90.0)
        obs.append(self.satellite_state.longitude_deg / 180.0)

        # Battery and storage
        obs.append(self.satellite_state.battery_soc)
        storage_util = self.satellite_state.data_storage_gb / \
                      self.satellite_state.storage_capacity_gb
        obs.append(storage_util)

        # Time in episode
        obs.append(self.current_time_s / self.episode_duration_s)

        # Task queue features
        pending_tasks = self.task_queue.get_pending_tasks()
        for i in range(self.max_tasks):
            if i < len(pending_tasks):
                task = pending_tasks[i]
                obs.append(task.target_latitude_deg / 90.0)
                obs.append(task.target_longitude_deg / 180.0)
                obs.append(task.priority)
                time_to_deadline = max(0, task.deadline_s - self.current_time_s)
                time_available = max(1, task.deadline_s - task.arrival_time_s)
                deadline_urgency = 1.0 - (time_to_deadline / time_available)
                obs.append(np.clip(deadline_urgency, 0, 1))
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary"""
        return {
            "time_s": self.current_time_s,
            "battery_soc": self.satellite_state.battery_soc,
            "storage_gb": self.satellite_state.data_storage_gb,
            "tasks_completed": self.task_queue.count_completed(),
            "tasks_pending": self.task_queue.count_pending(),
            "episode_progress": self.current_time_s / self.episode_duration_s,
        }

    def render(self):
        """Render environment (optional)"""
        pass

    def close(self):
        """Close environment"""
        pass
