"""
Observation Task Management Module

Handles creation, scheduling, and completion of observation tasks.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class TaskStatus(Enum):
    """Task status enumeration"""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ObservationTask:
    """Single observation task"""

    task_id: int
    target_latitude_deg: float
    target_longitude_deg: float
    priority: float  # [0, 1] importance weight
    arrival_time_s: float  # When task was added to queue
    deadline_s: float  # Latest completion time
    required_duration_s: float  # Observation duration needed
    data_rate_mbps: float = 10.0
    status: TaskStatus = TaskStatus.PENDING
    completion_time_s: Optional[float] = None


class TaskGenerator:
    """
    Generates observation tasks for the satellite

    Creates tasks with realistic parameters:
    - Geographic distribution
    - Priority distribution (some urgent, some routine)
    - Time windows and deadlines
    - Data requirements
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize task generator

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

    def generate_tasks(self, num_tasks: int, current_time_s: float,
                      episode_duration_s: float, max_priority: float = 1.0) -> List[ObservationTask]:
        """
        Generate a batch of observation tasks

        Args:
            num_tasks: Number of tasks to generate
            current_time_s: Current simulation time
            episode_duration_s: Total episode duration
            max_priority: Maximum priority value

        Returns:
            List of ObservationTask objects
        """
        tasks = []

        for i in range(num_tasks):
            # Random geographic distribution
            latitude = np.random.uniform(-60, 60)  # Exclude polar regions
            longitude = np.random.uniform(-180, 180)

            # Priority distribution: 30% high, 70% normal
            if np.random.rand() < 0.3:
                priority = np.random.uniform(0.7, max_priority)
            else:
                priority = np.random.uniform(0.3, 0.7)

            # Arrival time (spread throughout early episode)
            arrival_time = current_time_s + np.random.uniform(0, episode_duration_s * 0.3)

            # Deadline (reasonable but challenging)
            deadline = arrival_time + np.random.uniform(1800, 7200)  # 30 min to 2 hours

            # Observation duration (typically 5-10 minutes)
            duration = np.random.uniform(300, 600)

            task = ObservationTask(
                task_id=i,
                target_latitude_deg=latitude,
                target_longitude_deg=longitude,
                priority=priority,
                arrival_time_s=arrival_time,
                deadline_s=deadline,
                required_duration_s=duration,
                data_rate_mbps=np.random.uniform(8, 15),
            )

            tasks.append(task)

        return tasks

    @staticmethod
    def compute_task_value(task: ObservationTask, current_time_s: float) -> float:
        """
        Compute value/reward for completing a task

        Accounts for:
        - Priority (higher priority = higher value)
        - Urgency (deadlines approaching)
        - Already completed vs pending

        Args:
            task: The task to evaluate
            current_time_s: Current simulation time

        Returns:
            Task value/reward score
        """
        if task.status == TaskStatus.COMPLETED:
            return 0.0

        # Base reward from priority
        reward = task.priority

        # Bonus for completing high-priority tasks
        if task.priority > 0.7:
            reward *= 1.5

        # Time urgency (increases as deadline approaches)
        time_remaining = max(0, task.deadline_s - current_time_s)
        time_available = max(1, task.deadline_s - task.arrival_time_s)

        if time_remaining < (time_available * 0.2):
            # Critical deadline
            reward *= 2.0
        elif time_remaining < (time_available * 0.5):
            # Approaching deadline
            reward *= 1.3

        return reward


class TaskQueue:
    """Manages the queue of pending observation tasks"""

    def __init__(self):
        """Initialize task queue"""
        self.tasks: List[ObservationTask] = []
        self.completed_tasks: List[ObservationTask] = []

    def add_tasks(self, tasks: List[ObservationTask]) -> None:
        """Add tasks to queue"""
        self.tasks.extend(tasks)
        self.tasks.sort(key=lambda t: t.priority, reverse=True)

    def get_pending_tasks(self) -> List[ObservationTask]:
        """Get all pending tasks"""
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def complete_task(self, task_id: int, completion_time_s: float) -> bool:
        """
        Mark task as completed

        Args:
            task_id: ID of task to complete
            completion_time_s: Time of completion

        Returns:
            True if task was successfully completed, False otherwise
        """
        for task in self.tasks:
            if task.task_id == task_id:
                task.status = TaskStatus.COMPLETED
                task.completion_time_s = completion_time_s
                self.completed_tasks.append(task)
                return True

        return False

    def get_task_by_id(self, task_id: int) -> Optional[ObservationTask]:
        """Get task by ID"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task

        return None

    def count_pending(self) -> int:
        """Count pending tasks"""
        return len(self.get_pending_tasks())

    def count_completed(self) -> int:
        """Count completed tasks"""
        return len(self.completed_tasks)

    def compute_total_reward(self, current_time_s: float) -> float:
        """
        Compute total reward from all completed tasks

        Args:
            current_time_s: Current simulation time

        Returns:
            Sum of rewards from completed tasks
        """
        total_reward = 0.0

        for task in self.completed_tasks:
            reward = TaskGenerator.compute_task_value(task, current_time_s)
            total_reward += reward

        return total_reward
