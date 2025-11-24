"""
Base agent abstract class for reinforcement learning agents.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from environment.snake_env import SnakeEnv

TrainingMetrics = dict[str, list[float]]


class BaseAgent(ABC):
    """Abstract base class for reinforcement learning agents."""

    @abstractmethod
    def get_action(self, state: NDArray[np.int8], training: bool = True) -> int:
        """
        Select an action given the current state.

        Args:
            state: Current state observation
            training: Whether the agent is in training mode

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def update(
        self,
        state: NDArray[np.int8],
        action: int,
        reward: float,
        next_state: NDArray[np.int8],
        done: bool,
    ) -> None:
        """
        Update the agent's policy based on experience.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Save the agent's model to a file.

        Args:
            filepath: Path to save the model
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Load the agent's model from a file.

        Args:
            filepath: Path to load the model from
        """
        pass

    @abstractmethod
    def train(
        self,
        env: "SnakeEnv",
        num_episodes: int,
        save_interval: int = 100,
        model_dir: str = "models",
    ) -> TrainingMetrics:
        """
        Train the agent on the given environment.

        Args:
            env: Environment instance to train on
            num_episodes: Number of episodes to train
            save_interval: Interval for saving model checkpoints
            model_dir: Directory to save model checkpoints

        Returns:
            Training metrics
        """
        pass
