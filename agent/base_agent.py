"""
Base agent abstract class for reinforcement learning agents.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


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
