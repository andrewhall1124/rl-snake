"""
Random agent implementation for baseline comparison.
"""

import numpy as np
from numpy.typing import NDArray

from agent.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that selects actions uniformly at random."""

    def __init__(self, action_space: int = 3):
        """
        Initialize random agent.

        Args:
            action_space: Number of possible actions
        """
        self.action_space = action_space

    def get_action(self, training: bool = True) -> int:
        """
        Select a random action.

        Args:
            state: Current state observation (unused)
            training: Whether the agent is in training mode (unused)

        Returns:
            Randomly selected action
        """
        return np.random.randint(0, self.action_space)
