"""
Cycle agent that systematically visits every square in the grid.
"""

from typing import TYPE_CHECKING

from agent.base_agent import BaseAgent

if TYPE_CHECKING:
    from environment.snake_env import SnakeEnv


class CycleAgent(BaseAgent):
    """Agent that follows a Hamiltonian cycle through all grid squares."""

    def __init__(self, env: "SnakeEnv"):
        """
        Initialize cycle agent.

        Args:
            env: Environment instance to interact with
        """
        self.env = env
        self.grid_size = env.grid_size
        self.action_space = env.action_space
        self.cycle_path = self._generate_hamiltonian_cycle()
        self.cycle_index = {pos: i for i, pos in enumerate(self.cycle_path)}

    def _generate_hamiltonian_cycle(self) -> list[tuple[int, int]]:
        """
        Generate a Hamiltonian cycle visiting every cell exactly once.

        Pattern creates a boustrophedon (zigzag) path with perimeter return:
        - Start at (0,0)
        - Zigzag through interior columns (0 to n-2)
        - After reaching top-right area, traverse the perimeter back to start:
          * Right edge down
          * Bottom edge left
          * Left edge up (back to (0,0))

        Returns:
            List of (row, col) positions forming a complete cycle
        """
        path = [
            # Right
            (0, 0),
            # Down 1
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            # Right
            (8, 1),
            # Up 2
            (8, 2),
            (7, 2),
            (6, 2),
            (5, 2),
            (4, 2),
            (3, 2),
            (2, 2),
            (1, 2),
            # Right
            (0, 2),
            # Down 3
            (0, 3),
            (1, 3),
            (2, 3),
            (3, 3),
            (4, 3),
            (5, 3),
            (6, 3),
            (7, 3),
            # Right
            (8, 3),
            # Up 4
            (8, 4),
            (7, 4),
            (6, 4),
            (5, 4),
            (4, 4),
            (3, 4),
            (2, 4),
            (1, 4),
            # Right
            (0, 4),
            # Down 5
            (0, 5),
            (1, 5),
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 5),
            (6, 5),
            (7, 5),
            # Right
            (8, 5),
            # Up 6
            (8, 6),
            (7, 6),
            (6, 6),
            (5, 6),
            (4, 6),
            (3, 6),
            (2, 6),
            (1, 6),
            # Right
            (0, 6),
            # Down 7
            (0, 7),
            (1, 7),
            (2, 7),
            (3, 7),
            (4, 7),
            (5, 7),
            (6, 7),
            (7, 7),
            # Right
            (8, 7),
            # Up 8
            (8, 8),
            (7, 8),
            (6, 8),
            (5, 8),
            (4, 8),
            (3, 8),
            (2, 8),
            (1, 8),
            # Right
            (0, 8),
            # Down 9
            (0, 9),
            (1, 9),
            (2, 9),
            (3, 9),
            (4, 9),
            (5, 9),
            (6, 9),
            (7, 9),
            (8, 9),
            # Left
            (9, 9),
            (9, 8),
            (9, 7),
            (9, 6),
            (9, 5),
            (9, 4),
            (9, 3),
            (9, 2),
            (9, 1),
            (9, 0),
            # Up 0
            (8, 0),
            (7, 0),
            (6, 0),
            (5, 0),
            (4, 0),
            (3, 0),
            (2, 0),
            (1, 0),
        ]

        return path

    def _get_direction(self, from_pos: tuple[int, int], to_pos: tuple[int, int]) -> int:
        """
        Get the absolute direction from one position to an adjacent position.

        Returns:
            Direction as int matching environment's Direction enum:
            UP=0, RIGHT=1, DOWN=2, LEFT=3
        """
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]

        if dr == -1:
            return 0  # UP
        elif dr == 1:
            return 2  # DOWN
        elif dc == -1:
            return 3  # LEFT
        else:  # dc == 1
            return 1  # RIGHT

    def _direction_to_action(self, current_dir: int, target_dir: int) -> int:
        """
        Convert from current direction to target direction into a relative action.

        Direction enum: UP=0, RIGHT=1, DOWN=2, LEFT=3 (clockwise)

        Args:
            current_dir: Current facing direction
            target_dir: Desired direction

        Returns:
            Action (0=STRAIGHT, 1=LEFT, 2=RIGHT)
        """
        # Both directions are already in clockwise order (0,1,2,3 = UP,RIGHT,DOWN,LEFT)
        diff = (target_dir - current_dir) % 4

        if diff == 0:
            return 0  # STRAIGHT
        elif diff == 1:
            return 2  # RIGHT turn (clockwise)
        elif diff == 3:
            return 1  # LEFT turn (counter-clockwise, same as -1 mod 4)
        else:
            # diff == 2 means 180° - shouldn't happen in valid cycle
            # but if it does, just turn right (will need another turn next step)
            return 2

    def get_action(self, training: bool = True) -> int:  # noqa: ARG002
        """
        Select action to follow the Hamiltonian cycle.

        Args:
            training: Whether the agent is in training mode (unused)

        Returns:
            Selected action (0=STRAIGHT, 1=LEFT, 2=RIGHT)
        """
        # Get current snake head position and direction
        head = self.env.snake[0]
        current_dir = self.env.direction.value  # Convert IntEnum to int

        # Find where we are in the cycle
        current_idx = self.cycle_index[head]
        next_idx = (current_idx + 1) % len(self.cycle_path)
        next_pos = self.cycle_path[next_idx]

        # Determine what direction we need to go
        target_dir = self._get_direction(head, next_pos)

        # Convert to relative action
        action = self._direction_to_action(current_dir, target_dir)

        return action
