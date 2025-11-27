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
        n = self.grid_size
        path = []

        # Start at (0, 0)
        path.append((0, 0))

        # Zigzag through columns 1 to n-2
        for col in range(1, n - 1):
            if col % 2 == 1:
                # Odd columns: go down from row 0 to n-2
                for row in range(n - 1):
                    path.append((row, col))
            else:
                # Even columns: go up from row n-2 to 0
                for row in range(n - 2, -1, -1):
                    path.append((row, col))

        # Last full column (n-1)
        for row in range(n):
            path.append((row, n - 1))

        # Traverse bottom edge (right to left) along row n-1
        for col in range(n - 1, -1, -1):
            path.append((n - 1, col))

        # Traverse left edge (bottom to top) along column 0
        for row in range(n - 2, 0, -1):
            path.append((row, 0))

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
