"""
A* pathfinding agent for Snake game.
"""

from heapq import heappop, heappush
from typing import TYPE_CHECKING

from agent.base_agent import BaseAgent

if TYPE_CHECKING:
    from environment.snake_env import SnakeEnv


class AStarAgent(BaseAgent):
    """Agent that uses A* pathfinding to find the shortest path to food."""

    def __init__(self, env: "SnakeEnv"):
        """
        Initialize A* agent.

        Args:
            env: Environment instance to interact with
        """
        self.env = env
        self.grid_size = env.grid_size
        self.action_space = env.action_space

    def _manhattan_distance(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two positions.

        Args:
            pos1: First position (row, col)
            pos2: Second position (row, col)

        Returns:
            Manhattan distance
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_neighbors(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Get valid neighboring positions.

        Args:
            pos: Current position (row, col)

        Returns:
            List of valid neighboring positions
        """
        neighbors = []
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # UP, RIGHT, DOWN, LEFT

        for dr, dc in directions:
            new_pos = (pos[0] + dr, pos[1] + dc)
            # Check if within grid bounds
            if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                neighbors.append(new_pos)

        return neighbors

    def _astar(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> list[tuple[int, int]] | None:
        """
        Find shortest path from start to goal using A* pathfinding.

        Treats current snake body as obstacles, excluding the tail since it
        will move away when the snake moves forward (unless food is eaten).
        """
        counter = 0
        open_set = [(0, counter, start)]
        counter += 1

        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start: 0}
        closed_set: set[tuple[int, int]] = set()

        # Treat snake body as obstacles, but exclude tail (it will move)
        snake_set = set(self.env.snake)
        if len(self.env.snake) > 1:
            snake_set.discard(self.env.snake[-1])

        while open_set:
            _, _, current = heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in self._get_neighbors(current):
                # Skip if already visited or blocked by snake
                if neighbor in closed_set or neighbor in snake_set:
                    continue

                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f = tentative_g_score + self._manhattan_distance(neighbor, goal)
                    heappush(open_set, (f, counter, neighbor))
                    counter += 1

        return None

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
        diff = (target_dir - current_dir) % 4

        if diff == 0:
            return 0  # STRAIGHT
        elif diff == 1:
            return 2  # RIGHT turn (clockwise)
        elif diff == 3:
            return 1  # LEFT turn (counter-clockwise)
        else:
            # diff == 2 means 180° - can't turn around, choose right
            return 2

    def get_action(self, training: bool = True) -> int:  # noqa: ARG002
        """
        Select action using A* pathfinding to food.

        Args:
            training: Whether the agent is in training mode (unused)

        Returns:
            Selected action (0=STRAIGHT, 1=LEFT, 2=RIGHT)
        """
        # Get current snake head position and direction
        head = self.env.snake[0]
        current_dir = self.env.direction.value

        # Find path to food using A*
        path = self._astar(head, self.env.food)

        # If no path found or path is too short, go straight (fallback)
        if path is None or len(path) < 2:
            return 0  # STRAIGHT

        # Get next position in path
        next_pos = path[1]

        # Determine what direction we need to go
        target_dir = self._get_direction(head, next_pos)

        # Convert to relative action
        action = self._direction_to_action(current_dir, target_dir)

        return action
