from collections import deque
from enum import IntEnum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from rich.console import Console

Position: TypeAlias = tuple[int, int]
State: TypeAlias = NDArray[np.int8]
StepResult: TypeAlias = tuple[State, float, bool, dict[str, int]]


class Direction(IntEnum):
    """Snake movement directions."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Action(IntEnum):
    """Snake actions relative to current direction."""

    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2


class SnakeEnv:
    """Snake environment with gym-style interface for Q-learning."""

    def __init__(
        self, grid_size: int = 10, max_steps: int = 1000, seed: int | None = None
    ) -> None:
        """
        Initialize Snake environment.

        Args:
            grid_size: Size of the square grid
            max_steps: Maximum steps per episode to prevent infinite loops
            seed: Random seed for reproducibility
        """
        self.grid_size: int = grid_size
        self.max_steps: int = max_steps
        self.rng: np.random.RandomState = np.random.RandomState(seed)

        # Game state
        self.snake: deque[Position]
        self.direction: Direction
        self.food: Position
        self.steps: int = 0
        self.score: int = 0
        self._initialized: bool = False

        # Action mapping: relative to current direction
        # 0 = straight, 1 = left turn, 2 = right turn
        self.action_space: int = 3
        self.state_size: int = grid_size * grid_size  # Grid state size

    def reset(self) -> State:
        """Reset the environment to initial state."""
        # Initialize snake in the middle, length 3
        middle = self.grid_size // 2
        self.snake = deque(
            [(middle, middle), (middle, middle + 1), (middle, middle + 2)]
        )
        self.direction = Direction.LEFT
        self.steps = 0
        self.score = 0
        self._initialized = True

        # Place food
        self._place_food()

        return self._get_state()

    def _ensure_initialized(self) -> None:
        """Ensure environment has been reset before use."""
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

    def _place_food(self) -> None:
        """Place food at random empty position."""
        # Check if board is full (win condition)
        if len(self.snake) >= self.grid_size * self.grid_size:
            return  # No space for food - game won!

        while True:
            food = (
                self.rng.randint(0, self.grid_size),
                self.rng.randint(0, self.grid_size),
            )
            if food not in self.snake:
                self.food = food
                break

    def step(self, action: Action) -> StepResult:
        """
        Execute one step in the environment.

        Args:
            action: Action.STRAIGHT, Action.LEFT, or Action.RIGHT (relative to current direction)

        Returns:
            state: New state after action
            reward: Reward received
            done: Whether episode is finished
            info: Additional information
        """
        self._ensure_initialized()
        self.steps += 1

        # Store previous distance to food for reward shaping
        head = self.snake[0]
        prev_distance = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

        # Convert relative action to new direction
        self.direction = self._get_new_direction(action)

        # Calculate new head position
        delta = self._get_direction_delta(self.direction)
        new_head = (head[0] + delta[0], head[1] + delta[1])

        # Check collisions
        done = False
        reward = 0.0  # Base reward

        # Wall collision
        if (
            new_head[0] < 0
            or new_head[0] >= self.grid_size
            or new_head[1] < 0
            or new_head[1] >= self.grid_size
        ):
            done = True
            reward = -10
            return self._get_state(), reward, done, {"score": self.score}

        # Self collision (exclude tail position unless eating food, since tail will move)
        collision_check = list(self.snake)[:-1] if new_head != self.food else self.snake
        if new_head in collision_check:
            done = True
            reward = -10
            return self._get_state(), reward, done, {"score": self.score}

        # Move snake
        self.snake.appendleft(new_head)

        # Check if food eaten
        if new_head == self.food:
            reward = 10
            self.score += 1

            # Check for win condition before placing new food
            if len(self.snake) >= self.grid_size * self.grid_size:
                done = True  # Won the game!
            else:
                self._place_food()
        else:
            # Remove tail if no food eaten
            self.snake.pop()

            # Reward shaping: encourage moving toward food
            new_distance = abs(new_head[0] - self.food[0]) + abs(
                new_head[1] - self.food[1]
            )
            if new_distance < prev_distance:
                reward = 0.1  # Small reward for getting closer
            else:
                reward = -0.1  # Small penalty for getting farther

        # Check max steps
        if self.steps >= self.max_steps:
            done = True

        state = self._get_state()
        info = {"score": self.score}

        return state, reward, done, info

    def _get_new_direction(self, action: Action) -> Direction:
        """Convert relative action to absolute direction."""
        if action == Action.STRAIGHT:
            return self.direction
        elif action == Action.LEFT:
            return Direction((self.direction.value - 1) % 4)
        else:  # Right turn (action == Action.RIGHT)
            return Direction((self.direction.value + 1) % 4)

    def _get_direction_delta(self, direction: Direction) -> Position:
        """Get (row, col) delta for a direction."""
        deltas = {
            Direction.UP: (-1, 0),
            Direction.RIGHT: (0, 1),
            Direction.DOWN: (1, 0),
            Direction.LEFT: (0, -1),
        }
        return deltas[direction]

    def _get_state(self) -> State:
        """
        Get current state as flattened grid.

        Grid encoding:
        - 0: Empty cell
        - 1: Snake body
        - 2: Snake head
        - 3: Food
        """
        # Initialize empty grid
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Place snake body
        for pos in list(self.snake)[1:]:
            grid[pos[0], pos[1]] = 1

        # Place snake head
        head = self.snake[0]
        grid[head[0], head[1]] = 2

        # Place food
        grid[self.food[0], self.food[1]] = 3

        # Flatten and return
        return grid.flatten()

    def _is_danger(self, relative_action: Action) -> int:
        """Check if there's danger in a direction (relative action)."""
        test_direction = self._get_new_direction(relative_action)
        head = self.snake[0]
        delta = self._get_direction_delta(test_direction)
        test_pos = (head[0] + delta[0], head[1] + delta[1])

        # Check wall collision
        if (
            test_pos[0] < 0
            or test_pos[0] >= self.grid_size
            or test_pos[1] < 0
            or test_pos[1] >= self.grid_size
        ):
            return 1

        # Check self collision
        collision_check = list(self.snake)[:-1] if test_pos != self.food else self.snake
        if test_pos in collision_check:
            return 1

        return 0

    def get_features(self) -> NDArray[np.int8]:
        """
        Get 11-dimensional feature vector for Q-learning.

        Returns:
            11-dimensional feature vector with:
            - Food direction (4 binary): up, down, left, right
            - Danger detection (3 binary): straight, left, right
            - Current direction (4 one-hot): up, right, down, left
        """
        self._ensure_initialized()

        head = self.snake[0]

        # Food direction (relative to head)
        food_up = int(self.food[0] < head[0])
        food_down = int(self.food[0] > head[0])
        food_left = int(self.food[1] < head[1])
        food_right = int(self.food[1] > head[1])

        # Danger detection
        danger_straight = self._is_danger(Action.STRAIGHT)
        danger_left = self._is_danger(Action.LEFT)
        danger_right = self._is_danger(Action.RIGHT)

        # Current direction (one-hot)
        dir_up = int(self.direction == Direction.UP)
        dir_right = int(self.direction == Direction.RIGHT)
        dir_down = int(self.direction == Direction.DOWN)
        dir_left = int(self.direction == Direction.LEFT)

        return np.array(
            [
                food_up,
                food_down,
                food_left,
                food_right,
                danger_straight,
                danger_left,
                danger_right,
                dir_up,
                dir_right,
                dir_down,
                dir_left,
            ],
            dtype=np.int8,
        )

    def render(self) -> None:
        """Render the environment with colored output using rich."""
        self._ensure_initialized()
        console = Console()
        console.clear()

        # Build grid as string
        lines = []

        # Top border
        border = "┌" + "─" * (self.grid_size * 2) + "┐"
        lines.append(border)

        for row_idx in range(self.grid_size):
            row = "│"
            for col_idx in range(self.grid_size):
                pos = (row_idx, col_idx)

                if pos == self.snake[0]:
                    # Snake head - bright green
                    row += "[bright_green]██[/]"
                elif pos in self.snake:
                    # Snake body - green
                    row += "[green]██[/]"
                elif pos == self.food:
                    # Food - red
                    row += "[red]██[/]"
                else:
                    # Empty space
                    row += "  "

            row += "│"
            lines.append(row)

        # Bottom border
        border = "└" + "─" * (self.grid_size * 2) + "┘"
        lines.append(border)

        console.print()
        for line in lines:
            console.print(line)
        console.print(f"Score: {self.score} | Steps: {self.steps}")
