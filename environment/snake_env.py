from collections import deque
from enum import IntEnum
from typing import TypeAlias

import imageio
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

Position: TypeAlias = tuple[int, int]
State: TypeAlias = NDArray[np.int8]
StepResult: TypeAlias = tuple[State, float, bool, dict[str, int]]


class Direction(IntEnum):
    """Snake movement directions."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


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
        self.snake: deque[Position] | None = None
        self.direction: Direction | None = None
        self.food: Position | None = None
        self.steps: int = 0
        self.score: int = 0

        # Action mapping: relative to current direction
        # 0 = straight, 1 = left turn, 2 = right turn
        self.action_space: int = 3
        self.state_size: int = 11  # Feature vector size

        self.frames = []  # store PIL images for GIF/MP4 export

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

        # Place food
        self._place_food()

        return self._get_state()

    def _place_food(self) -> None:
        """Place food at random empty position."""
        while True:
            food = (
                self.rng.randint(0, self.grid_size),
                self.rng.randint(0, self.grid_size),
            )
            if food not in self.snake:
                self.food = food
                break

    def step(self, action: int) -> StepResult:
        """
        Execute one step in the environment.

        Args:
            action: 0=straight, 1=left, 2=right (relative to current direction)

        Returns:
            state: New state after action
            reward: Reward received
            done: Whether episode is finished
            info: Additional information
        """
        self.steps += 1

        # Convert relative action to new direction
        self.direction = self._get_new_direction(action)

        # Calculate new head position
        head = self.snake[0]
        delta = self._get_direction_delta(self.direction)
        new_head = (head[0] + delta[0], head[1] + delta[1])

        # Check collisions
        done = False
        reward = -0.01  # Small penalty per step

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

        # Self collision
        if new_head in self.snake:
            done = True
            reward = -10
            return self._get_state(), reward, done, {"score": self.score}

        # Move snake
        self.snake.appendleft(new_head)

        # Check if food eaten
        if new_head == self.food:
            reward = 10
            self.score += 1
            self._place_food()
        else:
            # Remove tail if no food eaten
            self.snake.pop()

        # Check max steps
        if self.steps >= self.max_steps:
            done = True

        state = self._get_state()
        info = {"score": self.score}

        return state, reward, done, info

    def _get_new_direction(self, action: int) -> Direction:
        """Convert relative action to absolute direction."""
        if action == 0:  # Straight
            return self.direction
        elif action == 1:  # Left turn
            return Direction((self.direction.value - 1) % 4)
        else:  # Right turn (action == 2)
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
        Get current state as 11-dimensional feature vector.

        Features:
        - Food direction (4 binary): up, down, left, right
        - Danger detection (3 binary): straight, left, right
        - Current direction (4 one-hot): up, right, down, left
        """
        head = self.snake[0]

        # Food direction (relative to head)
        food_up = int(self.food[0] < head[0])
        food_down = int(self.food[0] > head[0])
        food_left = int(self.food[1] < head[1])
        food_right = int(self.food[1] > head[1])

        # Danger detection
        danger_straight = self._is_danger(0)
        danger_left = self._is_danger(1)
        danger_right = self._is_danger(2)

        # Current direction (one-hot)
        dir_up = int(self.direction == Direction.UP)
        dir_right = int(self.direction == Direction.RIGHT)
        dir_down = int(self.direction == Direction.DOWN)
        dir_left = int(self.direction == Direction.LEFT)

        state = np.array(
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

        return state

    def _is_danger(self, relative_action: int) -> int:
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
        if test_pos in self.snake:
            return 1

        return 0

    # def render(self, mode: str = "human") -> None:
    #     """Render the environment (text-based)."""
    #     if mode != "human":
    #         return

    #     grid = [[" " for _ in range(self.grid_size)] for _ in range(self.grid_size)]

    #     # Place snake
    #     for i, segment in enumerate(self.snake):
    #         if i == 0:
    #             grid[segment[0]][segment[1]] = "H"  # Head
    #         else:
    #             grid[segment[0]][segment[1]] = "o"  # Body

    #     # Place food
    #     grid[self.food[0]][self.food[1]] = "F"

    #     # Print grid
    #     print("\n" + "=" * (self.grid_size * 2 + 1))
    #     for row in grid:
    #         print("|" + "|".join(row) + "|")
    #     print("=" * (self.grid_size * 2 + 1))
    #     print(f"Score: {self.score} | Steps: {self.steps}")

    def render(self, mode: str = "human", pixel_size: int = 20) -> None:
        """
        Retro 8-bit video-game-style renderer.
        Produces a PIL image and stores each frame for MP4/GIF output.
        """

        # Create blank RGB image
        img_size = (self.grid_size * pixel_size, self.grid_size * pixel_size)
        img = Image.new("RGB", img_size, color=(10, 10, 10))  # dark background
        draw = ImageDraw.Draw(img)

        # Colors (retro palette)
        COLOR_BG = (10, 10, 10)
        COLOR_GRID = (40, 40, 40)
        COLOR_SNAKE = (0, 255, 0)  # bright green
        COLOR_SNAKE_HEAD = (0, 200, 0)
        COLOR_FOOD = (255, 0, 0)

        # Draw grid lines (pixel-art effect)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x0 = c * pixel_size
                y0 = r * pixel_size
                x1 = x0 + pixel_size
                y1 = y0 + pixel_size

                # base grid cell
                draw.rectangle([x0, y0, x1, y1], fill=COLOR_BG, outline=COLOR_GRID)

        # Draw snake
        for i, (r, c) in enumerate(self.snake):
            x0 = c * pixel_size
            y0 = r * pixel_size
            x1 = x0 + pixel_size
            y1 = y0 + pixel_size

            if i == 0:
                color = COLOR_SNAKE_HEAD
            else:
                color = COLOR_SNAKE

            draw.rectangle([x0, y0, x1, y1], fill=color)

        # Draw food
        fr, fc = self.food
        x0 = fc * pixel_size
        y0 = fr * pixel_size
        x1 = x0 + pixel_size
        y1 = y0 + pixel_size
        draw.rectangle([x0, y0, x1, y1], fill=COLOR_FOOD)

        # Save frame for video export
        self.frames.append(img)

        # If human: draw to screen (optional)
        if mode == "human":
            img.show()

    def save_video(self, filename: str, fps: int = 10) -> None:
        """
        Save captured frames as MP4 or GIF based on file extension.
        """
        # Convert PIL images to numpy arrays
        frames_np = [np.array(f) for f in self.frames]

        imageio.mimsave(filename, frames_np, fps=fps)
        print(f"Saved video to {filename}")
