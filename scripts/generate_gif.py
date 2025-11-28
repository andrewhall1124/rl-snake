"""Generate GIFs of trained agents playing Snake."""

import os

import numpy as np
from PIL import Image

from environment.snake_env import SnakeEnv


def render_frame(env: SnakeEnv) -> np.ndarray:
    """
    Render the current environment state as a numpy array (RGB image).

    Args:
        env: Snake environment instance

    Returns:
        RGB image array of shape (height, width, 3)
    """
    grid_size = env.grid_size
    cell_size = 40  # pixels per cell

    # Create RGB image
    img_size = grid_size * cell_size
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)  # Black background

    for row in range(grid_size):
        for col in range(grid_size):
            pos = (row, col)

            # Calculate pixel coordinates
            y1 = row * cell_size
            y2 = (row + 1) * cell_size
            x1 = col * cell_size
            x2 = (col + 1) * cell_size

            if pos == env.snake[0]:
                # Snake head - bright green
                img[y1:y2, x1:x2] = [0, 200, 0]
            elif pos in env.snake:
                # Snake body - green
                img[y1:y2, x1:x2] = [0, 150, 0]
            elif pos == env.food:
                # Food - red
                img[y1:y2, x1:x2] = [255, 0, 0]

    # Draw grid lines
    for i in range(grid_size):
        # Horizontal lines
        img[i * cell_size, :] = [200, 200, 200]
        # Vertical lines
        img[:, i * cell_size] = [200, 200, 200]

    return img


def generate_agent_gif(
    agent,
    output_filename: str,
    num_episodes: int = 1,
    fps: int = 10,
) -> None:
    """
    Generate a GIF of an agent playing Snake in evaluation mode.

    Args:
        agent: Agent instance (must have get_action method and env attribute)
        output_filename: Filename for the GIF (will be saved in gifs/ directory)
        num_episodes: Number of episodes to include in the GIF
        fps: Frames per second for the GIF
    """
    output_path = f"gifs/{output_filename}"

    print("=" * 60)
    print("Generating Agent GIF")
    print("=" * 60)
    print(f"Agent Type: {type(agent).__name__}")
    print(f"Output: {output_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Grid Size: {agent.env.grid_size}x{agent.env.grid_size}")
    print(f"FPS: {fps}")
    print("=" * 60)

    # Use the agent's environment
    env = agent.env

    # Collect frames
    frames = []
    total_score = 0

    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        env.reset()
        done = False
        episode_steps = 0

        # Capture initial frame
        frames.append(render_frame(env))

        while not done:
            # Get action in evaluation mode (no exploration)
            action = agent.get_action(training=False)

            # Take step
            _, _, done, info = env.step(action)
            episode_steps += 1

            # Capture frame
            frames.append(render_frame(env))

        score = info["score"]
        total_score += score
        print(f"  Score: {score}")
        print(f"  Steps: {episode_steps}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert frames to PIL Images and save as GIF
    print(f"\nSaving GIF with {len(frames)} frames...")
    pil_frames = [Image.fromarray(frame) for frame in frames]

    # Calculate duration per frame in milliseconds
    duration = int(1000 / fps)

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,  # Loop forever
    )

    print("=" * 60)
    print("GIF Generation Complete!")
    print("=" * 60)
    print(f"Total Episodes: {num_episodes}")
    print(f"Average Score: {total_score / num_episodes:.2f}")
    print(f"Total Frames: {len(frames)}")
    print(f"Output File: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    from agent.q_learning_agent import QLearningAgent

    # Create environment and agent
    env = SnakeEnv(grid_size=10, max_steps=1000, seed=42)
    agent = QLearningAgent(env, seed=42)

    # Load trained model
    agent.load("models/q_table_final.pkl")

    # Generate GIF
    generate_agent_gif(
        agent=agent,
        output_filename="q_learning_agent.gif",
        num_episodes=3,  # Show 3 episodes
        fps=10,  # 10 frames per second
    )
