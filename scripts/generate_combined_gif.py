"""Generate a combined GIF showing multiple agents playing Snake simultaneously."""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from environment.snake_env import SnakeEnv


def render_frame_with_label(env: SnakeEnv, agent_name: str, score: int) -> np.ndarray:
    """
    Render the current environment state as a numpy array with agent name and score.

    Args:
        env: Snake environment instance
        agent_name: Name of the agent
        score: Current score

    Returns:
        RGB image array with label below the grid
    """
    grid_size = env.grid_size
    cell_size = 40  # pixels per cell
    label_height = 50  # pixels for label area

    # Create RGB image (grid + label area)
    img_size = grid_size * cell_size
    total_height = img_size + label_height
    img = np.zeros((total_height, img_size, 3), dtype=np.uint8)  # Black background

    # Draw the game grid
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
        img[i * cell_size, :img_size] = [200, 200, 200]
        # Vertical lines
        img[:img_size, i * cell_size] = [200, 200, 200]

    # Convert to PIL Image to add text and border
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Draw a thick white border around the grid
    border_thickness = 3
    for i in range(border_thickness):
        draw.rectangle(
            [i, i, img_size - 1 - i, img_size - 1 - i],
            outline=(255, 255, 255),
            width=1,
        )

    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        font = ImageFont.load_default()

    # Draw agent name and score
    text = f"{agent_name}\nScore: {score}"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (img_size - text_width) // 2
    text_y = img_size + 5

    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

    return np.array(pil_img)


def _create_grid_layout(frames: list[np.ndarray], columns: int) -> np.ndarray:
    """
    Arrange frames in a grid layout with specified number of columns.

    Args:
        frames: List of frame arrays to arrange
        columns: Number of columns in the grid

    Returns:
        Combined frame array with grid layout
    """
    num_frames = len(frames)
    rows = (num_frames + columns - 1) // columns  # Ceiling division

    # Get dimensions from first frame
    frame_height, frame_width = frames[0].shape[:2]

    # Create empty canvas
    total_width = frame_width * columns
    total_height = frame_height * rows
    canvas = np.zeros((total_height, total_width, 3), dtype=np.uint8)

    # Place frames in grid
    for idx, frame in enumerate(frames):
        row = idx // columns
        col = idx % columns

        y_start = row * frame_height
        y_end = y_start + frame_height
        x_start = col * frame_width
        x_end = x_start + frame_width

        canvas[y_start:y_end, x_start:x_end] = frame

    return canvas


def generate_combined_gif(
    agents: list[tuple[str, object]],
    output_filename: str,
    num_episodes: int = 1,
    fps: int = 10,
    columns: int = 3,
) -> None:
    """
    Generate a GIF showing multiple agents playing Snake simultaneously.

    Args:
        agents: List of (agent_name, agent_instance) tuples
        output_filename: Filename for the GIF (will be saved in gifs/ directory)
        num_episodes: Number of episodes to include in the GIF
        fps: Frames per second for the GIF
        columns: Number of columns in the grid layout
    """
    output_path = f"gifs/{output_filename}"

    print("=" * 60)
    print("Generating Combined Agent GIF")
    print("=" * 60)
    print(f"Agents: {', '.join([name for name, _ in agents])}")
    print(f"Output: {output_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Layout: {columns} columns")
    print(f"FPS: {fps}")
    print("=" * 60)

    frames = []

    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")

        # Reset all agents' environments
        for agent_name, agent in agents:
            agent.env.reset()

        # Track which agents are still active
        active_agents = {agent_name: True for agent_name, _ in agents}
        episode_steps = 0

        # Capture initial frame
        agent_frames = []
        for agent_name, agent in agents:
            frame = render_frame_with_label(agent.env, agent_name, 0)
            agent_frames.append(frame)
        combined_frame = _create_grid_layout(agent_frames, columns)
        frames.append(combined_frame)

        # Run all agents simultaneously
        while any(active_agents.values()):
            episode_steps += 1
            agent_frames = []

            for agent_name, agent in agents:
                if active_agents[agent_name]:
                    # Get action in evaluation mode (no exploration)
                    action = agent.get_action(training=False)

                    # Take step
                    _, _, done, info = agent.env.step(action)

                    if done:
                        active_agents[agent_name] = False
                        print(f"  {agent_name} finished - Score: {info['score']}")

                # Render current state (even if done)
                score = agent.env.score
                frame = render_frame_with_label(agent.env, agent_name, score)
                agent_frames.append(frame)

            # Combine all agent frames in grid layout
            combined_frame = _create_grid_layout(agent_frames, columns)
            frames.append(combined_frame)

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
    print(f"Total Frames: {len(frames)}")
    print(f"Output File: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    from agent.cycle_agent import CycleAgent
    from agent.dqn_agent import DQNAgent
    from agent.q_learning_agent import QLearningAgent
    from agent.random_agent import RandomAgent
    from agent.sarsa_agent import SARSAAgent

    # Create environments and agents
    seed = 42
    max_steps = 200

    # Random Agent
    random_env = SnakeEnv(grid_size=10, max_steps=max_steps, seed=seed)
    random_agent = RandomAgent(action_space=3)
    random_agent.env = random_env  # Add env attribute for compatibility

    # Q-Learning Agent
    q_env = SnakeEnv(grid_size=10, max_steps=max_steps, seed=seed)
    q_agent = QLearningAgent(q_env, seed=seed)
    q_agent.load("models/q_table_final.pkl")

    # SARSA Agent
    sarsa_env = SnakeEnv(grid_size=10, max_steps=max_steps, seed=seed)
    sarsa_agent = SARSAAgent(sarsa_env, seed=seed)
    sarsa_agent.load("models/sarsa_final.pkl")

    # DQN Agent
    dqn_env = SnakeEnv(grid_size=10, max_steps=max_steps, seed=seed)
    dqn_agent = DQNAgent(dqn_env, seed=seed)
    dqn_agent.load("models/dqn_final.pt")

    # Cycle Agent
    cycle_env = SnakeEnv(grid_size=10, max_steps=max_steps, seed=seed)
    cycle_agent = CycleAgent(cycle_env)

    # List of agents to compare
    agents = [
        ("Random", random_agent),
        ("Cycle", cycle_agent),
        ("SARSA", sarsa_agent),
        ("Q-Learning", q_agent),
        ("DQN", dqn_agent),
    ]

    # Generate combined GIF
    generate_combined_gif(
        agents=agents,
        output_filename="combined_agents.gif",
        num_episodes=3,  # Show 3 episodes
        fps=10,  # 10 frames per second
        columns=3,  # 3 columns layout
    )
