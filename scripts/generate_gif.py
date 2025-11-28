"""Generate GIFs of trained agents playing Snake."""

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


def generate_agent_gif(
    agent,
    agent_name: str,
    output_filename: str,
    num_episodes: int = 1,
    fps: int = 10,
    frame_skip: int = 1,
) -> None:
    """
    Generate a GIF of an agent playing Snake in evaluation mode.

    Args:
        agent: Agent instance (must have get_action method and env attribute)
        agent_name: Name of the agent to display in the GIF
        output_filename: Filename for the GIF (will be saved in gifs/ directory)
        num_episodes: Number of episodes to include in the GIF
        fps: Frames per second for the GIF
        frame_skip: Only capture every Nth frame (higher = faster GIF, fewer frames)
    """
    output_path = f"gifs/{output_filename}"

    print("=" * 60)
    print("Generating Agent GIF")
    print("=" * 60)
    print(f"Agent Name: {agent_name}")
    print(f"Agent Type: {type(agent).__name__}")
    print(f"Output: {output_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Grid Size: {agent.env.grid_size}x{agent.env.grid_size}")
    print(f"FPS: {fps}")
    print(f"Frame Skip: {frame_skip}")
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
        frames.append(render_frame_with_label(env, agent_name, 0))

        while not done:
            # Get action in evaluation mode (no exploration)
            action = agent.get_action(training=False)

            # Take step
            _, _, done, info = env.step(action)
            episode_steps += 1

            # Capture frame (only every Nth frame based on frame_skip)
            if episode_steps % frame_skip == 0 or done:
                frames.append(render_frame_with_label(env, agent_name, env.score))

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
    from agent.dqn_agent import DQNAgent
    from agent.hamiltonian_cycle_agent import HamiltonianCycleAgent
    from agent.q_learning_agent import QLearningAgent
    from agent.random_agent import RandomAgent
    from agent.sarsa_agent import SARSAAgent

    # Configuration
    seed = 42
    max_steps = 1000
    num_episodes = 3
    fps = 10

    # List of agents to generate GIFs for
    agents = []

    # Random Agent
    random_env = SnakeEnv(grid_size=10, max_steps=max_steps, seed=seed)
    random_agent = RandomAgent(action_space=3)
    random_agent.env = random_env
    agents.append(("Random", random_agent, "random_agent.gif"))

    # Q-Learning Agent
    q_env = SnakeEnv(grid_size=10, max_steps=max_steps, seed=seed)
    q_agent = QLearningAgent(q_env, seed=seed)
    q_agent.load("models/q_table_final.pkl")
    agents.append(("Q-Learning", q_agent, "q_learning_agent.gif"))

    # SARSA Agent
    sarsa_env = SnakeEnv(grid_size=10, max_steps=max_steps, seed=seed)
    sarsa_agent = SARSAAgent(sarsa_env, seed=seed)
    sarsa_agent.load("models/sarsa_final.pkl")
    agents.append(("SARSA", sarsa_agent, "sarsa_agent.gif"))

    # DQN Agent
    dqn_env = SnakeEnv(grid_size=10, max_steps=max_steps, seed=seed)
    dqn_agent = DQNAgent(dqn_env, seed=seed)
    dqn_agent.load("models/dqn_final.pt")
    agents.append(("DQN", dqn_agent, "dqn_agent.gif"))

    # Hamiltonian Cycle Agent
    hamiltonian_cycle_env = SnakeEnv(grid_size=10, max_steps=max_steps, seed=seed)
    hamiltonian_cycle_agent = HamiltonianCycleAgent(hamiltonian_cycle_env)
    agents.append(
        ("Hamiltonian Cycle", hamiltonian_cycle_agent, "hamiltonian_cycle_agent.gif")
    )

    # Generate GIF for each agent
    for agent_name, agent, output_filename in agents:
        generate_agent_gif(
            agent=agent,
            agent_name=agent_name,
            output_filename=output_filename,
            num_episodes=num_episodes,
            fps=fps,
        )
        print("\n")

    # Hamiltonian Cycle Agent (full example)
    hamiltonian_cycle_env = SnakeEnv(grid_size=10, max_steps=3000, seed=seed)
    hamiltonian_cycle_agent = HamiltonianCycleAgent(hamiltonian_cycle_env)
    generate_agent_gif(
        agent=hamiltonian_cycle_agent,
        agent_name="Hamiltonian Cycle",
        output_filename="hamiltonian_cycle_agent_fast.gif",
        num_episodes=1,
        fps=100,
        frame_skip=5,  # Capture every 10th frame = 10x faster
    )
