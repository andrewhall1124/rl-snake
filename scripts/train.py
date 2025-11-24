"""
Training script for Snake Q-Learning agent.
"""

from environment import SnakeEnv
from agent import QLearningAgent
import config


def main():
    """Main training entry point."""
    print(f"Grid Size: {config.GRID_SIZE}x{config.GRID_SIZE}")
    print(f"Max Steps per Episode: {config.MAX_STEPS_PER_EPISODE}")
    print()

    # Initialize environment and agent
    env = SnakeEnv(
        grid_size=config.GRID_SIZE,
        max_steps=config.MAX_STEPS_PER_EPISODE,
        seed=config.RANDOM_SEED
    )

    agent = QLearningAgent(
        action_size=env.action_space,
        learning_rate=config.LEARNING_RATE,
        discount_factor=config.DISCOUNT_FACTOR,
        epsilon=config.EPSILON_START,
        epsilon_decay=config.EPSILON_DECAY,
        epsilon_min=config.EPSILON_MIN,
        seed=config.RANDOM_SEED
    )

    # Train the agent
    agent.train(
        env=env,
        num_episodes=config.NUM_EPISODES,
        save_interval=config.SAVE_INTERVAL,
        model_dir=config.MODEL_DIR
    )


if __name__ == '__main__':
    main()
