"""
Training script for Snake Q-Learning agent.
"""

from agent import QLearningAgent, RandomAgent
from config import config
from environment import SnakeEnv


def main() -> None:
    """Main training entry point."""
    print(f"Grid Size: {config.environment.grid_size}x{config.environment.grid_size}")
    print(f"Max Steps per Episode: {config.environment.max_steps_per_episode}")
    print()

    # Initialize environment and agent
    env = SnakeEnv(
        grid_size=config.environment.grid_size,
        max_steps=config.environment.max_steps_per_episode,
        seed=config.random_seed,
    )

    agent = QLearningAgent(
        action_size=env.action_space,
        learning_rate=config.agent.learning_rate,
        discount_factor=config.agent.discount_factor,
        epsilon=config.agent.epsilon_start,
        epsilon_decay=config.agent.epsilon_decay,
        epsilon_min=config.agent.epsilon_min,
        seed=config.random_seed,
    )

    # Train the agent
    agent.train(
        env=env,
        num_episodes=config.training.num_episodes,
        save_interval=config.training.save_interval,
        model_dir=config.logging.model_dir,
    )


if __name__ == "__main__":
    main()
