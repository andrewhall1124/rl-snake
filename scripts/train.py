"""
Training script for Snake Q-Learning agent.
"""

from agent import DQNAgent, QLearningAgent, RandomAgent, SARSAAgent
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

    # # DQN
    # agent = DQNAgent(
    #     env=env,
    #     hidden_size=128,
    #     learning_rate=0.001,
    #     discount_factor=0.95,
    #     epsilon=1,
    #     epsilon_decay=0.995,
    #     epsilon_min=0.01,
    #     buffer_capacity=10000,
    #     batch_size=64,
    #     target_update=10,
    #     seed=42,
    # )

    # # Q-Learning
    # agent = QLearningAgent(
    #     env=env,
    #     learning_rate=config.agent.learning_rate,
    #     discount_factor=config.agent.discount_factor,
    #     epsilon=config.agent.epsilon_start,
    #     epsilon_decay=config.agent.epsilon_decay,
    #     epsilon_min=config.agent.epsilon_min,
    #     seed=config.random_seed,
    # )

    # SARSA
    agent = SARSAAgent(
        env=env,
        learning_rate=config.agent.learning_rate,
        discount_factor=config.agent.discount_factor,
        epsilon=config.agent.epsilon_start,
        epsilon_decay=config.agent.epsilon_decay,
        epsilon_min=config.agent.epsilon_min,
        seed=config.random_seed,
    )

    # Train the agent
    agent.train(
        num_episodes=config.training.num_episodes,
        save_interval=config.training.save_interval,
        model_dir=config.logging.model_dir,
    )


if __name__ == "__main__":
    main()
