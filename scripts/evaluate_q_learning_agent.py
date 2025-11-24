"""
Evaluation script for trained Snake Q-Learning agent.
"""

import os
import numpy as np
import time

from environment import SnakeEnv
from agent import QLearningAgent
import config


def evaluate(model_path='models/q_table_final.pkl', num_episodes=None, render=None):
    """
    Evaluate trained agent.

    Args:
        model_path: Path to saved Q-table
        num_episodes: Number of episodes to evaluate (default from config)
        render: Whether to render episodes (default from config)
    """
    if num_episodes is None:
        num_episodes = config.EVAL_EPISODES
    if render is None:
        render = config.RENDER_EVAL

    print("=" * 50)
    print("Snake Q-Learning Evaluation")
    print("=" * 50)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first using train.py")
        return

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
        epsilon=0.0,  # Pure exploitation during evaluation
        seed=config.RANDOM_SEED
    )

    # Load trained Q-table
    agent.load(model_path)

    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    episode_scores = []

    print(f"\nRunning {num_episodes} evaluation episodes...")
    print("=" * 50)

    # Evaluation loop
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        if render:
            print(f"\n--- Episode {episode} ---")
            env.render()

        while not done:
            # Select action (greedy, no exploration)
            action = agent.get_action(state, training=False)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            steps += 1
            state = next_state

            if render:
                time.sleep(0.2)  # Slow down for viewing
                env.render()

        # Store metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_scores.append(info['score'])

        if episode % 10 == 0 or render:
            print(f"Episode {episode}: Score = {info['score']}, "
                  f"Reward = {total_reward:.2f}, Steps = {steps}")

    # Print statistics
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average Score: {np.mean(episode_scores):.2f} ± {np.std(episode_scores):.2f}")
    print(f"Max Score: {np.max(episode_scores):.0f}")
    print(f"Min Score: {np.min(episode_scores):.0f}")
    print("=" * 50)

    return {
        'avg_reward': np.mean(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'avg_score': np.mean(episode_scores),
        'max_score': np.max(episode_scores),
        'min_score': np.min(episode_scores)
    }


def evaluate_with_render(model_path='models/q_table_final.pkl', num_episodes=5):
    """Evaluate and render a few episodes."""
    print("\nEvaluating with visualization...")
    evaluate(model_path=model_path, num_episodes=num_episodes, render=True)


if __name__ == '__main__':
    import sys

    # Parse command line arguments
    model_path = 'models/q_table_final.pkl'
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    # Run evaluation
    evaluate(model_path=model_path)

    # Ask if user wants to see rendered episodes
    response = input("\nWould you like to watch the agent play? (y/n): ")
    if response.lower() == 'y':
        num_episodes = int(input("How many episodes to watch? (default 5): ") or "5")
        evaluate_with_render(model_path=model_path, num_episodes=num_episodes)
