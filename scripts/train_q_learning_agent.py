"""
Training script for Snake Q-Learning agent.
"""

import os
import numpy as np
from collections import deque
from tqdm import tqdm

from environment import SnakeEnv
from agent import QLearningAgent
import config


def train():
    """Main training loop."""
    print("=" * 50)
    print("Snake Q-Learning Training")
    print("=" * 50)
    print(f"Episodes: {config.NUM_EPISODES}")
    print(f"Grid Size: {config.GRID_SIZE}x{config.GRID_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Discount Factor: {config.DISCOUNT_FACTOR}")
    print(f"Epsilon: {config.EPSILON_START} -> {config.EPSILON_MIN} (decay: {config.EPSILON_DECAY})")
    print("=" * 50)

    # Create directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)

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

    # Metrics tracking
    episode_rewards = []
    episode_scores = []
    reward_window = deque(maxlen=100)
    score_window = deque(maxlen=100)

    # Training loop
    for episode in tqdm(range(1, config.NUM_EPISODES + 1), desc="Training"):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # Select and perform action
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            # Update Q-table
            agent.update(state, action, reward, next_state, done)

            # Update counters
            total_reward += reward
            steps += 1
            state = next_state

        # Decay epsilon after episode
        agent.decay_epsilon()

        # Track metrics
        episode_rewards.append(total_reward)
        episode_scores.append(info['score'])
        reward_window.append(total_reward)
        score_window.append(info['score'])

        # Save Q-table periodically
        if episode % config.SAVE_INTERVAL == 0:
            save_path = os.path.join(config.MODEL_DIR, f'q_table_episode_{episode}.pkl')
            agent.save(save_path)

    # Final save
    final_path = os.path.join(config.MODEL_DIR, 'q_table_final.pkl')
    agent.save(final_path)

    # Print summary
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Total Episodes: {len(episode_rewards)}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Average Score: {np.mean(episode_scores):.2f}")
    print(f"Max Score: {np.max(episode_scores):.0f}")
    print(f"Final Epsilon: {agent.epsilon:.3f}")
    print(f"Q-table States: {len(agent.q_table)}")
    print("=" * 50)


if __name__ == '__main__':
    train()
