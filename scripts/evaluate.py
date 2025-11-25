"""
Evaluation script for trained Snake agents (agent-agnostic).
"""

import time

import numpy as np

from agent import BaseAgent, DQNAgent, QLearningAgent
from config import config
from environment import SnakeEnv

EvalResults = dict[str, float]


def evaluate(
    agent: BaseAgent,
    env: SnakeEnv,
    num_episodes: int | None = None,
    render: bool | None = None,
) -> EvalResults:
    """
    Evaluate an agent on the environment.

    Args:
        agent: Instantiated agent to evaluate
        env: Environment instance to evaluate on
        num_episodes: Number of episodes to evaluate (default from config)
        render: Whether to render episodes (default from config)

    Returns:
        Dictionary containing evaluation metrics
    """
    if num_episodes is None:
        num_episodes = config.evaluation.num_episodes
    if render is None:
        render = config.evaluation.render

    print("=" * 50)
    print("Snake Agent Evaluation")
    print("=" * 50)

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
                time.sleep(0.1)  # Slow down for viewing
                env.render()

        # Store metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_scores.append(info["score"])

        if episode % 10 == 0 or render:
            print(
                f"Episode {episode}: Score = {info['score']}, "
                f"Reward = {total_reward:.2f}, Steps = {steps}"
            )

    # Print statistics
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Episodes: {num_episodes}")
    print(
        f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
    )
    print(
        f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}"
    )
    print(
        f"Average Score: {np.mean(episode_scores):.2f} ± {np.std(episode_scores):.2f}"
    )
    print(f"Max Score: {np.max(episode_scores):.0f}")
    print(f"Min Score: {np.min(episode_scores):.0f}")
    print("=" * 50)

    results: EvalResults = {
        "avg_reward": float(np.mean(episode_rewards)),
        "avg_length": float(np.mean(episode_lengths)),
        "avg_score": float(np.mean(episode_scores)),
        "max_score": float(np.max(episode_scores)),
        "min_score": float(np.min(episode_scores)),
    }
    return results


if __name__ == "__main__":
    # Initialize environment
    env = SnakeEnv(
        grid_size=config.environment.grid_size,
        max_steps=config.environment.max_steps_per_episode,
        seed=config.random_seed,
    )

    # # Random
    # agent = RandomAgent(action_space=env.action_space)

    # Q-Learning
    agent = QLearningAgent(
        action_size=env.action_space,
        epsilon=0.0,  # Pure exploitation during evaluation
        seed=config.random_seed,
    )
    agent.load("models/q_table_final.pkl")

    # # DQN
    # agent = DQNAgent(
    #     state_size=env.state_size,
    #     action_size=env.action_space,
    #     learning_rate=config.agent.learning_rate,
    #     discount_factor=config.agent.discount_factor,
    #     epsilon=config.agent.epsilon_start,
    #     epsilon_decay=config.agent.epsilon_decay,
    #     epsilon_min=config.agent.epsilon_min,
    #     seed=config.random_seed,
    # )
    # agent.load('models/dqn_final.pt')

    # Run evaluation
    evaluate(agent=agent, env=env)

    # Ask if user wants to see rendered episodes
    response = input("\nWould you like to watch the agent play? (y/n): ")
    if response.lower() == "y":
        num_episodes = int(input("How many episodes to watch? (default 5): ") or "5")
        evaluate(agent=agent, env=env, num_episodes=num_episodes, render=True)
