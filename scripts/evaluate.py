"""
Evaluation script for trained Snake agents (agent-agnostic).
"""

import os
import time

import numpy as np

import config
from environment import SnakeEnv


def _infer_agent_class(model_path):
    """
    Infer agent class from model path filename.

    Args:
        model_path: Path to the saved model

    Returns:
        Agent class
    """
    from agent import QLearningAgent

    # Simple inference based on filename patterns
    # Can be extended to support more agent types
    filename = os.path.basename(model_path).lower()

    if "q_table" in filename or "q_learning" in filename:
        return QLearningAgent

    # Default to Q-Learning agent
    print(
        f"Warning: Could not infer agent type from '{model_path}', defaulting to QLearningAgent"
    )
    return QLearningAgent


def evaluate(
    agent_class=None,
    model_path="models/q_table_final.pkl",
    agent_kwargs=None,
    num_episodes=None,
    render=None,
):
    """
    Evaluate trained agent (works with any agent class).

    Args:
        agent_class: Agent class to use (if None, will infer from model_path)
        model_path: Path to saved model
        agent_kwargs: Dict of kwargs to pass to agent constructor (optional)
        num_episodes: Number of episodes to evaluate (default from config)
        render: Whether to render episodes (default from config)
    """
    if num_episodes is None:
        num_episodes = config.EVAL_EPISODES
    if render is None:
        render = config.RENDER_EVAL

    print("=" * 50)
    print("Snake Agent Evaluation")
    print("=" * 50)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first using train.py")
        return

    # Initialize environment
    env = SnakeEnv(
        grid_size=config.GRID_SIZE,
        max_steps=config.MAX_STEPS_PER_EPISODE,
        seed=config.RANDOM_SEED,
    )

    # Infer agent class from model path if not provided
    if agent_class is None:
        agent_class = _infer_agent_class(model_path)
        print(f"Inferred agent class: {agent_class.__name__}")

    # Initialize agent with default or provided kwargs
    if agent_kwargs is None:
        agent_kwargs = {}

    # Add common parameters if not already specified
    if "action_size" not in agent_kwargs:
        agent_kwargs["action_size"] = env.action_space
    if "epsilon" not in agent_kwargs:
        agent_kwargs["epsilon"] = 0.0  # Pure exploitation during evaluation
    if "seed" not in agent_kwargs:
        agent_kwargs["seed"] = config.RANDOM_SEED

    agent = agent_class(**agent_kwargs)

    # Load trained model
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

    return {
        "avg_reward": np.mean(episode_rewards),
        "avg_length": np.mean(episode_lengths),
        "avg_score": np.mean(episode_scores),
        "max_score": np.max(episode_scores),
        "min_score": np.min(episode_scores),
    }


def evaluate_with_render(
    agent_class=None,
    model_path="models/q_table_final.pkl",
    agent_kwargs=None,
    num_episodes=5,
):
    """Evaluate and render a few episodes."""
    print("\nEvaluating with visualization...")
    evaluate(
        agent_class=agent_class,
        model_path=model_path,
        agent_kwargs=agent_kwargs,
        num_episodes=num_episodes,
        render=True,
    )


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    model_path = "models/q_table_final.pkl"
    agent_class = None  # Will be inferred from model path
    agent_kwargs = None

    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    # Optional: Explicitly specify agent class via command line
    # Example: python evaluate.py models/dqn_final.pkl --agent DQNAgent
    if len(sys.argv) > 3 and sys.argv[2] == "--agent":
        agent_name = sys.argv[3]
        # Dynamically import agent class
        import agent

        agent_class = getattr(agent, agent_name)

    # Run evaluation
    evaluate(agent_class=agent_class, model_path=model_path, agent_kwargs=agent_kwargs)

    # Ask if user wants to see rendered episodes
    response = input("\nWould you like to watch the agent play? (y/n): ")
    if response.lower() == "y":
        num_episodes = int(input("How many episodes to watch? (default 5): ") or "5")
        evaluate_with_render(
            agent_class=agent_class,
            model_path=model_path,
            agent_kwargs=agent_kwargs,
            num_episodes=num_episodes,
        )
