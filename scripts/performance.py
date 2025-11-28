"""
Performance comparison script for RL algorithms.
Measures and compares Q-Learning, SARSA, and DQN on score and steps.
"""

import os

import numpy as np
import polars as pl
from great_tables import GT, loc, style

from agent import DQNAgent, QLearningAgent, SARSAAgent
from config import config
from environment import SnakeEnv


def evaluate_agent(agent, env: SnakeEnv, num_episodes: int = 100) -> dict[str, float]:
    """
    Evaluate a single agent and return performance metrics.

    Args:
        agent: Agent to evaluate
        env: Environment instance
        num_episodes: Number of evaluation episodes

    Returns:
        Dictionary with performance metrics
    """
    scores = []
    steps = []

    for episode in range(num_episodes):
        env.reset()
        total_steps = 0
        done = False

        while not done:
            action = agent.get_action(training=False)
            _, _, done, info = env.step(action)
            total_steps += 1

        scores.append(info["score"])
        steps.append(total_steps)

        if (episode + 1) % 25 == 0:
            print(f"  Progress: {episode + 1}/{num_episodes} episodes completed")

    return {
        "avg_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "max_score": float(np.max(scores)),
        "min_score": float(np.min(scores)),
        "avg_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "max_steps": float(np.max(steps)),
        "min_steps": float(np.min(steps)),
    }


def main():
    """Run performance comparison across all algorithms."""
    print("=" * 70)
    print("RL Snake Performance Comparison")
    print("=" * 70)

    # Initialize environment
    env = SnakeEnv(
        grid_size=config.environment.grid_size,
        max_steps=config.environment.max_steps_per_episode,
        seed=config.random_seed,
    )

    num_episodes = 100
    results = {}

    # Evaluate Q-Learning
    print("\n[1/3] Evaluating Q-Learning Agent...")
    q_agent = QLearningAgent(
        env=env,
        epsilon=0.0,
        seed=config.random_seed,
    )
    q_agent.load("models/q_table_final.pkl")
    results["Q-Learning"] = evaluate_agent(q_agent, env, num_episodes)

    # Evaluate SARSA
    print("\n[2/3] Evaluating SARSA Agent...")
    sarsa_agent = SARSAAgent(
        env=env,
        epsilon=0.0,
        seed=config.random_seed,
    )
    sarsa_agent.load("models/sarsa_final.pkl")
    results["SARSA"] = evaluate_agent(sarsa_agent, env, num_episodes)

    # Evaluate DQN
    print("\n[3/3] Evaluating DQN Agent...")
    dqn_agent = DQNAgent(env=env)
    dqn_agent.load("models/dqn_final.pt")
    results["DQN"] = evaluate_agent(dqn_agent, env, num_episodes)

    # Display results
    print("\n" + "=" * 70)
    print("Performance Results")
    print("=" * 70)
    print(f"\nEvaluation episodes per algorithm: {num_episodes}")
    print("\n" + "-" * 70)
    print(f"{'Algorithm':<15} {'Avg Score':<15} {'Avg Steps':<15} {'Max Score':<15}")
    print("-" * 70)

    for algorithm, metrics in results.items():
        print(
            f"{algorithm:<15} "
            f"{metrics['avg_score']:<15.2f} "
            f"{metrics['avg_steps']:<15.1f} "
            f"{metrics['max_score']:<15.0f}"
        )

    # Detailed statistics table
    print("\n" + "=" * 70)
    print("Detailed Statistics")
    print("=" * 70)

    # Create DataFrame for better display
    df_data = []
    for algorithm, metrics in results.items():
        df_data.append(
            {
                "Algorithm": algorithm,
                "Avg Score": f"{metrics['avg_score']:.2f} ± {metrics['std_score']:.2f}",
                "Max Score": f"{metrics['max_score']:.0f}",
                "Min Score": f"{metrics['min_score']:.0f}",
                "Avg Steps": f"{metrics['avg_steps']:.1f} ± {metrics['std_steps']:.1f}",
                "Max Steps": f"{metrics['max_steps']:.0f}",
                "Min Steps": f"{metrics['min_steps']:.0f}",
            }
        )

    df = pl.DataFrame(df_data)
    print("\n" + str(df))

    # Create and save great_tables visualization
    print("\n" + "=" * 70)
    print("Saving formatted table...")
    print("=" * 70)

    # Prepare data for great_tables with separate numeric columns
    gt_data = []
    for algorithm, metrics in results.items():
        gt_data.append(
            {
                "Algorithm": algorithm,
                "Avg Score": metrics["avg_score"],
                "Std Score": metrics["std_score"],
                "Max Score": metrics["max_score"],
                "Min Score": metrics["min_score"],
                "Avg Steps": metrics["avg_steps"],
                "Std Steps": metrics["std_steps"],
                "Max Steps": metrics["max_steps"],
                "Min Steps": metrics["min_steps"],
            }
        )

    gt_df = pl.DataFrame(gt_data)

    # Create great_tables table
    gt_table = (
        GT(gt_df)
        .tab_header(
            title="RL Snake Performance Comparison",
            subtitle=f"Results from {num_episodes} evaluation episodes per algorithm",
        )
        .tab_spanner(
            label="Score Metrics",
            columns=["Avg Score", "Std Score", "Max Score", "Min Score"],
        )
        .tab_spanner(
            label="Step Metrics",
            columns=["Avg Steps", "Std Steps", "Max Steps", "Min Steps"],
        )
        .fmt_number(
            columns=["Avg Score", "Std Score"],
            decimals=2,
        )
        .fmt_number(
            columns=["Max Score", "Min Score"],
            decimals=0,
        )
        .fmt_number(
            columns=["Avg Steps", "Std Steps"],
            decimals=1,
        )
        .fmt_number(
            columns=["Max Steps", "Min Steps"],
            decimals=0,
        )
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.body(
                columns="Avg Score",
                rows=pl.col("Avg Score").eq(pl.col("Avg Score").max()),
            ),
        )
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.body(
                columns="Max Score",
                rows=pl.col("Max Score").eq(pl.col("Max Score").max()),
            ),
        )
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.body(
                columns="Avg Steps",
                rows=pl.col("Avg Steps").eq(pl.col("Avg Steps").max()),
            ),
        )
        .cols_align(
            align="center",
            columns=[
                "Avg Score",
                "Std Score",
                "Max Score",
                "Min Score",
                "Avg Steps",
                "Std Steps",
                "Max Steps",
                "Min Steps",
            ],
        )
        .cols_align(align="left", columns="Algorithm")
    )

    # Save as HTML
    os.makedirs("results", exist_ok=True)
    output_file = "results/performance_results.png"
    gt_table.save(output_file, scale=3)
    print(f"Table saved to: {output_file}")

    # Determine best performers
    print("\n" + "=" * 70)
    print("Best Performers")
    print("=" * 70)

    best_avg_score = max(results.items(), key=lambda x: x[1]["avg_score"])
    best_max_score = max(results.items(), key=lambda x: x[1]["max_score"])
    best_avg_steps = max(results.items(), key=lambda x: x[1]["avg_steps"])

    print(
        f"Best Average Score: {best_avg_score[0]} ({best_avg_score[1]['avg_score']:.2f})"
    )
    print(
        f"Best Maximum Score: {best_max_score[0]} ({best_max_score[1]['max_score']:.0f})"
    )
    print(
        f"Most Steps (Longest Survival): {best_avg_steps[0]} ({best_avg_steps[1]['avg_steps']:.1f})"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
