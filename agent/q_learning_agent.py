import os
import pickle
from collections import defaultdict, deque
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from agent.base_agent import BaseAgent

if TYPE_CHECKING:
    from environment.snake_env import SnakeEnv

QTable: TypeAlias = defaultdict[tuple[int, ...], NDArray[np.float64]]
TrainingMetrics: TypeAlias = dict[str, list[float]]
AgentStats: TypeAlias = dict[str, int | float]


class QLearningAgent(BaseAgent):
    """Q-learning agent with epsilon-greedy exploration."""

    def __init__(
        self,
        action_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """
        Initialize Q-learning agent.

        Args:
            action_size: Number of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            seed: Random seed for reproducibility
        """
        self.action_size: int = action_size
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min

        # Q-table: dictionary mapping state tuples to action values
        # Using defaultdict to initialize unseen states to zeros
        self.q_table: QTable = defaultdict(lambda: np.zeros(action_size))

        self.rng: np.random.RandomState = np.random.RandomState(seed)

    def get_action(self, state: NDArray[np.int8], training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state (numpy array)
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            action: Selected action
        """
        state_tuple = tuple(state)

        # Epsilon-greedy exploration
        if training and self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.action_size)
        else:
            # Exploit: choose best action
            q_values = self.q_table[state_tuple]
            return np.argmax(q_values)

    def update(
        self,
        state: NDArray[np.int8],
        action: int,
        reward: float,
        next_state: NDArray[np.int8],
        done: bool,
    ) -> None:
        """
        Update Q-table using Q-learning update rule.

        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        state_tuple = tuple(state)
        next_state_tuple = tuple(next_state)

        # Current Q-value
        current_q = self.q_table[state_tuple][action]

        # Calculate target Q-value
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state_tuple])
            target_q = reward + self.discount_factor * max_next_q

        # Update Q-value
        self.q_table[state_tuple][action] = current_q + self.learning_rate * (
            target_q - current_q
        )

    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Save Q-table to file."""
        # Convert defaultdict to regular dict for pickling
        q_table_dict = dict(self.q_table)
        save_data = {
            "q_table": q_table_dict,
            "epsilon": self.epsilon,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)

    def load(self, filepath: str) -> None:
        """Load Q-table from file."""
        with open(filepath, "rb") as f:
            save_data = pickle.load(f)

        # Restore Q-table as defaultdict
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))
        self.q_table.update(save_data["q_table"])

        self.epsilon = save_data["epsilon"]
        self.action_size = save_data["action_size"]
        self.learning_rate = save_data["learning_rate"]
        self.discount_factor = save_data["discount_factor"]
        self.epsilon_min = save_data["epsilon_min"]
        self.epsilon_decay = save_data["epsilon_decay"]

        print(f"Q-table loaded from {filepath}")
        print(f"Number of states in Q-table: {len(self.q_table)}")

    def get_stats(self) -> AgentStats:
        """Get statistics about the Q-table."""
        return {"num_states": len(self.q_table), "epsilon": self.epsilon}

    def train(
        self,
        env: "SnakeEnv",
        num_episodes: int,
        save_interval: int = 100,
        model_dir: str = "models",
    ) -> TrainingMetrics:
        """
        Train the agent on the given environment.

        Args:
            env: Environment instance to train on
            num_episodes: Number of episodes to train
            save_interval: Interval for saving Q-table checkpoints
            model_dir: Directory to save model checkpoints

        Returns:
            dict: Training metrics including episode_rewards and episode_scores
        """
        print("=" * 50)
        print("Snake Q-Learning Training")
        print("=" * 50)
        print(f"Episodes: {num_episodes}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Discount Factor: {self.discount_factor}")
        print(
            f"Epsilon: {self.epsilon} -> {self.epsilon_min} (decay: {self.epsilon_decay})"
        )
        print("=" * 50)

        # Create directories
        os.makedirs(model_dir, exist_ok=True)

        # Metrics tracking
        episode_rewards = []
        episode_scores = []
        reward_window = deque(maxlen=100)
        score_window = deque(maxlen=100)

        # Training loop
        for episode in tqdm(range(1, num_episodes + 1), desc="Training"):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done:
                # Select and perform action
                action = self.get_action(state, training=True)
                next_state, reward, done, info = env.step(action)

                # Update Q-table
                self.update(state, action, reward, next_state, done)

                # Update counters
                total_reward += reward
                steps += 1
                state = next_state

            # Decay epsilon after episode
            self.decay_epsilon()

            # Track metrics
            episode_rewards.append(total_reward)
            episode_scores.append(info["score"])
            reward_window.append(total_reward)
            score_window.append(info["score"])

            # Save Q-table periodically
            if episode % save_interval == 0:
                save_path = os.path.join(model_dir, f"q_table_episode_{episode}.pkl")
                self.save(save_path)

        # Final save
        final_path = os.path.join(model_dir, "q_table_final.pkl")
        self.save(final_path)

        # Print summary
        print("\n" + "=" * 50)
        print("Training Complete!")
        print("=" * 50)
        print(f"Total Episodes: {len(episode_rewards)}")
        print(f"Average Reward: {np.mean(episode_rewards):.2f}")
        print(f"Average Score: {np.mean(episode_scores):.2f}")
        print(f"Max Score: {np.max(episode_scores):.0f}")
        print(f"Final Epsilon: {self.epsilon:.3f}")
        print(f"Q-table States: {len(self.q_table)}")
        print("=" * 50)

        return {"episode_rewards": episode_rewards, "episode_scores": episode_scores}
