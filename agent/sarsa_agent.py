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


class SARSAAgent(BaseAgent):
    """SARSA agent with epsilon-greedy exploration."""

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
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table: QTable = defaultdict(lambda: np.zeros(action_size))
        self.rng = np.random.RandomState(seed)

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
        next_action: int,
        done: bool,
    ) -> None:
        """
        SARSA update rule:

        Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
        """

        s = tuple(state)
        s_next = tuple(next_state)

        current_q = self.q_table[s][action]

        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_table[s_next][next_action]

        self.q_table[s][action] += self.learning_rate * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        q_dict = dict(self.q_table)
        save_data = {
            "q_table": q_dict,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "action_size": self.action_size,
        }
        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.q_table = defaultdict(lambda: np.zeros(self.action_size))
        self.q_table.update(data["q_table"])

        self.epsilon = data["epsilon"]
        self.learning_rate = data["learning_rate"]
        self.discount_factor = data["discount_factor"]
        self.epsilon_min = data["epsilon_min"]
        self.epsilon_decay = data["epsilon_decay"]
        self.action_size = data["action_size"]

    def get_stats(self) -> AgentStats:
        return {"num_states": len(self.q_table), "epsilon": self.epsilon}

    def train(
        self,
        env: "SnakeEnv",
        num_episodes: int,
        save_interval: int = 100,
        model_dir: str = "models",
    ) -> TrainingMetrics:
        print("=" * 50)
        print("Snake SARSA Training")
        print("=" * 50)

        os.makedirs(model_dir, exist_ok=True)

        episode_rewards = []
        episode_scores = []

        for episode in tqdm(range(1, num_episodes + 1), desc="Training"):
            state = env.reset()
            action = self.get_action(state, training=True)

            total_reward = 0
            done = False

            while not done:
                next_state, reward, done, info = env.step(action)

                # Select next action from SARSA policy
                next_action = self.get_action(next_state, training=True)

                # SARSA update
                self.update(state, action, reward, next_state, next_action, done)

                state = next_state
                action = next_action
                total_reward += reward

            self.decay_epsilon()

            episode_rewards.append(total_reward)
            episode_scores.append(info["score"])

            if episode % save_interval == 0:
                path = os.path.join(model_dir, f"sarsa_episode_{episode}.pkl")
                self.save(path)

        final = os.path.join(model_dir, "sarsa_final.pkl")
        self.save(final)

        return {"episode_rewards": episode_rewards, "episode_scores": episode_scores}
