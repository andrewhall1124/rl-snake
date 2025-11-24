import numpy as np
import pickle
from collections import defaultdict


class QLearningAgent:
    """Q-learning agent with epsilon-greedy exploration."""

    def __init__(self, action_size, learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, seed=None):
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
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: dictionary mapping state tuples to action values
        # Using defaultdict to initialize unseen states to zeros
        self.q_table = defaultdict(lambda: np.zeros(action_size))

        self.rng = np.random.RandomState(seed)

    def get_action(self, state, training=True):
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

    def update(self, state, action, reward, next_state, done):
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
        self.q_table[state_tuple][action] = current_q + self.learning_rate * (target_q - current_q)

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save Q-table to file."""
        # Convert defaultdict to regular dict for pickling
        q_table_dict = dict(self.q_table)
        save_data = {
            'q_table': q_table_dict,
            'epsilon': self.epsilon,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

    def load(self, filepath):
        """Load Q-table from file."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        # Restore Q-table as defaultdict
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))
        self.q_table.update(save_data['q_table'])

        self.epsilon = save_data['epsilon']
        self.action_size = save_data['action_size']
        self.learning_rate = save_data['learning_rate']
        self.discount_factor = save_data['discount_factor']
        self.epsilon_min = save_data['epsilon_min']
        self.epsilon_decay = save_data['epsilon_decay']

        print(f"Q-table loaded from {filepath}")
        print(f"Number of states in Q-table: {len(self.q_table)}")

    def get_stats(self):
        """Get statistics about the Q-table."""
        return {
            'num_states': len(self.q_table),
            'epsilon': self.epsilon
        }
