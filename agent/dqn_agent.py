import os
import pickle
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from tqdm import tqdm

from agent.base_agent import BaseAgent

if TYPE_CHECKING:
    from environment.snake_env import SnakeEnv


class DQN(nn.Module):
    """Deep Q-Network model."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initialize DQN model.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layers
            output_size: Number of actions
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int) -> None:
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum size of buffer
        """
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: NDArray[np.int8],
        action: int,
        reward: float,
        next_state: NDArray[np.int8],
        done: bool,
    ) -> None:
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """Sample random batch from buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in indices]
        )

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """Deep Q-Network agent with experience replay and target network."""

    def __init__(
        self,
        env: "SnakeEnv",
        hidden_size: int = 128,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update: int = 10,
        seed: int | None = None,
    ) -> None:
        """
        Initialize DQN agent.

        Args:
            env: Environment instance to interact with
            hidden_size: Size of hidden layers in neural network
            learning_rate: Learning rate for optimizer
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            buffer_capacity: Capacity of replay buffer
            batch_size: Batch size for training
            target_update: Frequency of target network updates (in episodes)
            seed: Random seed for reproducibility
        """
        self.env: "SnakeEnv" = env
        self.action_space: int = env.action_space
        self.feature_size: int = 11  # From env.get_features()
        self.hidden_size: int = hidden_size
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min
        self.batch_size: int = batch_size
        self.target_update: int = target_update

        # Set random seeds
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Device configuration
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )

        # Initialize networks
        self.policy_net = DQN(self.feature_size, hidden_size, self.action_space).to(
            self.device
        )
        self.target_net = DQN(self.feature_size, hidden_size, self.action_space).to(
            self.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def get_action(self, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Selected action
        """
        # Get feature vector from environment
        features = self.env.get_features()

        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_space)
        else:
            # Exploit: choose best action using policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def store_transition(
        self,
        state: NDArray[np.int8],
        action: int,
        reward: float,
        next_state: NDArray[np.int8],
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """
        Perform one training step using experience replay.

        Returns:
            Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = (
                rewards + (1 - dones) * self.discount_factor * next_q_values
            )

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self) -> None:
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Save model to file."""
        save_data = {
            "policy_net_state": self.policy_net.state_dict(),
            "target_net_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "feature_size": self.feature_size,
            "hidden_size": self.hidden_size,
            "action_space": self.action_space,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        torch.save(save_data, filepath)

    def load(self, filepath: str) -> None:
        """Load model from file."""
        save_data = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(save_data["policy_net_state"])
        self.target_net.load_state_dict(save_data["target_net_state"])
        self.optimizer.load_state_dict(save_data["optimizer_state"])
        self.epsilon = save_data["epsilon"]
        self.feature_size = save_data["feature_size"]
        self.hidden_size = save_data["hidden_size"]
        self.action_space = save_data["action_space"]
        self.learning_rate = save_data["learning_rate"]
        self.discount_factor = save_data["discount_factor"]
        self.epsilon_min = save_data["epsilon_min"]
        self.epsilon_decay = save_data["epsilon_decay"]

        print(f"Model loaded from {filepath}")
        print(f"Epsilon: {self.epsilon:.3f}")

    def get_stats(self) -> dict[str, int | float]:
        """Get statistics about the agent."""
        return {
            "buffer_size": len(self.replay_buffer),
            "epsilon": self.epsilon,
        }

    def train(
        self,
        num_episodes: int,
        save_interval: int = 100,
        model_dir: str = "models",
    ) -> dict[str, list[float]]:
        """
        Train the agent on the given environment.

        Args:
            num_episodes: Number of episodes to train
            save_interval: Interval for saving model checkpoints
            model_dir: Directory to save model checkpoints

        Returns:
            Training metrics including episode_rewards and episode_scores
        """
        print("=" * 50)
        print("Snake DQN Training")
        print("=" * 50)
        print(f"Episodes: {num_episodes}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Discount Factor: {self.discount_factor}")
        print(
            f"Epsilon: {self.epsilon} -> {self.epsilon_min} (decay: {self.epsilon_decay})"
        )
        print(f"Batch Size: {self.batch_size}")
        print(f"Buffer Capacity: {self.replay_buffer.buffer.maxlen}")
        print(f"Target Update Frequency: {self.target_update} episodes")
        print(f"Device: {self.device}")
        print("=" * 50)

        # Create directories
        os.makedirs(model_dir, exist_ok=True)

        # Metrics tracking
        episode_rewards = []
        episode_scores = []
        episode_losses = []
        reward_window = deque(maxlen=100)
        score_window = deque(maxlen=100)

        # Training loop
        for episode in tqdm(range(1, num_episodes + 1), desc="Training"):
            self.env.reset()
            total_reward = 0
            total_loss = 0
            steps = 0
            done = False

            while not done:
                # Get current state
                current_state = self.env.get_features()

                # Select and perform action
                action = self.get_action(training=True)
                _, reward, done, info = self.env.step(action)

                # Get next state
                next_state = self.env.get_features()

                # Store transition
                self.store_transition(current_state, action, reward, next_state, done)

                # Train on batch
                loss = self.train_step()
                total_loss += loss

                # Update counters
                total_reward += reward
                steps += 1

            # Decay epsilon after episode
            self.decay_epsilon()

            # Update target network periodically
            if episode % self.target_update == 0:
                self.update_target_network()

            # Track metrics
            episode_rewards.append(total_reward)
            episode_scores.append(info["score"])
            episode_losses.append(total_loss / max(steps, 1))
            reward_window.append(total_reward)
            score_window.append(info["score"])

            # Save model periodically
            if episode % save_interval == 0:
                save_path = os.path.join(model_dir, f"dqn_episode_{episode}.pt")
                self.save(save_path)

        # Final save
        final_path = os.path.join(model_dir, "dqn_final.pt")
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
        print(f"Buffer Size: {len(self.replay_buffer)}")
        print("=" * 50)

        return {
            "episode_rewards": episode_rewards,
            "episode_scores": episode_scores,
            "episode_losses": episode_losses,
        }
