import os
import random
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.typing import NDArray
from tqdm import tqdm

from agent.base_agent import BaseAgent

if TYPE_CHECKING:
    from environment.snake_env import SnakeEnv


class DQN(nn.Module):
    """Deep Q-Network for approximating Q-values."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        """
        Initialize DQN network.

        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_size: Size of hidden layers
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

        # Better initialization for more stable training
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(self, capacity: int, seed: int | None = None):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum size of buffer
            seed: Random seed for reproducibility
        """
        self.buffer = deque(maxlen=capacity)
        if seed is not None:
            random.seed(seed)

    def push(
        self,
        state: NDArray[np.int8],
        action: int,
        reward: float,
        next_state: NDArray[np.int8],
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """Deep Q-Network agent with experience replay and target network."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.0005,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 50000,
        batch_size: int = 64,
        hidden_size: int = 128,
        target_update: int = 10,
        seed: int | None = None,
    ) -> None:
        """
        Initialize DQN agent.

        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            hidden_size: Size of hidden layers in network
            target_update: Frequency of target network updates
            seed: Random seed for reproducibility
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update

        # Set random seeds
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Device configuration
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )

        # Q-Networks
        self.policy_net = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and replay buffer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size, seed)

    def get_action(self, state: NDArray[np.int8], training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state observation (flattened grid)
            training: Whether the agent is in training mode

        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            # Ensure state is flattened
            state_flat = state.flatten() if state.ndim > 1 else state
            state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def update(
        self,
        state: NDArray[np.int8],
        action: int,
        reward: float,
        next_state: NDArray[np.int8],
        done: bool,
    ) -> float | None:
        """
        Store transition in replay buffer and train if enough samples.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished

        Returns:
            Loss value if training occurred, None otherwise
        """
        self.memory.push(state, action, reward, next_state, done)

        if len(self.memory) < self.batch_size:
            return None

        return self._train_step()

    def _train_step(self) -> float:
        """Perform one training step on a batch from replay buffer."""
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q values from target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.discount_factor * next_q

        # Compute loss
        loss = F.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self) -> None:
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath: str) -> None:
        """Save agent state to file."""
        save_data = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        torch.save(save_data, filepath)

    def load(self, filepath: str) -> None:
        """Load agent state from file."""
        save_data = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(save_data["policy_net_state_dict"])
        self.target_net.load_state_dict(save_data["target_net_state_dict"])
        self.optimizer.load_state_dict(save_data["optimizer_state_dict"])
        self.epsilon = save_data["epsilon"]
        self.state_size = save_data["state_size"]
        self.action_size = save_data["action_size"]
        self.learning_rate = save_data["learning_rate"]
        self.discount_factor = save_data["discount_factor"]
        self.epsilon_min = save_data["epsilon_min"]
        self.epsilon_decay = save_data["epsilon_decay"]

        print(f"Model loaded from {filepath}")
        print(f"Epsilon: {self.epsilon:.3f}")

    def train(
        self,
        env: "SnakeEnv",
        num_episodes: int,
        save_interval: int = 100,
        model_dir: str = "models",
    ) -> dict:
        """
        Train the DQN agent on the given environment.

        Args:
            env: Environment instance to train on
            num_episodes: Number of episodes to train
            save_interval: Interval for saving model checkpoints
            model_dir: Directory to save model checkpoints

        Returns:
            dict: Training metrics including episode_rewards and episode_scores
        """
        print("=" * 50)
        print("Snake DQN Training")
        print("=" * 50)
        print(f"Episodes: {num_episodes}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Discount Factor: {self.discount_factor}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Buffer Size: {len(self.memory.buffer)}")
        print(
            f"Epsilon: {self.epsilon} -> {self.epsilon_min} (decay: {self.epsilon_decay})"
        )
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
            state = env.reset()
            total_reward = 0
            total_loss = 0
            loss_count = 0
            done = False

            while not done:
                # Select and perform action
                action = self.get_action(state, training=True)
                next_state, reward, done, info = env.step(action)

                # Update network
                loss = self.update(state, action, reward, next_state, done)
                if loss is not None:
                    total_loss += loss
                    loss_count += 1

                # Update counters
                total_reward += reward
                state = next_state

            # Decay epsilon after episode
            self.decay_epsilon()

            # Update target network periodically
            if episode % self.target_update == 0:
                self.update_target_network()

            # Track metrics
            avg_loss = total_loss / loss_count if loss_count > 0 else 0
            episode_rewards.append(total_reward)
            episode_scores.append(info["score"])
            episode_losses.append(avg_loss)
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
        print(f"Buffer Size: {len(self.memory)}")
        print("=" * 50)

        return {
            "episode_rewards": episode_rewards,
            "episode_scores": episode_scores,
            "episode_losses": episode_losses,
        }
