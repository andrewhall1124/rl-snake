"""
Configuration file for Snake Q-Learning project.
Centralized hyperparameters and settings.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentConfig:
    """Environment configuration settings."""

    grid_size: int = 10
    max_steps_per_episode: int = 1000


@dataclass(frozen=True)
class AgentConfig:
    """Agent hyperparameters."""

    learning_rate: float = 0.1  # Alpha: how much to update Q-values
    discount_factor: float = 0.95  # Gamma: importance of future rewards
    epsilon_start: float = 1.0  # Initial exploration rate
    epsilon_decay: float = 0.995  # Decay rate per episode
    epsilon_min: float = 0.01  # Minimum exploration rate


@dataclass(frozen=True)
class TrainingConfig:
    """Training settings."""

    num_episodes: int = 10000
    print_interval: int = num_episodes  # Print progress every N episodes
    save_interval: int = num_episodes  # Save Q-table every N episodes


@dataclass(frozen=True)
class LoggingConfig:
    """Logging settings."""

    log_dir: str = "logs"
    model_dir: str = "models"


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation settings."""

    num_episodes: int = 100  # Number of episodes to run during evaluation
    render: bool = False  # Whether to render during evaluation


@dataclass(frozen=True)
class Config:
    """Main configuration dataclass."""

    environment: EnvironmentConfig = EnvironmentConfig()
    agent: AgentConfig = AgentConfig()
    training: TrainingConfig = TrainingConfig()
    logging: LoggingConfig = LoggingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    random_seed: int = 42


# Default configuration instance
config = Config()
