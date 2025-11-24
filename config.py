"""
Configuration file for Snake Q-Learning project.
Centralized hyperparameters and settings.
"""

# Environment settings
GRID_SIZE = 10
MAX_STEPS_PER_EPISODE = 1000

# Agent hyperparameters
LEARNING_RATE = 0.1  # Alpha: how much to update Q-values
DISCOUNT_FACTOR = 0.95  # Gamma: importance of future rewards
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.995  # Decay rate per episode
EPSILON_MIN = 0.01  # Minimum exploration rate

# Training settings
NUM_EPISODES = 5000
PRINT_INTERVAL = 100  # Print progress every N episodes
SAVE_INTERVAL = 500  # Save Q-table every N episodes

# Logging
LOG_DIR = "logs"
MODEL_DIR = "models"

# Random seed for reproducibility
RANDOM_SEED = 42

# Evaluation settings
EVAL_EPISODES = 100  # Number of episodes to run during evaluation
RENDER_EVAL = False  # Whether to render during evaluation
