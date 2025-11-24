# Snake Reinforecment Learning Project

## Installation

1. Clone or download this project
2. Install dependencies:

```bash
uv sync
```

## Usage

### Q-Learning
#### Training

Train the agent for 5000 episodes (default):

```bash
python scripts/train_q_learning_agent.py
```

Training progress will be displayed every 100 episodes. The Q-table will be saved:
- Periodically every 500 episodes to `models/q_table_episode_<N>.pkl`
- Final Q-table to `models/q_table_final.pkl`

#### Evaluation

Evaluate the trained agent:

```bash
python scripts/evaluate_q_learning_agent.py
```

This will run 100 evaluation episodes (pure exploitation, no exploration) and display statistics.

To evaluate a specific checkpoint:

```bash
python scripts/evaluate_q_learning_agent.py models/q_table_episode_2500.pkl
```

To watch the agent play with visualization:

```bash
python scripts/evaluate_q_learning_agent.py
# Answer 'y' when prompted to watch episodes
```

## Configuration

All hyperparameters are centralized in [config.py](config.py):

### Environment Settings
- `GRID_SIZE`: Size of the square grid (default: 10)
- `MAX_STEPS_PER_EPISODE`: Maximum steps to prevent infinite loops (default: 1000)

### Agent Hyperparameters
- `LEARNING_RATE` (α): How much to update Q-values (default: 0.1)
  - Higher values make the agent learn faster but less stable
  - Lower values make learning slower but more stable
- `DISCOUNT_FACTOR` (γ): Importance of future rewards (default: 0.95)
  - Values close to 1 make the agent more far-sighted
  - Values close to 0 make the agent focus on immediate rewards
- `EPSILON_START`: Initial exploration rate (default: 1.0)
- `EPSILON_DECAY`: Decay rate per episode (default: 0.995)
- `EPSILON_MIN`: Minimum exploration rate (default: 0.01)

### Training Settings
- `NUM_EPISODES`: Number of training episodes (default: 5000)
- `PRINT_INTERVAL`: How often to print progress (default: 100)
- `SAVE_INTERVAL`: How often to save Q-table (default: 500)

### Rewards
Defined in `snake_env.py`:
- **+10**: Eating food
- **-10**: Dying (wall or self collision)
- **-0.01**: Each step (encourages efficiency)

## How Q-Learning Works

Q-learning learns a Q-table that maps (state, action) pairs to expected cumulative rewards. The update rule is:

```
Q(s,a) ← Q(s,a) + α[r + γ max(Q(s',a')) - Q(s,a)]
```

Where:
- `s`: Current state
- `a`: Action taken
- `r`: Reward received
- `s'`: Next state
- `α`: Learning rate
- `γ`: Discount factor

The agent uses **epsilon-greedy exploration**:
- With probability ε: take random action (explore)
- With probability 1-ε: take best known action (exploit)

Over time, ε decays from 1.0 to 0.01, shifting from exploration to exploitation.

## State Representation

The state is represented as an 11-dimensional binary/one-hot vector:

1. **Food Direction (4 features)**: Where is food relative to head?
   - Is food up? (1 or 0)
   - Is food down? (1 or 0)
   - Is food left? (1 or 0)
   - Is food right? (1 or 0)

2. **Danger Detection (3 features)**: Is there danger if we go...
   - Straight? (1 = danger, 0 = safe)
   - Left? (1 = danger, 0 = safe)
   - Right? (1 = danger, 0 = safe)

3. **Current Direction (4 features)**: One-hot encoding
   - Moving up? (1 or 0)
   - Moving right? (1 or 0)
   - Moving down? (1 or 0)
   - Moving left? (1 or 0)

This representation is converted to a tuple for use as a dictionary key in the Q-table.

## Actions

Actions are **relative to the current direction**:
- **0**: Go straight
- **1**: Turn left
- **2**: Turn right

This prevents the agent from making illegal 180° turns (e.g., going directly from up to down).

## Expected Results

With default hyperparameters:
- Early episodes (1-1000): Agent explores randomly, poor performance
- Mid training (1000-3000): Agent starts learning patterns, scores improve
- Late training (3000-5000): Performance stabilizes, consistent food collection

Typical final performance:
- Average score: 5-15 food items per episode
- Max score: 20-40+ food items
- Q-table size: 500-2000 unique states
