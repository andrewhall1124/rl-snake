# Q-Learning Agent Documentation

## `get_action(training: bool = True) -> int`

```python
def get_action(training=True):
   state = env.get_features()
   # Explore
   if training and random() < epsilon:
      return random_action()
   # Exploit
   else:
      q_values = q_table[state]
      return argmax(q_values)
```

## `train(num_episodes: int)`

```python
def train(num_episodes: int):
   for episode in range(1, num_episodes + 1):
      # Start new episode
      env.reset()
      total_reward = 0
      done = False

      while not done:
         # 1. Get current state
         current_state = env.get_features()

         # 2. Choose action (epsilon-greedy)
         action = get_action(training=True)

         # 3. Take action in environment
         _, reward, done, info = env.step(action)

         # 4. Update Q-table using Q-learning update rule
         next_state = env.get_features()
         current_q = q_table[current_state][action]

         if done:
            target_q = reward
         else:
            max_next_q = max(q_table[next_state])
            target_q = reward + discount_factor * max_next_q

         q_table[current_state][action] = current_q + learning_rate * (target_q - current_q)

      # Decay epsilon
      epsilon = max(epsilon_min, epsilon * epsilon_decay)
```
