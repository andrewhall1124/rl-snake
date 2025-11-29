# DQN Agent Documentation

## `get_action(training: bool = True) -> int`

```python
def get_action(training=True):
   features = env.get_features()
   # Explore
   if training and random() < epsilon:
      return random_action()
   # Exploit
   else:
      state_tensor = FloatTensor(features).unsqueeze(0)
      q_values = policy_net(state_tensor)
      return argmax(q_values)
```

## `train_step() -> float`

```python
def train_step():
   if len(replay_buffer) < batch_size:
      return 0.0

   # Sample random batch from replay buffer
   states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

   # Convert to tensors
   states = FloatTensor(states)
   actions = LongTensor(actions)
   rewards = FloatTensor(rewards)
   next_states = FloatTensor(next_states)
   dones = FloatTensor(dones)

   # Current Q values from policy network
   current_q_values = policy_net(states).gather(1, actions.unsqueeze(1))

   # Target Q values from target network
   next_q_values = target_net(next_states).max(1)[0]
   target_q_values = rewards + (1 - dones) * discount_factor * next_q_values

   # Compute loss and backpropagate
   loss = MSELoss()(current_q_values.squeeze(), target_q_values)
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()

   return loss
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

         # 4. Get next state
         next_state = env.get_features()

         # 5. Store transition in replay buffer
         replay_buffer.push(current_state, action, reward, next_state, done)

         # 6. Train on random batch from replay buffer
         loss = train_step()

         total_reward += reward

      # Decay epsilon after episode
      epsilon = max(epsilon_min, epsilon * epsilon_decay)

      # Update target network periodically
      if episode % target_update == 0:
         target_net.load_state_dict(policy_net.state_dict())
```
