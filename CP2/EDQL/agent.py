import random
import numpy as np
from collections import defaultdict
from prioritized_memory import PrioritizedReplayBuffer

class DQLPERAgent:
    def __init__(self, actions_fn, alpha=0.5, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, buffer_size=2000,
                 batch_size=32, priority_alpha=0.6):
        self.Q1 = defaultdict(lambda: defaultdict(float))
        self.Q2 = defaultdict(lambda: defaultdict(float))
        self.actions_fn = actions_fn  # function(state) → list of valid actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.replay = PrioritizedReplayBuffer(buffer_size, alpha=priority_alpha)

    def select_action(self, state):
        actions = self.actions_fn(state)
        if not actions:
            return None
        if random.random() < self.epsilon:
            return random.choice(actions)
        else:
            q_sum = {a: self.Q1[state][a] + self.Q2[state][a] for a in actions}
            return max(q_sum, key=q_sum.get)

    def update(self, s, a, r, s_, done):
        # Estimate TD error for priority
        actions_next = self.actions_fn(s_) if not done else []
        max_a = max(actions_next, key=lambda x: self.Q1[s_][x] + self.Q2[s_][x], default=0)
        target = r if done else r + self.gamma * self.Q2[s_][max_a]
        td_error = abs(target - self.Q1[s][a])  # Use Q1 for priority
        self.replay.add((s, a, r, s_, done), td_error)

        if len(self.replay) < self.batch_size:
            return

        batch, indices, weights = self.replay.sample(self.batch_size)

        for i, (s, a, r, s_, done) in enumerate(batch):
            if random.random() < 0.5:
                max_a = max(self.Q1[s_], key=self.Q1[s_].get, default=a) if not done else 0
                target = r if done else r + self.gamma * self.Q2[s_][max_a]
                delta = target - self.Q1[s][a]
                self.Q1[s][a] += self.alpha * delta
            else:
                max_a = max(self.Q2[s_], key=self.Q2[s_].get, default=a) if not done else 0
                target = r if done else r + self.gamma * self.Q1[s_][max_a]
                delta = target - self.Q2[s][a]
                self.Q2[s][a] += self.alpha * delta

            # Update priority in buffer
            td_error = abs(delta)
            self.replay.update(indices[i], td_error)

        # Decay ε
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
