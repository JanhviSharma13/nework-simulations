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
        self.actions_fn = actions_fn
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.replay = PrioritizedReplayBuffer(buffer_size, alpha=priority_alpha)

    def select_action(self, state):
        if state is None:
            return None
            
        actions = self.actions_fn(state)
        if not actions:
            return None
            
        if random.random() < self.epsilon:
            return random.choice(actions)
        else:
            q_sum = {a: self.Q1[state][a] + self.Q2[state][a] for a in actions}
            return max(q_sum, key=q_sum.get)

    def update(self, s, a, r, s_, done):
        if s is None or a is None or s_ is None:
            return
            
        # Store experience with initial TD error
        actions_next = self.actions_fn(s_) if not done else []
        max_a = max(actions_next, key=lambda x: self.Q1[s_][x] + self.Q2[s_][x], default=None) if actions_next else None
        target = r if done else r + self.gamma * (self.Q2[s_][max_a] if max_a else 0)
        td_error = abs(target - self.Q1[s][a])
        self.replay.add((s, a, r, s_, done), td_error)

        # Learn from replay buffer
        if len(self.replay) >= self.batch_size:
            batch, indices, _ = self.replay.sample(self.batch_size)
            
            for i, (s_b, a_b, r_b, s_b_next, done_b) in enumerate(batch):
                # Randomly select which Q to update
                if random.random() < 0.5:
                    Q = self.Q1
                    Q_target = self.Q2
                else:
                    Q = self.Q2
                    Q_target = self.Q1
                    
                # Calculate target
                actions_next = self.actions_fn(s_b_next) if not done_b else []
                max_a_next = max(actions_next, key=lambda x: self.Q1[s_b_next][x] + self.Q2[s_b_next][x], default=None) if actions_next else None
                
                if done_b:
                    target = r_b
                else:
                    target = r_b + self.gamma * (Q_target[s_b_next][max_a_next] if max_a_next else 0)
                
                # Update Q-value
                delta = target - Q[s_b][a_b]
                Q[s_b][a_b] += self.alpha * delta
                
                # Update priority
                self.replay.update(indices[i], abs(delta))
        
        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)