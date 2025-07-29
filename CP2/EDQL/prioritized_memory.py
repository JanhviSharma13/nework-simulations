import random
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0
        self.epsilon = 1e-6

    def add(self, experience, td_error):
        priority = (abs(td_error) + self.epsilon) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Importance-sampling weights (optional, not used in agent update yet)
        total = len(self.buffer)
        weights = (total * probs[indices]) ** -1
        weights /= weights.max()

        return samples, indices, weights

    def update(self, idx, new_td_error):
        new_priority = (abs(new_td_error) + self.epsilon) ** self.alpha
        self.priorities[idx] = new_priority

    def __len__(self):
        return len(self.buffer)
