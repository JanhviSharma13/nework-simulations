import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from edql_rg_env import EDQLRGEnv
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt

# --- Q-Network ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.model(x)

# --- Prioritized Replay Buffer with TD Error Logging ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha  # how much prioritization is used
        self.td_errors = []  # Track TD errors for logging

    def push(self, transition):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(weights),
            indices
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
        # Store TD errors for logging
        self.td_errors.extend(priorities)

    def get_td_error_stats(self):
        """Get TD error statistics for logging"""
        if not self.td_errors:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        
        td_array = np.array(self.td_errors)
        return {
            "mean": float(np.mean(td_array)),
            "std": float(np.std(td_array)),
            "min": float(np.min(td_array)),
            "max": float(np.max(td_array))
        }

    def clear_td_errors(self):
        """Clear TD errors after logging"""
        self.td_errors = []

    def __len__(self):
        return len(self.buffer)

# --- Utility: Flatten observation ---
def flatten(obs):
    return np.array(obs, dtype=np.float32).flatten()

# --- Soft Target Update Function ---
def soft_update(target, source, tau=0.005):
    """
    Soft update target network parameters using exponential moving average
    target = tau * source + (1 - tau) * target
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

# --- Enhanced CSV Logging Class with TD Error Support ---
class TrainingLogger:
    def __init__(self, filename="training_log.csv"):
        self.filename = filename
        self.fieldnames = [
            'episode', 'total_steps', 'epsilon', 'total_reward', 'avg_loss',
            'avg_mwt', 'avg_speed', 'route_steps', 'learning_status',
            'buffer_size', 'training_steps', 'best_action_freq',
            'worst_action_freq', 'action_entropy', 'q1_avg', 'q2_avg',
            'td_error_mean', 'td_error_std', 'td_error_min', 'td_error_max'
        ]
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_episode(self, episode_data):
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(episode_data)

# --- TD Error Histogram Plotting ---
def plot_td_error_histogram(td_errors, episode, save_dir="td_error_plots"):
    """Plot and save TD error histogram"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.figure(figsize=(10, 6))
    plt.hist(td_errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'TD Error Distribution - Episode {episode}')
    plt.xlabel('TD Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {np.mean(td_errors):.4f}\nStd: {np.std(td_errors):.4f}\nMin: {np.min(td_errors):.4f}\nMax: {np.max(td_errors):.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/td_error_episode_{episode}.png', dpi=300, bbox_inches='tight')
    plt.close()

# --- Training Function ---
def train_edql_rg():
    # Environment setup for Rakab-Ganj
    env = EDQLRGEnv(
        net_file="rg.net.xml",
        route_file="rg.rou.xml",
        use_gui=False,  # Set to True for debugging
        max_steps=2000
    )
    
    state_dim = 4  # Updated for Rakab-Ganj (speed, lane_pos, lane_index, edge_progress)
    action_dim = 8
    
    # Networks
    q1 = QNetwork(state_dim, action_dim)
    q2 = QNetwork(state_dim, action_dim)
    target_q1 = QNetwork(state_dim, action_dim)
    target_q2 = QNetwork(state_dim, action_dim)
    
    # Copy weights to target networks
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())
    
    # Optimizers
    optimizer1 = optim.Adam(q1.parameters(), lr=0.001)
    optimizer2 = optim.Adam(q2.parameters(), lr=0.001)
    
    # Replay buffer
    replay_buffer = PrioritizedReplayBuffer(100000)
    
    # Training parameters
    episodes = 1000
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 64
    gamma = 0.99
    tau = 0.005
    
    # Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TrainingLogger(f"edql_rg_training_log_{timestamp}.csv")
    
    # Training loop
    total_steps = 0
    training_steps = 0
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = flatten(state)
        total_reward = 0
        route_steps = 0
        episode_td_errors = []
        
        # Episode statistics
        action_counts = [0] * action_dim
        q1_values = []
        q2_values = []
        
        while True:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q1_val = q1(torch.FloatTensor(state).unsqueeze(0))
                    q2_val = q2(torch.FloatTensor(state).unsqueeze(0))
                    action = torch.min(q1_val, q2_val).argmax().item()
            
            action_counts[action] += 1
            
            # Take action
            next_state, reward, done = env.step(action)
            next_state = flatten(next_state)
            
            # Store transition
            replay_buffer.push((state, action, reward, next_state, done))
            
            # Training step
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(batch_size)
                
                # Compute target Q values
                with torch.no_grad():
                    target_q1_next = target_q1(next_states)
                    target_q2_next = target_q2(next_states)
                    target_q_next = torch.min(target_q1_next, target_q2_next)
                    target_q = rewards + gamma * target_q_next * (1 - dones)
                
                # Compute current Q values
                q1_current = q1(states).gather(1, actions.unsqueeze(1))
                q2_current = q2(states).gather(1, actions.unsqueeze(1))
                
                # Compute TD errors
                td_error1 = torch.abs(target_q - q1_current).detach().numpy()
                td_error2 = torch.abs(target_q - q2_current).detach().numpy()
                td_errors = np.maximum(td_error1, td_error2).flatten()
                
                # Store TD errors for episode logging
                episode_td_errors.extend(td_errors)
                
                # Update priorities
                replay_buffer.update_priorities(indices, td_errors)
                
                # Compute losses
                loss1 = (weights * (target_q - q1_current) ** 2).mean()
                loss2 = (weights * (target_q - q2_current) ** 2).mean()
                
                # Backward pass
                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()
                
                optimizer2.zero_grad()
                loss2.backward()
                optimizer2.step()
                
                # Soft update target networks
                soft_update(target_q1, q1, tau)
                soft_update(target_q2, q2, tau)
                
                training_steps += 1
                
                # Store Q values for logging
                q1_values.extend(q1_current.detach().numpy().flatten())
                q2_values.extend(q2_current.detach().numpy().flatten())
            
            state = next_state
            total_reward += reward
            route_steps += 1
            total_steps += 1
            
            if done:
                break
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Calculate episode statistics
        best_action = np.argmax(action_counts)
        worst_action = np.argmin(action_counts)
        best_action_freq = action_counts[best_action] / sum(action_counts) if sum(action_counts) > 0 else 0
        worst_action_freq = action_counts[worst_action] / sum(action_counts) if sum(action_counts) > 0 else 0
        
        # Calculate action entropy
        action_probs = np.array(action_counts) / sum(action_counts) if sum(action_counts) > 0 else np.ones(action_dim) / action_dim
        action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
        
        # Get TD error statistics
        td_stats = replay_buffer.get_td_error_stats()
        
        # Log episode data
        episode_data = {
            'episode': episode,
            'total_steps': total_steps,
            'epsilon': epsilon,
            'total_reward': total_reward,
            'avg_loss': 0,  # Could be computed if needed
            'avg_mwt': 0,   # Could be computed if needed
            'avg_speed': 0,  # Could be computed if needed
            'route_steps': route_steps,
            'learning_status': 'active',
            'buffer_size': len(replay_buffer),
            'training_steps': training_steps,
            'best_action_freq': best_action_freq,
            'worst_action_freq': worst_action_freq,
            'action_entropy': action_entropy,
            'q1_avg': np.mean(q1_values) if q1_values else 0,
            'q2_avg': np.mean(q2_values) if q2_values else 0,
            'td_error_mean': td_stats['mean'],
            'td_error_std': td_stats['std'],
            'td_error_min': td_stats['min'],
            'td_error_max': td_stats['max']
        }
        
        logger.log_episode(episode_data)
        
        # Plot TD error histogram every 50 episodes
        if episode % 50 == 0 and episode_td_errors:
            plot_td_error_histogram(episode_td_errors, episode)
        
        # Clear TD errors after logging
        replay_buffer.clear_td_errors()
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}/{episodes} - Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f} - Steps: {route_steps}")
            print(f"TD Error Stats - Mean: {td_stats['mean']:.4f}, Std: {td_stats['std']:.4f}")
    
    # Save trained models
    torch.save(q1.state_dict(), f"trained_edql_rg_qnet1_{timestamp}.pth")
    torch.save(q2.state_dict(), f"trained_edql_rg_qnet2_{timestamp}.pth")
    
    env.close()
    print("Training completed!")

if __name__ == "__main__":
    train_edql_rg() 