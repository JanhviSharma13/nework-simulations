import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import csv
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import sys

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the EDQL environments
try:
    from edql_rakabganj_env import EDQLRakabGanjEnv
    from edql_cp_env import EDQLCPEnv
    from edql_cp2_env import EDQLCP2Env
    from edql_safdarjung_env import EDQLSafdarjungEnv
    from edql_chandnichowk_env import EDQLChandniChowkEnv
except ImportError:
    print("⚠️ EDQL environment files not found. Creating simple test environments...")
    # Create simple test environments if imports fail
    class SimpleTestEnv:
        def __init__(self, net_file, route_file, use_gui=False, max_steps=1000):
            self.net_file = net_file
            self.route_file = route_file
            self.use_gui = use_gui
            self.max_steps = max_steps
            self.step_count = 0
            
        def reset(self):
            self.step_count = 0
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), {}
            
        def step(self, action):
            self.step_count += 1
            obs = np.array([random.uniform(0, 20), random.uniform(0, 100), 
                           random.randint(0, 2), random.uniform(0, 1)], dtype=np.float32)
            reward = random.uniform(-10, 10)
            done = self.step_count >= self.max_steps
            return obs, reward, done
            
        def close(self):
            pass
    
    EDQLRakabGanjEnv = SimpleTestEnv
    EDQLCPEnv = SimpleTestEnv
    EDQLCP2Env = SimpleTestEnv
    EDQLSafdarjungEnv = SimpleTestEnv
    EDQLChandniChowkEnv = SimpleTestEnv

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
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.td_errors = []  # Track TD errors for logging

    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(float(max_priority))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return [], [], [], [], [], []

        # Calculate sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), indices)

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            if idx < len(self.priorities):
                self.priorities[idx] = float(abs(td_error) + 1e-6)
                self.td_errors.append(float(abs(td_error)))

    def get_td_error_stats(self):
        if not self.td_errors:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        
        errors = np.array(self.td_errors)
        return {
            "mean": float(np.mean(errors)),
            "std": float(np.std(errors)),
            "min": float(np.min(errors)),
            "max": float(np.max(errors))
        }

    def clear_td_errors(self):
        self.td_errors = []

# --- EDQL Agent with TD Error Logging ---
class EDQLAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, 
                 buffer_size=10000, batch_size=32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Q-Networks (Q1 and Q2 for Double Q-Learning)
        self.q1 = QNetwork(state_dim, action_dim)
        self.q2 = QNetwork(state_dim, action_dim)
        
        # Target networks
        self.q1_target = QNetwork(state_dim, action_dim)
        self.q2_target = QNetwork(state_dim, action_dim)
        
        # Copy weights to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.optimizer1 = optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.optimizer2 = optim.Adam(self.q2.parameters(), lr=learning_rate)
        
        # Replay buffer with TD error logging
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        
        # TD error logging
        self.episode_td_errors = []
        self.td_error_log = []

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q1(state_tensor)
        return q_values.argmax().item()

    def train(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones, indices = self.replay_buffer.sample(self.batch_size)
        
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.BoolTensor(dones)

        # Current Q-values
        current_q1 = self.q1(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        current_q2 = self.q2(states_tensor).gather(1, actions_tensor.unsqueeze(1))

        # Next Q-values (Double Q-Learning)
        with torch.no_grad():
            next_actions = self.q1(next_states_tensor).argmax(1)
            next_q1 = self.q1_target(next_states_tensor).gather(1, next_actions.unsqueeze(1))
            next_q2 = self.q2_target(next_states_tensor).gather(1, next_actions.unsqueeze(1))
            
            # Use minimum of Q1 and Q2 for target (clipped double Q-learning)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards_tensor.unsqueeze(1) + (self.gamma * next_q * ~dones_tensor.unsqueeze(1))

        # Calculate TD errors
        td_error1 = target_q - current_q1
        td_error2 = target_q - current_q2

        # Loss
        loss1 = (td_error1 ** 2).mean()
        loss2 = (td_error2 ** 2).mean()

        # Backward pass
        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()

        # Update priorities based on TD errors
        td_errors = torch.max(td_error1.abs(), td_error2.abs()).detach().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        # Log TD errors for this training step
        self.episode_td_errors.extend(td_errors.flatten().tolist())

        return (loss1.item() + loss2.item()) / 2

    def update_target_networks(self):
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_episode_td_stats(self):
        if not self.episode_td_errors:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        
        errors = np.array(self.episode_td_errors)
        return {
            "mean": float(np.mean(errors)),
            "std": float(np.std(errors)),
            "min": float(np.min(errors)),
            "max": float(np.max(errors))
        }

    def clear_episode_td_errors(self):
        self.episode_td_errors = []

# --- Model Persistence Functions ---
def auto_load_latest_model(area_name):
    """Auto-load the latest model for an area"""
    pattern = f"edql_model_episode_*_{area_name.lower()}.pth"
    model_files = glob.glob(pattern)
    
    if not model_files:
        print(f"📁 No existing model found for {area_name}. Starting fresh training.")
        return None, 0, 1.0
    
    # Find the latest model by episode number
    latest_model = max(model_files, key=lambda x: int(x.split('_')[2]))
    episode_num = int(latest_model.split('_')[2])
    
    print(f"📁 Loading latest model: {latest_model} (Episode {episode_num})")
    return latest_model, episode_num, None

def load_model_checkpoint(agent, model_path, start_episode=0, epsilon=None):
    """Load model checkpoint and resume training"""
    try:
        checkpoint = torch.load(model_path)
        agent.q1.load_state_dict(checkpoint['q1_state_dict'])
        agent.q2.load_state_dict(checkpoint['q2_state_dict'])
        agent.q1_target.load_state_dict(checkpoint['q1_state_dict'])
        agent.q2_target.load_state_dict(checkpoint['q2_state_dict'])
        
        if epsilon is None:
            epsilon = checkpoint.get('epsilon', 1.0)
        
        agent.epsilon = epsilon
        print(f"✅ Model loaded successfully. Starting from episode {start_episode}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def save_model_checkpoint(agent, episode, area_name):
    """Save model checkpoint with area-specific naming"""
    model_filename = f"edql_model_episode_{episode}_{area_name.lower()}.pth"
    torch.save({
        'q1_state_dict': agent.q1.state_dict(),
        'q2_state_dict': agent.q2.state_dict(),
        'episode': episode,
        'epsilon': agent.epsilon
    }, model_filename)
    print(f"💾 Model saved: {model_filename}")

# --- Training Function with TD Error Logging and Persistence ---
def train_edql_delhi(env_class, env_args, agent, area_name, episodes=1000, max_steps=1000, 
                     target_update_freq=100, log_freq=10, save_freq=100, auto_load=True):
    """Train EDQL agent with comprehensive TD error logging and model persistence"""
    
    # Auto-load latest model if enabled
    start_episode = 0
    if auto_load:
        model_path, loaded_episode, loaded_epsilon = auto_load_latest_model(area_name)
        if model_path:
            if load_model_checkpoint(agent, model_path, loaded_episode, loaded_epsilon):
                start_episode = loaded_episode
                episodes += start_episode  # Continue training for full duration
    
    # Create CSV logger for TD errors
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"training_log_{area_name.lower()}_{timestamp}.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'total_reward', 'steps', 'epsilon', 'loss', 
                     'td_mean', 'td_std', 'td_min', 'td_max']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        print(f"🎯 Starting EDQL training for {area_name} with TD error logging")
        print(f"📊 TD error log: {csv_filename}")
        print(f"🔄 Starting from episode {start_episode}")
        
        for episode in range(start_episode, episodes):
            env = env_class(**env_args)
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            
            # Clear episode TD errors
            agent.clear_episode_td_errors()
            
            for step in range(max_steps):
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                
                # Store experience
                agent.replay_buffer.add(state, action, reward, next_state, done)
                
                # Train agent
                loss = agent.train()
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Update target networks
            if episode % target_update_freq == 0:
                agent.update_target_networks()
            
            # Decay epsilon
            agent.decay_epsilon()
            
            # Get TD error statistics
            td_stats = agent.get_episode_td_stats()
            
            # Log to CSV
            log_data = {
                'episode': episode,
                'total_reward': total_reward,
                'steps': steps,
                'epsilon': agent.epsilon,
                'loss': loss if 'loss' in locals() else 0.0,
                'td_mean': td_stats['mean'],
                'td_std': td_stats['std'],
                'td_min': td_stats['min'],
                'td_max': td_stats['max']
            }
            writer.writerow(log_data)
            
            # Print progress
            if episode % log_freq == 0:
                print(f"Episode {episode}/{episodes} | "
                      f"Reward: {total_reward:.2f} | "
                      f"Steps: {steps} | "
                      f"Epsilon: {agent.epsilon:.3f} | "
                      f"TD Error: {td_stats['mean']:.4f} ± {td_stats['std']:.4f}")
            
            # Save model
            if episode % save_freq == 0:
                save_model_checkpoint(agent, episode, area_name)
            
            env.close()
        
        # Save final model
        save_model_checkpoint(agent, episodes-1, area_name)
        
        print(f"✅ {area_name} training completed! TD error log saved to: {csv_filename}")
        return csv_filename

# --- TD Error Histogram Plotting ---
def plot_td_error_histogram(csv_filename, save_plot=True):
    """Plot TD error histogram from CSV log"""
    try:
        import pandas as pd
        
        # Read CSV data
        df = pd.read_csv(csv_filename)
        
        # Create histogram
        plt.figure(figsize=(12, 8))
        
        # Plot TD error distribution
        plt.subplot(2, 2, 1)
        plt.hist(df['td_mean'], bins=50, alpha=0.7, color='blue')
        plt.title('TD Error Mean Distribution')
        plt.xlabel('TD Error Mean')
        plt.ylabel('Frequency')
        
        # Plot TD error std distribution
        plt.subplot(2, 2, 2)
        plt.hist(df['td_std'], bins=50, alpha=0.7, color='red')
        plt.title('TD Error Std Distribution')
        plt.xlabel('TD Error Std')
        plt.ylabel('Frequency')
        
        # Plot TD error over episodes
        plt.subplot(2, 2, 3)
        plt.plot(df['episode'], df['td_mean'], label='Mean', alpha=0.7)
        plt.plot(df['episode'], df['td_std'], label='Std', alpha=0.7)
        plt.title('TD Error Statistics Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('TD Error')
        plt.legend()
        
        # Plot reward vs TD error
        plt.subplot(2, 2, 4)
        plt.scatter(df['td_mean'], df['total_reward'], alpha=0.5)
        plt.title('Reward vs TD Error Mean')
        plt.xlabel('TD Error Mean')
        plt.ylabel('Total Reward')
        
        plt.tight_layout()
        
        if save_plot:
            plot_filename = csv_filename.replace('.csv', '_histogram.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"📊 TD error histogram saved to: {plot_filename}")
        
        plt.show()
        
    except ImportError:
        print("⚠️ pandas not available. Skipping histogram plot.")
    except Exception as e:
        print(f"❌ Error plotting histogram: {e}")

# --- Main Training Function ---
def main():
    """Main function to train EDQL on all Delhi networks with TD error logging and persistence"""
    print("🚀 EDQL Training with TD Error Logging and Model Persistence for Delhi Networks")
    print("=" * 80)
    
    # Define Delhi areas and their configurations
    delhi_areas = [
        {
            "name": "CP2",
            "env_class": EDQLCP2Env,
            "env_args": {
                "net_file": "EDQL-CP2/cp2_cleaned.net.xml",
                "route_file": "EDQL-CP2/cp2_simple.rou.xml",
                "use_gui": False,
                "max_steps": 1000
            }
        },
        {
            "name": "RakabGanj",
            "env_class": EDQLRakabGanjEnv,
            "env_args": {
                "net_file": "EDQL-RakabGanj/rakabganj_cleaned.net.xml",
                "route_file": "EDQL-RakabGanj/rakabganj_simple.rou.xml",
                "use_gui": False,
                "max_steps": 1000
            }
        },
        {
            "name": "Safdarjung",
            "env_class": EDQLSafdarjungEnv,
            "env_args": {
                "net_file": "../../Safdarjung/safdarjung_cleaned.net.xml",
                "route_file": "../../Safdarjung/safdarjung_simple.rou.xml",
                "use_gui": False,
                "max_steps": 1000
            }
        },
        {
            "name": "ChandniChowk",
            "env_class": EDQLChandniChowkEnv,
            "env_args": {
                "net_file": "../../Chandni-Chowk/chandnichowk_cleaned.net.xml",
                "route_file": "../../Chandni-Chowk/chandnichowk_simple.rou.xml",
                "use_gui": False,
                "max_steps": 1000
            }
        }
    ]
    
    training_logs = {}
    
    for area in delhi_areas:
        print(f"\n🏗️ Training EDQL for {area['name']}...")
        
        # Create agent
        agent = EDQLAgent(
            state_dim=4,
            action_dim=8,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            buffer_size=10000,
            batch_size=32
        )
        
        # Train agent with persistence
        csv_filename = train_edql_delhi(
            env_class=area['env_class'],
            env_args=area['env_args'],
            agent=agent,
            area_name=area['name'],
            episodes=500,  # Reduced for faster training
            max_steps=1000,
            target_update_freq=50,
            log_freq=10,
            save_freq=100,
            auto_load=True  # Enable auto-loading
        )
        
        training_logs[area['name']] = csv_filename
        
        print(f"✅ {area['name']} training completed!")
    
    print("\n🎯 All Delhi networks trained with TD error logging and model persistence!")
    print("📊 Training logs:")
    for area, log_file in training_logs.items():
        print(f"   - {area}: {log_file}")
    print("📈 Ready for graph generation phase!")

if __name__ == "__main__":
    main() 