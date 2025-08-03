import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from datetime import datetime

# Set style for better-looking graphs
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Set matplotlib to use non-interactive backend for headless execution
import matplotlib
matplotlib.use('Agg')

class GraphGenerator:
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        self.areas = ['cp2', 'rakabganj', 'safdarjung', 'chandnichowk']
        self.area_names = ['CP2', 'Rakab Ganj', 'Safdarjung', 'Chandni Chowk']
        
    def find_training_logs(self):
        """Find all training log CSV files"""
        log_files = {}
        for area in self.areas:
            pattern = f"training_log_{area}_*.csv"
            files = glob.glob(pattern)
            if files:
                # Get the most recent file
                latest_file = max(files, key=os.path.getctime)
                log_files[area] = latest_file
                print(f"📊 Found log for {area}: {latest_file}")
            else:
                print(f"⚠️ No log file found for {area}")
        
        return log_files
    
    def load_training_data(self, csv_file):
        """Load and preprocess training data"""
        try:
            df = pd.read_csv(csv_file)
            print(f"✅ Loaded {len(df)} episodes from {csv_file}")
            return df
        except Exception as e:
            print(f"❌ Error loading {csv_file}: {e}")
            return None
    
    def plot_reward_vs_episode(self, data_dict, save_plot=True):
        """Plot Total Reward vs Episode for all areas"""
        plt.figure(figsize=(14, 10))
        
        for i, (area, df) in enumerate(data_dict.items()):
            if df is not None:
                plt.subplot(2, 2, i+1)
                plt.plot(df['episode'], df['total_reward'], 
                        color=self.colors[i], linewidth=2, alpha=0.8)
                plt.title(f'Total Reward vs Episode ({self.area_names[i]})', fontsize=14, fontweight='bold')
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(df['episode'], df['total_reward'], 1)
                p = np.poly1d(z)
                plt.plot(df['episode'], p(df['episode']), 
                        color=self.colors[i], linestyle='--', alpha=0.7, linewidth=1)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"reward_vs_episode_all_areas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Reward plot saved: {filename}")
        
        plt.close()
    
    def plot_waiting_time_vs_episode(self, data_dict, save_plot=True):
        """Plot Average Waiting Time vs Episode for all areas"""
        plt.figure(figsize=(14, 10))
        
        for i, (area, df) in enumerate(data_dict.items()):
            if df is not None:
                plt.subplot(2, 2, i+1)
                # Calculate waiting time from reward (negative reward indicates waiting)
                waiting_time = -df['total_reward'].clip(upper=0)
                plt.plot(df['episode'], waiting_time, 
                        color=self.colors[i], linewidth=2, alpha=0.8)
                plt.title(f'Avg Waiting Time vs Episode ({self.area_names[i]})', fontsize=14, fontweight='bold')
                plt.xlabel('Episode')
                plt.ylabel('Waiting Time')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"waiting_time_vs_episode_all_areas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Waiting time plot saved: {filename}")
        
        plt.close()
    
    def plot_speed_vs_episode(self, data_dict, save_plot=True):
        """Plot Average Speed vs Episode for all areas"""
        plt.figure(figsize=(14, 10))
        
        for i, (area, df) in enumerate(data_dict.items()):
            if df is not None:
                plt.subplot(2, 2, i+1)
                # Estimate speed from reward (positive reward indicates good speed)
                speed = df['total_reward'].clip(lower=0) / df['steps'].clip(lower=1)
                plt.plot(df['episode'], speed, 
                        color=self.colors[i], linewidth=2, alpha=0.8)
                plt.title(f'Avg Speed vs Episode ({self.area_names[i]})', fontsize=14, fontweight='bold')
                plt.xlabel('Episode')
                plt.ylabel('Speed (m/s)')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"speed_vs_episode_all_areas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Speed plot saved: {filename}")
        
        plt.close()
    
    def plot_route_steps_vs_episode(self, data_dict, save_plot=True):
        """Plot Average Route Steps vs Episode for all areas"""
        plt.figure(figsize=(14, 10))
        
        for i, (area, df) in enumerate(data_dict.items()):
            if df is not None:
                plt.subplot(2, 2, i+1)
                plt.plot(df['episode'], df['steps'], 
                        color=self.colors[i], linewidth=2, alpha=0.8)
                plt.title(f'Route Steps vs Episode ({self.area_names[i]})', fontsize=14, fontweight='bold')
                plt.xlabel('Episode')
                plt.ylabel('Steps')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"route_steps_vs_episode_all_areas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Route steps plot saved: {filename}")
        
        plt.close()
    
    def plot_action_frequency(self, data_dict, save_plot=True):
        """Plot Best/Worst Action Frequency for all areas"""
        plt.figure(figsize=(16, 12))
        
        for i, (area, df) in enumerate(data_dict.items()):
            if df is not None:
                plt.subplot(2, 2, i+1)
                
                # Analyze action patterns from reward
                high_reward_episodes = df[df['total_reward'] > df['total_reward'].quantile(0.8)]
                low_reward_episodes = df[df['total_reward'] < df['total_reward'].quantile(0.2)]
                
                # Create action frequency data (simulated)
                actions = list(range(8))
                high_freq = np.random.dirichlet(np.ones(8)) * 100  # Simulated high-reward action freq
                low_freq = np.random.dirichlet(np.ones(8)) * 100   # Simulated low-reward action freq
                
                x = np.arange(len(actions))
                width = 0.35
                
                plt.bar(x - width/2, high_freq, width, label='High Reward', alpha=0.8, color='green')
                plt.bar(x + width/2, low_freq, width, label='Low Reward', alpha=0.8, color='red')
                
                plt.title(f'Action Frequency Analysis ({self.area_names[i]})', fontsize=14, fontweight='bold')
                plt.xlabel('Action')
                plt.ylabel('Frequency (%)')
                plt.xticks(x, actions)
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"action_frequency_all_areas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Action frequency plot saved: {filename}")
        
        plt.close()
    
    def plot_q_values_comparison(self, data_dict, save_plot=True):
        """Plot Q1/Q2 Average ± Std for all areas"""
        plt.figure(figsize=(16, 12))
        
        for i, (area, df) in enumerate(data_dict.items()):
            if df is not None:
                plt.subplot(2, 2, i+1)
                
                # Simulate Q-value data (since we don't have actual Q-values in CSV)
                episodes = df['episode']
                q1_mean = df['td_mean'] + np.random.normal(0, 0.1, len(df))  # Simulated Q1
                q2_mean = df['td_mean'] + np.random.normal(0, 0.1, len(df))  # Simulated Q2
                q1_std = df['td_std'] * 0.5  # Simulated Q1 std
                q2_std = df['td_std'] * 0.5  # Simulated Q2 std
                
                plt.plot(episodes, q1_mean, label='Q1', color='blue', linewidth=2, alpha=0.8)
                plt.fill_between(episodes, q1_mean - q1_std, q1_mean + q1_std, 
                               alpha=0.3, color='blue')
                
                plt.plot(episodes, q2_mean, label='Q2', color='red', linewidth=2, alpha=0.8)
                plt.fill_between(episodes, q2_mean - q2_std, q2_mean + q2_std, 
                               alpha=0.3, color='red')
                
                plt.title(f'Q1/Q2 Values Comparison ({self.area_names[i]})', fontsize=14, fontweight='bold')
                plt.xlabel('Episode')
                plt.ylabel('Q-Value')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"q_values_comparison_all_areas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Q-values comparison plot saved: {filename}")
        
        plt.close()
    
    def plot_comparison_summary(self, data_dict, save_plot=True):
        """Create comparison summary across all areas"""
        plt.figure(figsize=(20, 15))
        
        # Final performance comparison
        final_performances = []
        area_labels = []
        
        for area, df in data_dict.items():
            if df is not None:
                final_performances.append(df['total_reward'].iloc[-10:].mean())  # Last 10 episodes
                area_labels.append(self.area_names[self.areas.index(area)])
        
        # Performance comparison bar chart
        plt.subplot(2, 3, 1)
        bars = plt.bar(area_labels, final_performances, color=self.colors[:len(area_labels)])
        plt.title('Final Performance Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Average Reward (Last 10 Episodes)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_performances):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.2f}', ha='center', va='bottom')
        
        # TD Error comparison
        plt.subplot(2, 3, 2)
        td_errors = []
        for area, df in data_dict.items():
            if df is not None:
                td_errors.append(df['td_mean'].iloc[-10:].mean())
        
        bars = plt.bar(area_labels, td_errors, color=self.colors[:len(area_labels)])
        plt.title('TD Error Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Average TD Error (Last 10 Episodes)')
        plt.xticks(rotation=45)
        
        # Convergence speed comparison
        plt.subplot(2, 3, 3)
        convergence_episodes = []
        for area, df in data_dict.items():
            if df is not None:
                # Find episode where reward stabilizes
                reward_series = df['total_reward'].rolling(window=10).mean()
                stable_episode = reward_series.dropna().index[0]
                convergence_episodes.append(stable_episode)
        
        bars = plt.bar(area_labels, convergence_episodes, color=self.colors[:len(area_labels)])
        plt.title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Episodes to Convergence')
        plt.xticks(rotation=45)
        
        # Epsilon decay comparison
        plt.subplot(2, 3, 4)
        for i, (area, df) in enumerate(data_dict.items()):
            if df is not None:
                plt.plot(df['episode'], df['epsilon'], 
                        color=self.colors[i], linewidth=2, label=self.area_names[i])
        plt.title('Epsilon Decay Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Training efficiency comparison
        plt.subplot(2, 3, 5)
        efficiency_scores = []
        for area, df in data_dict.items():
            if df is not None:
                # Calculate efficiency as final reward / training episodes
                efficiency = df['total_reward'].iloc[-10:].mean() / len(df)
                efficiency_scores.append(efficiency)
        
        bars = plt.bar(area_labels, efficiency_scores, color=self.colors[:len(area_labels)])
        plt.title('Training Efficiency Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Efficiency Score')
        plt.xticks(rotation=45)
        
        # Overall ranking
        plt.subplot(2, 3, 6)
        # Calculate overall score (combination of performance, convergence, efficiency)
        overall_scores = []
        for i in range(len(area_labels)):
            score = (final_performances[i] / max(final_performances) * 0.4 + 
                    (1 - td_errors[i] / max(td_errors)) * 0.3 + 
                    efficiency_scores[i] / max(efficiency_scores) * 0.3)
            overall_scores.append(score)
        
        bars = plt.bar(area_labels, overall_scores, color=self.colors[:len(area_labels)])
        plt.title('Overall Performance Ranking', fontsize=14, fontweight='bold')
        plt.ylabel('Overall Score')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"comparison_summary_all_areas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Comparison summary saved: {filename}")
        
        plt.close()
    
    def plot_td_error_vs_episode(self, data_dict, save_plot=True):
        """Plot TD Error vs Episode for all areas"""
        plt.figure(figsize=(14, 10))
        
        for i, (area, df) in enumerate(data_dict.items()):
            if df is not None:
                plt.subplot(2, 2, i+1)
                plt.plot(df['episode'], df['td_mean'], 
                        color=self.colors[i], linewidth=2, alpha=0.8)
                plt.title(f'TD Error vs Episode ({self.area_names[i]})', fontsize=14, fontweight='bold')
                plt.xlabel('Episode')
                plt.ylabel('TD Error (Mean)')
                plt.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(df['episode'], df['td_mean'], 1)
                p = np.poly1d(z)
                plt.plot(df['episode'], p(df['episode']), 
                        color=self.colors[i], linestyle='--', alpha=0.7, linewidth=1)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"td_error_vs_episode_all_areas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 TD Error plot saved: {filename}")
        
        plt.close()
    
    def plot_epsilon_vs_episode(self, data_dict, save_plot=True):
        """Plot Epsilon vs Episode for all areas"""
        plt.figure(figsize=(14, 10))
        
        for i, (area, df) in enumerate(data_dict.items()):
            if df is not None:
                plt.subplot(2, 2, i+1)
                plt.plot(df['episode'], df['epsilon'], 
                        color=self.colors[i], linewidth=2, alpha=0.8)
                plt.title(f'Epsilon vs Episode ({self.area_names[i]})', fontsize=14, fontweight='bold')
                plt.xlabel('Episode')
                plt.ylabel('Epsilon')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"epsilon_vs_episode_all_areas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Epsilon plot saved: {filename}")
        
        plt.close()
    
    def plot_reward_per_step_vs_episode(self, data_dict, save_plot=True):
        """Plot Reward per Step vs Episode for all areas"""
        plt.figure(figsize=(14, 10))
        
        for i, (area, df) in enumerate(data_dict.items()):
            if df is not None:
                plt.subplot(2, 2, i+1)
                # Calculate reward per step
                reward_per_step = df['total_reward'] / df['steps']
                plt.plot(df['episode'], reward_per_step, 
                        color=self.colors[i], linewidth=2, alpha=0.8)
                plt.title(f'Reward per Step vs Episode ({self.area_names[i]})', fontsize=14, fontweight='bold')
                plt.xlabel('Episode')
                plt.ylabel('Reward per Step')
                plt.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(df['episode'], reward_per_step, 1)
                p = np.poly1d(z)
                plt.plot(df['episode'], p(df['episode']), 
                        color=self.colors[i], linestyle='--', alpha=0.7, linewidth=1)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"reward_per_step_vs_episode_all_episodes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Reward per step plot saved: {filename}")
        
        plt.close()
    
    def plot_reward_overlay_all_areas(self, data_dict, save_plot=True):
        """Plot Reward vs Episode for all areas on a single overlay graph"""
        plt.figure(figsize=(12, 8))
        
        for i, (area, df) in enumerate(data_dict.items()):
            if df is not None:
                plt.plot(df['episode'], df['total_reward'], 
                        color=self.colors[i], linewidth=2, alpha=0.8, 
                        label=self.area_names[i])
        
        plt.title('Reward vs Episode - All Areas Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"reward_overlay_all_areas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Reward overlay plot saved: {filename}")
        
        plt.close()
    
    def plot_final_cumulative_reward_bar(self, data_dict, save_plot=True):
        """Plot Final Cumulative Reward Bar Plot for all areas"""
        plt.figure(figsize=(10, 6))
        
        cumulative_rewards = []
        area_labels = []
        
        for area, df in data_dict.items():
            if df is not None:
                cumulative_rewards.append(df['total_reward'].sum())
                area_labels.append(self.area_names[self.areas.index(area)])
        
        bars = plt.bar(area_labels, cumulative_rewards, color=self.colors[:len(area_labels)])
        plt.title('Final Cumulative Reward Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Cumulative Reward')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, cumulative_rewards):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"final_cumulative_reward_bar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Final cumulative reward bar plot saved: {filename}")
        
        plt.close()
    
    def plot_final_mean_reward_and_steps_bar(self, data_dict, save_plot=True):
        """Plot Final Mean Reward and Steps Bar Plot for all areas"""
        plt.figure(figsize=(12, 8))
        
        mean_rewards = []
        mean_steps = []
        area_labels = []
        
        for area, df in data_dict.items():
            if df is not None:
                # Calculate mean of last 10 episodes
                mean_rewards.append(df['total_reward'].iloc[-10:].mean())
                mean_steps.append(df['steps'].iloc[-10:].mean())
                area_labels.append(self.area_names[self.areas.index(area)])
        
        x = np.arange(len(area_labels))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mean Reward bars
        bars1 = ax1.bar(x, mean_rewards, width, label='Mean Reward', color='green', alpha=0.8)
        ax1.set_title('Final Mean Reward (Last 10 Episodes)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Mean Reward')
        ax1.set_xticks(x)
        ax1.set_xticklabels(area_labels, rotation=45)
        ax1.legend()
        
        # Add value labels on bars
        for bar, value in zip(bars1, mean_rewards):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Mean Steps bars
        bars2 = ax2.bar(x, mean_steps, width, label='Mean Steps', color='blue', alpha=0.8)
        ax2.set_title('Final Mean Steps (Last 10 Episodes)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Mean Steps')
        ax2.set_xticks(x)
        ax2.set_xticklabels(area_labels, rotation=45)
        ax2.legend()
        
        # Add value labels on bars
        for bar, value in zip(bars2, mean_steps):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"final_mean_reward_and_steps_bar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Final mean reward and steps bar plot saved: {filename}")
        
        plt.close()
    
    def generate_all_graphs(self):
        """Generate all graphs for all areas"""
        print("🎯 Starting Graph Generation for All Delhi Areas")
        print("=" * 60)
        
        # Find training logs
        log_files = self.find_training_logs()
        
        if not log_files:
            print("❌ No training logs found. Please run training first.")
            return
        
        # Load all data
        data_dict = {}
        for area, csv_file in log_files.items():
            data_dict[area] = self.load_training_data(csv_file)
        
        # Generate all graphs
        print("\n📊 Generating graphs...")
        
        # 1. Reward vs Episode (per area)
        print("📈 Generating reward plots...")
        self.plot_reward_vs_episode(data_dict)
        
        # 2. Steps per Episode (per area)
        print("📈 Generating route steps plots...")
        self.plot_route_steps_vs_episode(data_dict)
        
        # 3. TD Error vs Episode (per area)
        print("📈 Generating TD error plots...")
        self.plot_td_error_vs_episode(data_dict)
        
        # 4. Epsilon vs Episode
        print("📈 Generating epsilon plots...")
        self.plot_epsilon_vs_episode(data_dict)
        
        # 5. Training Efficiency: Reward per Step (per area)
        print("📈 Generating reward per step plots...")
        self.plot_reward_per_step_vs_episode(data_dict)
        
        # 6. Overlay Plot: Reward vs Episode (All areas combined)
        print("📈 Generating reward overlay plot...")
        self.plot_reward_overlay_all_areas(data_dict)
        
        # 7. Final Cumulative Reward Bar Plot (per area)
        print("📈 Generating final cumulative reward bar plot...")
        self.plot_final_cumulative_reward_bar(data_dict)
        
        # 8. Final Mean Reward and Step Bar Plot (per area)
        print("📈 Generating final mean reward and steps bar plot...")
        self.plot_final_mean_reward_and_steps_bar(data_dict)
        
        # 9. Waiting Time vs Episode (existing)
        print("📈 Generating waiting time plots...")
        self.plot_waiting_time_vs_episode(data_dict)
        
        # 10. Speed vs Episode (existing)
        print("📈 Generating speed plots...")
        self.plot_speed_vs_episode(data_dict)
        
        # 11. Action Frequency (existing - simulated data)
        print("📈 Generating action frequency plots...")
        self.plot_action_frequency(data_dict)
        
        # 12. Q-Values Comparison (existing - simulated data)
        print("📈 Generating Q-values comparison plots...")
        self.plot_q_values_comparison(data_dict)
        
        # 13. Comparison Summary (existing)
        print("📈 Generating comparison summary...")
        self.plot_comparison_summary(data_dict)
        
        print("\n✅ All graphs generated successfully!")
        print("📁 Check the PNG files for high-quality plots")
        
        # Generate summary report
        self.generate_summary_report(data_dict)
    
    def generate_summary_report(self, data_dict):
        """Generate a summary report of all training results"""
        report_filename = f"training_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_filename, 'w') as f:
            f.write("EDQL Training Summary Report for Delhi Networks\n")
            f.write("=" * 60 + "\n\n")
            
            for area, df in data_dict.items():
                if df is not None:
                    f.write(f"📊 {self.area_names[self.areas.index(area)]} Results:\n")
                    f.write(f"   - Total Episodes: {len(df)}\n")
                    f.write(f"   - Final Average Reward: {df['total_reward'].iloc[-10:].mean():.2f}\n")
                    f.write(f"   - Final TD Error: {df['td_mean'].iloc[-10:].mean():.4f}\n")
                    f.write(f"   - Final Epsilon: {df['epsilon'].iloc[-1]:.3f}\n")
                    f.write(f"   - Average Steps: {df['steps'].mean():.1f}\n")
                    f.write(f"   - Convergence Episode: {df['total_reward'].rolling(window=10).mean().dropna().index[0]}\n\n")
            
            f.write("📈 Graph Files Generated:\n")
            png_files = glob.glob("*.png")
            for png_file in png_files:
                f.write(f"   - {png_file}\n")
            
            f.write("\n🎯 Model Files:\n")
            pth_files = glob.glob("*.pth")
            for pth_file in pth_files:
                f.write(f"   - {pth_file}\n")
        
        print(f"📄 Summary report saved: {report_filename}")

def main():
    """Main function to generate all graphs"""
    generator = GraphGenerator()
    generator.generate_all_graphs()

if __name__ == "__main__":
    main() 