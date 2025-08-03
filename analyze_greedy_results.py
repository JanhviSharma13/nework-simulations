import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import os

class GreedyResultsAnalyzer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load the CSV data"""
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(self.df)} records from {self.csv_file}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        return True
    
    def plot_travel_time_vs_episode(self):
        """1. Travel Time per Episode (Line Plot)"""
        plt.figure(figsize=(12, 6))
        
        # Group by area and plot
        for area in self.df['map_name'].unique():
            area_data = self.df[self.df['map_name'] == area]
            plt.plot(area_data.index, area_data['total_time'], 
                    marker='o', label=area, linewidth=2, markersize=6)
        
        # Add moving average
        if len(self.df) > 25:
            window = min(25, len(self.df) // 4)
            moving_avg = self.df['total_time'].rolling(window=window, center=True).mean()
            plt.plot(self.df.index, moving_avg, 'k--', alpha=0.7, linewidth=2, label=f'Moving Avg (window={window})')
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Total Travel Time (seconds)', fontsize=12)
        plt.title('Travel Time vs Episode - Greedy Algorithm', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('greedy_travel_time_vs_episode.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: greedy_travel_time_vs_episode.png")
    
    def plot_arrival_success_rate(self):
        """2. Arrival Success Rate over Episodes"""
        plt.figure(figsize=(12, 6))
        
        # Calculate success rate with rolling window
        window_size = max(3, len(self.df) // 10)
        success_rate = self.df['arrival_success'].rolling(window=window_size, center=True).mean() * 100
        
        plt.plot(self.df.index, success_rate, 'b-', linewidth=2, marker='o', markersize=4)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='0% Success Rate')
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Success Rate (%)', fontsize=12)
        plt.title('Arrival Success Rate over Episodes - Greedy Algorithm', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-5, 105)
        plt.tight_layout()
        plt.savefig('greedy_success_rate_vs_episode.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: greedy_success_rate_vs_episode.png")
    
    def plot_route_length_vs_episode(self):
        """3. Route Length vs Episode"""
        plt.figure(figsize=(12, 6))
        
        # Group by area and plot
        for area in self.df['map_name'].unique():
            area_data = self.df[self.df['map_name'] == area]
            plt.plot(area_data.index, area_data['route_length'], 
                    marker='s', label=area, linewidth=2, markersize=6)
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Route Length (edges)', fontsize=12)
        plt.title('Route Length vs Episode - Greedy Algorithm', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('greedy_route_length_vs_episode.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: greedy_route_length_vs_episode.png")
    
    def plot_avg_travel_time_by_area(self):
        """4. Average Travel Time by Area (Bar Chart)"""
        plt.figure(figsize=(10, 6))
        
        area_stats = self.df.groupby('map_name')['total_time'].agg(['mean', 'std']).reset_index()
        
        bars = plt.bar(area_stats['map_name'], area_stats['mean'], 
                      yerr=area_stats['std'], capsize=5, alpha=0.7)
        
        # Color bars by area
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xlabel('Area', fontsize=12)
        plt.ylabel('Average Travel Time (seconds)', fontsize=12)
        plt.title('Average Travel Time by Area - Greedy Algorithm', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('greedy_avg_travel_time_by_area.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: greedy_avg_travel_time_by_area.png")
    
    def plot_success_rate_by_area(self):
        """5. Success Rate by Area (Bar Chart)"""
        plt.figure(figsize=(10, 6))
        
        area_stats = self.df.groupby('map_name')['arrival_success'].agg(['mean', 'count']).reset_index()
        area_stats['success_rate'] = area_stats['mean'] * 100
        
        bars = plt.bar(area_stats['map_name'], area_stats['success_rate'], alpha=0.7)
        
        # Color bars by success rate
        colors = ['red' if rate == 0 else 'green' if rate > 50 else 'orange' for rate in area_stats['success_rate']]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels on bars
        for bar, rate in zip(bars, area_stats['success_rate']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Area', fontsize=12)
        plt.ylabel('Success Rate (%)', fontsize=12)
        plt.title('Success Rate by Area - Greedy Algorithm', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 110)
        plt.tight_layout()
        plt.savefig('greedy_success_rate_by_area.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: greedy_success_rate_by_area.png")
    
    def plot_avg_route_length_by_area(self):
        """6. Average Route Length by Area (Bar Chart)"""
        plt.figure(figsize=(10, 6))
        
        area_stats = self.df.groupby('map_name')['route_length'].agg(['mean', 'std']).reset_index()
        
        bars = plt.bar(area_stats['map_name'], area_stats['mean'], 
                      yerr=area_stats['std'], capsize=5, alpha=0.7)
        
        # Color bars by area
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xlabel('Area', fontsize=12)
        plt.ylabel('Average Route Length (edges)', fontsize=12)
        plt.title('Average Route Length by Area - Greedy Algorithm', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('greedy_avg_route_length_by_area.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: greedy_avg_route_length_by_area.png")
    
    def plot_travel_time_vs_route_length(self):
        """7. Travel Time vs Route Length (Scatter Plot)"""
        plt.figure(figsize=(10, 6))
        
        # Color by area
        colors = {'Rakab Ganj': '#1f77b4', 'Safdarjung': '#ff7f0e', 
                 'CP2': '#2ca02c', 'Chandni Chowk': '#d62728'}
        
        for area in self.df['map_name'].unique():
            area_data = self.df[self.df['map_name'] == area]
            plt.scatter(area_data['route_length'], area_data['total_time'], 
                       c=colors.get(area, '#1f77b4'), label=area, alpha=0.7, s=50)
        
        plt.xlabel('Route Length (edges)', fontsize=12)
        plt.ylabel('Travel Time (seconds)', fontsize=12)
        plt.title('Travel Time vs Route Length - Greedy Algorithm', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('greedy_travel_time_vs_route_length.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: greedy_travel_time_vs_route_length.png")
    
    def plot_travel_time_vs_edges_visited(self):
        """8. Travel Time vs Number of Edges Visited (Scatter Plot)"""
        plt.figure(figsize=(10, 6))
        
        # Color by area
        colors = {'Rakab Ganj': '#1f77b4', 'Safdarjung': '#ff7f0e', 
                 'CP2': '#2ca02c', 'Chandni Chowk': '#d62728'}
        
        for area in self.df['map_name'].unique():
            area_data = self.df[self.df['map_name'] == area]
            plt.scatter(area_data['num_edges_visited'], area_data['total_time'], 
                       c=colors.get(area, '#1f77b4'), label=area, alpha=0.7, s=50)
        
        plt.xlabel('Number of Edges Visited', fontsize=12)
        plt.ylabel('Travel Time (seconds)', fontsize=12)
        plt.title('Travel Time vs Number of Edges Visited - Greedy Algorithm', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('greedy_travel_time_vs_edges_visited.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: greedy_travel_time_vs_edges_visited.png")
    
    def plot_decision_count_analysis(self):
        """9. Decision Count Analysis"""
        plt.figure(figsize=(12, 8))
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Decision count by area
        area_stats = self.df.groupby('map_name')['decision_count'].agg(['mean', 'std']).reset_index()
        bars1 = ax1.bar(area_stats['map_name'], area_stats['mean'], 
                        yerr=area_stats['std'], capsize=5, alpha=0.7)
        ax1.set_title('Average Decision Count by Area')
        ax1.set_ylabel('Decisions')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Decision count vs success
        ax2.scatter(self.df['decision_count'], self.df['arrival_success'], alpha=0.6)
        ax2.set_xlabel('Decision Count')
        ax2.set_ylabel('Success (0/1)')
        ax2.set_title('Decision Count vs Success')
        
        # 3. Decision count distribution
        ax3.hist(self.df['decision_count'], bins=10, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Decision Count')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Decision Count Distribution')
        
        # 4. Decision count vs travel time
        ax4.scatter(self.df['decision_count'], self.df['total_time'], alpha=0.6)
        ax4.set_xlabel('Decision Count')
        ax4.set_ylabel('Travel Time (seconds)')
        ax4.set_title('Decision Count vs Travel Time')
        
        plt.tight_layout()
        plt.savefig('greedy_decision_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: greedy_decision_analysis.png")
    
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        print("\n" + "="*60)
        print("📊 GREEDY ALGORITHM RESULTS SUMMARY")
        print("="*60)
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Total Tests: {len(self.df)}")
        print(f"   Success Rate: {(self.df['arrival_success'].sum() / len(self.df)) * 100:.1f}%")
        print(f"   Avg Travel Time: {self.df['total_time'].mean():.2f}s")
        print(f"   Avg Route Length: {self.df['route_length'].mean():.2f} edges")
        print(f"   Avg Decisions: {self.df['decision_count'].mean():.2f}")
        print(f"   Avg Steps: {self.df['num_steps'].mean():.2f}")
        
        print(f"\n📊 Area-wise Analysis:")
        for area in self.df['map_name'].unique():
            area_df = self.df[self.df['map_name'] == area]
            success_rate = (area_df['arrival_success'].sum() / len(area_df)) * 100
            avg_time = area_df['total_time'].mean()
            avg_decisions = area_df['decision_count'].mean()
            avg_route_length = area_df['route_length'].mean()
            
            print(f"   {area}:")
            print(f"     - Success Rate: {success_rate:.1f}%")
            print(f"     - Avg Travel Time: {avg_time:.2f}s")
            print(f"     - Avg Decisions: {avg_decisions:.2f}")
            print(f"     - Avg Route Length: {avg_route_length:.2f} edges")
        
        print(f"\n🎯 Key Insights:")
        if self.df['arrival_success'].sum() == 0:
            print("   ⚠️  No successful episodes - algorithm may need improvement")
        else:
            print("   ✅ Some successful episodes achieved")
        
        print(f"   📏 Route efficiency: {self.df['route_length'].mean():.2f} edges average")
        print(f"   ⚡ Decision efficiency: {self.df['decision_count'].mean():.2f} decisions average")
    
    def generate_all_plots(self):
        """Generate all requested plots"""
        print("🎨 Generating all plots for Greedy Algorithm Results...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Generate all plots
        self.plot_travel_time_vs_episode()
        self.plot_arrival_success_rate()
        self.plot_route_length_vs_episode()
        self.plot_avg_travel_time_by_area()
        self.plot_success_rate_by_area()
        self.plot_avg_route_length_by_area()
        self.plot_travel_time_vs_route_length()
        self.plot_travel_time_vs_edges_visited()
        self.plot_decision_count_analysis()
        
        # Generate summary
        self.generate_summary_statistics()
        
        print("\n✅ All plots generated successfully!")
        print("📁 Check the current directory for all PNG files")

def main():
    """Main function"""
    # Find the most recent greedy results file
    csv_files = [f for f in os.listdir('.') if f.startswith('greedy_results_') and f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No greedy results CSV files found!")
        return
    
    # Use the most recent file
    latest_file = max(csv_files, key=os.path.getctime)
    print(f"📊 Using results file: {latest_file}")
    
    # Create analyzer and generate plots
    analyzer = GreedyResultsAnalyzer(latest_file)
    analyzer.generate_all_plots()

if __name__ == "__main__":
    main() 