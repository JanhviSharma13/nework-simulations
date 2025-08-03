import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

class CSVCleaner:
    def __init__(self):
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
    
    def clean_csv_file(self, csv_file, area_name):
        """Clean a single CSV file according to specifications"""
        print(f"🧹 Cleaning {area_name}...")
        
        # Load the CSV
        df = pd.read_csv(csv_file)
        original_episodes = len(df)
        print(f"   Original episodes: {original_episodes}")
        
        # 1. Early Episode Truncation - Drop first 50 episodes
        df = df[df['episode'] >= 50].copy()
        print(f"   After truncation: {len(df)} episodes")
        
        # 2. Reward Outlier Clipping
        df['total_reward_clipped'] = df['total_reward'].clip(lower=-300, upper=300)
        
        # Count clipped values
        clipped_count = len(df[(df['total_reward'] < -300) | (df['total_reward'] > 300)])
        print(f"   Clipped {clipped_count} outlier rewards")
        
        # 3. Rolling Average Smoothing
        window_size = 25  # Adjust based on visual smoothness preference
        
        # Smooth reward
        df['total_reward_smoothed'] = df['total_reward_clipped'].rolling(window=window_size, center=True).mean()
        
        # Smooth TD error
        df['td_mean_smoothed'] = df['td_mean'].rolling(window=window_size, center=True).mean()
        
        # Smooth loss
        df['loss_smoothed'] = df['loss'].rolling(window=window_size, center=True).mean()
        
        # Calculate reward per step (smoothed)
        df['reward_per_step'] = df['total_reward_clipped'] / df['steps']
        df['reward_per_step_smoothed'] = df['reward_per_step'].rolling(window=window_size, center=True).mean()
        
        print(f"   Applied rolling average smoothing (window={window_size})")
        
        # Reset episode numbers to start from 0
        df['episode_original'] = df['episode'].copy()
        df['episode'] = df['episode'] - df['episode'].min()
        
        print(f"   Final cleaned episodes: {len(df)}")
        
        return df
    
    def clean_all_csv_files(self):
        """Clean all CSV files and save to Cleaned CSV folder"""
        print("🎯 Starting CSV Cleaning Process")
        print("=" * 50)
        
        # Find training logs
        log_files = self.find_training_logs()
        
        if not log_files:
            print("❌ No training logs found.")
            return
        
        cleaned_files = {}
        
        for area, csv_file in log_files.items():
            area_name = self.area_names[self.areas.index(area)]
            
            # Clean the CSV
            cleaned_df = self.clean_csv_file(csv_file, area_name)
            
            # Save cleaned CSV
            output_filename = f"Cleaned CSV/cleaned_{area}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            cleaned_df.to_csv(output_filename, index=False)
            print(f"   ✅ Saved cleaned CSV: {output_filename}")
            
            cleaned_files[area] = cleaned_df
        
        print(f"\n✅ All CSV files cleaned and saved to 'Cleaned CSV' folder!")
        print(f"📁 Cleaned files:")
        for area, df in cleaned_files.items():
            print(f"   - {area}: {len(df)} episodes")
        
        return cleaned_files

def main():
    """Main function to clean all CSV files"""
    cleaner = CSVCleaner()
    cleaned_files = cleaner.clean_all_csv_files()
    
    if cleaned_files:
        print("\n📊 Summary of cleaning process:")
        for area, df in cleaned_files.items():
            area_name = cleaner.area_names[cleaner.areas.index(area)]
            print(f"   {area_name}:")
            print(f"     - Episodes: {len(df)}")
            print(f"     - Reward range: {df['total_reward_clipped'].min():.1f} to {df['total_reward_clipped'].max():.1f}")
            print(f"     - Mean reward: {df['total_reward_clipped'].mean():.1f}")

if __name__ == "__main__":
    main() 