#!/usr/bin/env python3
"""
A* Algorithm Runner for SUMO Networks
Processes each area with comprehensive logging
"""

import os
import sys
import time
from datetime import datetime
from generate_trips import generate_random_trips
from build_astar_routes import generate_routes
import matplotlib.pyplot as plt
import numpy as np

def process_area(area_name, net_file, output_dir):
    """Process a single area with A* algorithm"""
    print(f"\n{'='*60}")
    print(f"PROCESSING {area_name}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths
    trips_file = os.path.join(output_dir, f"{area_name.lower()}_trips.xml")
    routes_file = os.path.join(output_dir, f"{area_name.lower()}_astar_routes.rou.xml")
    
    # Step 1: Generate random trips
    print(f"Step 1: Generating random trips for {area_name}...")
    try:
        generate_random_trips(net_file, trips_file)
    except Exception as e:
        print(f"❌ Failed to generate trips for {area_name}: {e}")
        return False
    
    # Step 2: Build A* routes
    print(f"Step 2: Building A* routes for {area_name}...")
    try:
        successful_routes, metrics = generate_routes(net_file, trips_file, routes_file, area_name)
        
        if successful_routes == 0:
            print(f"❌ No successful routes generated for {area_name}")
            return False
        
        print(f"✅ Generated {successful_routes} successful routes for {area_name}")
        
        # Move metrics file to output directory
        import glob
        metrics_files = glob.glob(f"astar_metrics_{area_name.lower()}_*.csv")
        if metrics_files:
            latest_metrics = max(metrics_files, key=os.path.getctime)
            new_metrics_path = os.path.join(output_dir, f"{area_name.lower()}_astar_metrics.csv")
            os.rename(latest_metrics, new_metrics_path)
            print(f"✅ Saved metrics to {new_metrics_path}")
        
        # Step 3: Generate plots
        if metrics:
            print(f"Step 3: Generating plots for {area_name}...")
            generate_plots(metrics, area_name, output_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to build routes for {area_name}: {e}")
        return False

def generate_plots(metrics, area_name, output_dir):
    """Generate plots from metrics data"""
    if not metrics:
        return
    
    # Convert to DataFrame for easier plotting
    import pandas as pd
    df = pd.DataFrame(metrics)
    
    # Plot 1: Trip Time vs Distance
    plt.figure(figsize=(10, 6))
    plt.scatter(df['total_distance'], df['total_time'], color='blue', alpha=0.6)
    
    # Add trend line
    if len(df) > 1:
        z = np.polyfit(df['total_distance'], df['total_time'], 1)
        p = np.poly1d(z)
        plt.plot(df['total_distance'], p(df['total_distance']), "r--", alpha=0.8)
    
    plt.xlabel("Trip Distance (meters)")
    plt.ylabel("Trip Time (seconds)")
    plt.title(f"Trip Time vs Distance - {area_name} (A*)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f"{area_name.lower()}_trip_scatter.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved scatter plot: {plot_file}")
    
    # Plot 2: A* Performance Metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Nodes expanded vs route length
    ax1.scatter(df['route_length'], df['total_nodes_expanded'], alpha=0.6)
    ax1.set_xlabel('Route Length (edges)')
    ax1.set_ylabel('Nodes Expanded')
    ax1.set_title('Search Efficiency')
    ax1.grid(True, alpha=0.3)
    
    # Search depth vs max frontier
    ax2.scatter(df['search_depth'], df['max_frontier_size'], alpha=0.6)
    ax2.set_xlabel('Search Depth')
    ax2.set_ylabel('Max Frontier Size')
    ax2.set_title('Search Complexity')
    ax2.grid(True, alpha=0.3)
    
    # Final costs
    ax3.scatter(df['final_heuristic_cost'], df['final_total_cost'], alpha=0.6)
    ax3.set_xlabel('Final Heuristic Cost')
    ax3.set_ylabel('Final Total Cost')
    ax3.set_title('Cost Analysis')
    ax3.grid(True, alpha=0.3)
    
    # Edges visited
    ax4.hist(df['num_edges_visited'], bins=20, alpha=0.7)
    ax4.set_xlabel('Edges Visited')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Edge Coverage Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    performance_file = os.path.join(output_dir, f"{area_name.lower()}_performance_metrics.png")
    plt.savefig(performance_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved performance metrics: {performance_file}")

def main():
    """Main execution function"""
    print("🚀 A* ALGORITHM RUNNER")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define areas and their network files
    areas = {
        'CP2': {
            'net_file': 'CP2/cp2.net.xml',
            'output_dir': 'A-Star/CP2'
        },
        'RakabGanj': {
            'net_file': 'Rakab-Ganj/rg.net.xml',
            'output_dir': 'A-Star/RakabGanj'
        },
        'Safdarjung': {
            'net_file': 'Safdarjung/SE.net.xml',
            'output_dir': 'A-Star/Safdarjung'
        },
        'ChandniChowk': {
            'net_file': 'Chandni-Chowk/ch.net.xml',
            'output_dir': 'A-Star/ChandniChowk'
        }
    }
    
    total_start_time = time.time()
    results = {}
    
    for area_name, config in areas.items():
        if os.path.exists(config['net_file']):
            success = process_area(area_name, config['net_file'], config['output_dir'])
            results[area_name] = success
        else:
            print(f"❌ Network file not found: {config['net_file']}")
            results[area_name] = False
    
    # Summary
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print("A* ALGORITHM RUNNER COMPLETE")
    print(f"{'='*60}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nResults:")
    for area, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {area}: {status}")

if __name__ == "__main__":
    main() 