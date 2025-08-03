#!/usr/bin/env python3
"""
Dijkstra Algorithm Runner for SUMO Networks
Processes each area with comprehensive logging
"""

import os
import sys
import time
from datetime import datetime
from build_dijkstra_routes import generate_routes
import matplotlib.pyplot as plt
import numpy as np

def process_area(area_name, net_file, trips_file, output_dir):
    """Process a single area with Dijkstra algorithm"""
    print(f"\n{'='*60}")
    print(f"PROCESSING {area_name}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths
    routes_file = os.path.join(output_dir, f"{area_name.lower()}_dijkstra_routes.rou.xml")
    
    # Step 1: Build Dijkstra routes
    print(f"Step 1: Building Dijkstra routes for {area_name}...")
    try:
        successful_routes, metrics = generate_routes(net_file, trips_file, routes_file, area_name)
        
        if successful_routes == 0:
            print(f"❌ No successful routes generated for {area_name}")
            return False
        
        print(f"✅ Generated {successful_routes} successful routes for {area_name}")
        
        # Move metrics file to output directory
        import glob
        metrics_files = glob.glob(f"dijkstra_metrics_{area_name.lower()}_*.csv")
        if metrics_files:
            latest_metrics = max(metrics_files, key=os.path.getctime)
            new_metrics_path = os.path.join(output_dir, f"{area_name.lower()}_dijkstra_metrics.csv")
            os.rename(latest_metrics, new_metrics_path)
            print(f"✅ Saved metrics to {new_metrics_path}")
        
        # Step 2: Generate plots
        if metrics:
            print(f"Step 2: Generating plots for {area_name}...")
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
    plt.scatter(df['total_distance'], df['total_time'], color='red', alpha=0.6)
    
    # Add trend line
    if len(df) > 1:
        z = np.polyfit(df['total_distance'], df['total_time'], 1)
        p = np.poly1d(z)
        plt.plot(df['total_distance'], p(df['total_distance']), "b--", alpha=0.8)
    
    plt.xlabel("Trip Distance (meters)")
    plt.ylabel("Trip Time (seconds)")
    plt.title(f"Trip Time vs Distance - {area_name} (Dijkstra)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f"{area_name.lower()}_trip_scatter.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved scatter plot: {plot_file}")
    
    # Plot 2: Dijkstra Performance Metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Nodes expanded vs route length
    ax1.scatter(df['route_length'], df['nodes_expanded'], alpha=0.6, color='red')
    ax1.set_xlabel('Route Length (edges)')
    ax1.set_ylabel('Nodes Expanded')
    ax1.set_title('Search Efficiency')
    ax1.grid(True, alpha=0.3)
    
    # Search depth vs nodes expanded
    ax2.scatter(df['search_depth'], df['nodes_expanded'], alpha=0.6, color='red')
    ax2.set_xlabel('Search Depth')
    ax2.set_ylabel('Nodes Expanded')
    ax2.set_title('Search Complexity')
    ax2.grid(True, alpha=0.3)
    
    # Route length distribution
    ax3.hist(df['route_length'], bins=20, alpha=0.7, color='red')
    ax3.set_xlabel('Route Length (edges)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Route Length Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Distance distribution
    ax4.hist(df['total_distance'], bins=20, alpha=0.7, color='red')
    ax4.set_xlabel('Total Distance (meters)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distance Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    performance_file = os.path.join(output_dir, f"{area_name.lower()}_performance_metrics.png")
    plt.savefig(performance_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved performance metrics: {performance_file}")

def main():
    """Main execution function"""
    print("🚀 DIJKSTRA ALGORITHM RUNNER")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define areas and their files - using A* generated trips
    areas = {
        'CP2': {
            'net_file': 'CP2/cp2.net.xml',
            'trips_file': 'A-Star/CP2/cp2_trips.xml',
            'output_dir': 'Dijkstra/CP2'
        },
        'RakabGanj': {
            'net_file': 'Rakab-Ganj/rg.net.xml',
            'trips_file': 'A-Star/RakabGanj/rakabganj_trips.xml',
            'output_dir': 'Dijkstra/RakabGanj'
        },
        'Safdarjung': {
            'net_file': 'Safdarjung/SE.net.xml',
            'trips_file': 'A-Star/Safdarjung/safdarjung_trips.xml',
            'output_dir': 'Dijkstra/Safdarjung'
        },
        'ChandniChowk': {
            'net_file': 'Chandni-Chowk/ch.net.xml',
            'trips_file': 'A-Star/ChandniChowk/chandnichowk_trips.xml',
            'output_dir': 'Dijkstra/ChandniChowk'
        }
    }
    
    total_start_time = time.time()
    results = {}
    
    for area_name, config in areas.items():
        if os.path.exists(config['net_file']) and os.path.exists(config['trips_file']):
            success = process_area(area_name, config['net_file'], config['trips_file'], config['output_dir'])
            results[area_name] = success
        else:
            print(f"❌ Files not found for {area_name}:")
            print(f"  Network: {config['net_file']} - {'Exists' if os.path.exists(config['net_file']) else 'Missing'}")
            print(f"  Trips: {config['trips_file']} - {'Exists' if os.path.exists(config['trips_file']) else 'Missing'}")
            results[area_name] = False
    
    # Summary
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print("DIJKSTRA ALGORITHM RUNNER COMPLETE")
    print(f"{'='*60}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nResults:")
    for area, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {area}: {status}")

if __name__ == "__main__":
    main() 