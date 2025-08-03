#!/usr/bin/env python3
"""
Complete Baseline Runner for SUMO Networks
Combines route cleaning + Dijkstra + A* algorithms
"""

import os
import sys
import time
import subprocess
from datetime import datetime

def run_route_cleaning():
    """Run the route cleaning algorithm"""
    print("="*60)
    print("STEP 1: ROUTE CLEANING")
    print("="*60)
    
    try:
        result = subprocess.run([sys.executable, "route_cleaning_algorithm.py"], 
                              capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            print("✅ Route cleaning completed successfully")
            print(result.stdout)
            return True
        else:
            print("❌ Route cleaning failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Route cleaning timed out (30 minutes)")
        return False
    except Exception as e:
        print(f"❌ Route cleaning error: {e}")
        return False

def run_baseline_algorithms():
    """Run the baseline algorithms (Dijkstra + A*)"""
    print("\n" + "="*60)
    print("STEP 2: BASELINE ALGORITHMS")
    print("="*60)
    
    try:
        result = subprocess.run([sys.executable, "dijkstra_astar_baselines.py"], 
                              capture_output=True, text=True, timeout=3600)  # 60 min timeout
        
        if result.returncode == 0:
            print("✅ Baseline algorithms completed successfully")
            print(result.stdout)
            return True
        else:
            print("❌ Baseline algorithms failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Baseline algorithms timed out (60 minutes)")
        return False
    except Exception as e:
        print(f"❌ Baseline algorithms error: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    print("Checking dependencies...")
    
    required_packages = ['pandas', 'networkx', 'xml']
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("Please install: pip install " + " ".join(missing_packages))
        return False
    
    # Check for SUMO
    try:
        import traci
        print("✅ SUMO traci available")
    except ImportError:
        print("⚠️  SUMO traci not available - will use fallback methods")
    
    try:
        import sumolib
        print("✅ SUMO sumolib available")
    except ImportError:
        print("⚠️  SUMO sumolib not available - will use fallback methods")
    
    return True

def main():
    """Main execution function"""
    print("🚀 COMPLETE BASELINE RUNNER")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependencies not met. Exiting.")
        return
    
    total_start_time = time.time()
    
    # Step 1: Route Cleaning
    cleaning_success = run_route_cleaning()
    
    if not cleaning_success:
        print("❌ Route cleaning failed. Stopping execution.")
        return
    
    # Step 2: Baseline Algorithms
    baseline_success = run_baseline_algorithms()
    
    if not baseline_success:
        print("❌ Baseline algorithms failed.")
        return
    
    # Summary
    total_time = time.time() - total_start_time
    print("\n" + "="*60)
    print("🎉 COMPLETE BASELINE RUNNER FINISHED")
    print("="*60)
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List generated files
    print("\n📁 Generated Files:")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    expected_files = [
        "dijkstra_reference_all_areas.csv",
        f"dijkstra_baseline_{timestamp}.csv",
        f"astar_baseline_{timestamp}.csv"
    ]
    
    for file in expected_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (missing)")
    
    # Check for cleaned route files
    cleaned_files = [
        "cp2_dijkstra_cleaned.rou.xml",
        "cp_dijkstra_cleaned.rou.xml", 
        "rakabganj_dijkstra_cleaned.rou.xml",
        "safdarjung_dijkstra_cleaned.rou.xml",
        "chandnichowk_dijkstra_cleaned.rou.xml"
    ]
    
    print("\n🧹 Cleaned Route Files:")
    for file in cleaned_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (missing)")

if __name__ == "__main__":
    main() 