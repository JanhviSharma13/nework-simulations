#!/usr/bin/env python3
"""
A* Algorithm Route Generator with Real SUMO/TraCI Simulation
Uses randomTrips.py for reproducible trip generation
"""

import os
import sys
import subprocess
import xml.etree.ElementTree as ET
import pandas as pd
import tempfile
import time
from datetime import datetime
from astar_agent import build_graph, a_star_path, run_simulation


def generate_trips_with_randomtrips(net_file: str, output_file: str, num_trips: int = 100) -> bool:
    """
    Generate trips using SUMO's randomTrips.py for reproducibility
    """
    print(f"Generating {num_trips} trips using randomTrips.py...")
    
    # Find randomTrips.py in SUMO tools
    sumo_home = os.environ.get('SUMO_HOME', '')
    if sumo_home:
        randomtrips_path = os.path.join(sumo_home, 'tools', 'randomTrips.py')
    else:
        # Try common locations
        possible_paths = [
            'C:/Program Files (x86)/Eclipse/Sumo/tools/randomTrips.py',
            'C:/Program Files/Eclipse/Sumo/tools/randomTrips.py',
            '/usr/share/sumo/tools/randomTrips.py',
            '/opt/sumo/tools/randomTrips.py'
        ]
        randomtrips_path = None
        for path in possible_paths:
            if os.path.exists(path):
                randomtrips_path = path
                break
    
    if not randomtrips_path or not os.path.exists(randomtrips_path):
        print("❌ randomTrips.py not found. Please set SUMO_HOME environment variable.")
        return False
    
    # Generate trips using randomTrips.py
    cmd = [
        sys.executable, randomtrips_path,
        '--net-file', net_file,
        '--output-trip-file', output_file,
        '--trip-attributes', 'depart="0"',
        '--num-trips', str(num_trips),
        '--min-distance', '100',
        '--max-distance', '5000',
        '--seed', '42'  # Fixed seed for reproducibility
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Generated trips: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate trips: {e}")
        print(f"Error output: {e.stderr}")
        return False


def parse_trips(trips_file: str) -> list:
    """
    Parse trips from XML file
    """
    tree = ET.parse(trips_file)
    root = tree.getroot()
    
    trips = []
    for trip in root.findall("trip"):
        trips.append({
            'from': trip.get('from'),
            'to': trip.get('to'),
            'depart': float(trip.get('depart', '0'))
        })
    
    return trips


def generate_routes(net_file: str, trips_file: str, output_file: str, area_name: str) -> tuple:
    """
    Generate A* routes for all trips with real SUMO simulation
    """
    print(f"Generating A* routes for {area_name}...")
    
    # Build graph
    G, net = build_graph(net_file)
    
    # Parse trips
    trips = parse_trips(trips_file)
    print(f"Processing {len(trips)} trips...")
    
    # Initialize data structures
    routes_root = ET.Element("routes")
    ET.SubElement(routes_root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5",
                  length="5", minGap="2.5", maxSpeed="70", guiShape="passenger")
    
    metrics = []
    successful_routes = 0
    
    for i, trip in enumerate(trips):
        print(f"Processing trip {i+1}/{len(trips)}...")
        
        # Find A* path
        edges, path_metrics = a_star_path(G, net, trip['from'], trip['to'])
        
        # Always log metrics (success and failure)
        if edges and path_metrics['success']:
            # Create vehicle with route
            veh = ET.SubElement(routes_root, "vehicle", id=f"veh{i}", type="car", depart=str(trip['depart']))
            ET.SubElement(veh, "route", edges=" ".join(edges))
            
            successful_routes += 1
            
            # Run simulation for this route - CREATE SINGLE VEHICLE ROUTE FILE
            temp_route_file = f"temp_astar_route_{i}.rou.xml"
            
            # Create a NEW route file with ONLY this vehicle
            sim_routes_root = ET.Element("routes")
            ET.SubElement(sim_routes_root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5",
                          length="5", minGap="2.5", maxSpeed="70", guiShape="passenger")
            
            sim_veh = ET.SubElement(sim_routes_root, "vehicle", id=f"veh{i}", type="car", depart=str(trip['depart']))
            ET.SubElement(sim_veh, "route", edges=" ".join(edges))
            
            # Write temporary route file
            temp_tree = ET.ElementTree(sim_routes_root)
            temp_tree.write(temp_route_file)
            
            # Run simulation
            print(f"Running simulation for trip {i}...")
            sim_metrics = run_simulation(net_file, temp_route_file)
            
            # Clean up temp file
            if os.path.exists(temp_route_file):
                os.remove(temp_route_file)
            
            # Combine path and simulation metrics
            combined_metrics = {
                'episode': i,
                'map_name': area_name,
                'start_edge': trip['from'],
                'goal_edge': trip['to'],
                'success': 1,
                'total_time': sim_metrics.get('total_time', 0),
                'route_length': path_metrics['num_edges'],
                'total_distance': path_metrics['total_distance'],
                'final_edge': sim_metrics.get('final_edge', ''),
                'num_steps': sim_metrics.get('num_steps', 0),
                'arrival_success': sim_metrics.get('arrival_success', False),
                'heuristic_used': path_metrics['heuristic_used'],
                'total_nodes_expanded': path_metrics['nodes_expanded'],
                'max_frontier_size': path_metrics['max_frontier_size'],
                'final_heuristic_cost': path_metrics['final_heuristic_cost'],
                'final_total_cost': path_metrics['final_total_cost'],
                'search_depth': path_metrics['search_depth'],
                'num_edges_visited': path_metrics['edges_visited']
            }
            
        else:
            # Log failed attempt
            combined_metrics = {
                'episode': i,
                'map_name': area_name,
                'start_edge': trip['from'],
                'goal_edge': trip['to'],
                'success': 0,
                'total_time': 0,
                'route_length': 0,
                'total_distance': 0,
                'final_edge': '',
                'num_steps': 0,
                'arrival_success': False,
                'heuristic_used': 'N/A',
                'total_nodes_expanded': path_metrics.get('nodes_expanded', 0),
                'max_frontier_size': path_metrics.get('max_frontier_size', 0),
                'final_heuristic_cost': 0,
                'final_total_cost': 0,
                'search_depth': 0,
                'num_edges_visited': 0,
                'error': path_metrics.get('error', 'Path not found')
            }
            print(f"  Failed to find path for trip {i}: {trip['from']} -> {trip['to']}")
        
        metrics.append(combined_metrics)
    
    # Save routes
    tree = ET.ElementTree(routes_root)
    tree.write(output_file)
    print(f"Saved {successful_routes} successful routes to {output_file}")
    
    # Save metrics
    if metrics:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = f"astar_metrics_{area_name.lower()}_{timestamp}.csv"
        df = pd.DataFrame(metrics)
        df.to_csv(metrics_file, index=False)
        print(f"Saved metrics to {metrics_file}")
    
    return successful_routes, metrics


def main():
    """Main execution for A* algorithm"""
    print("=" * 60)
    print("A* ALGORITHM ROUTE GENERATOR")
    print("=" * 60)
    
    start_time = time.time()
    
    # Define areas to process - use existing trip files
    areas = [
        ("CP2", "../CP2/cp2.net.xml", "../A-Star/CP2/cp2_trips.xml", "CP2/cp2_astar_routes.rou.xml"),
        ("ChandniChowk", "../Chandni-Chowk/ch.net.xml", "../A-Star/ChandniChowk/chandnichowk_trips.xml", "ChandniChowk/chandnichowk_astar_routes.rou.xml"),
        ("RakabGanj", "../Rakab-Ganj/rg.net.xml", "../A-Star/RakabGanj/rakabganj_trips.xml", "RakabGanj/rakabganj_astar_routes.rou.xml"),
        ("Safdarjung", "../Safdarjung/SE.net.xml", "../A-Star/Safdarjung/safdarjung_trips.xml", "Safdarjung/safdarjung_astar_routes.rou.xml")
    ]
    
    results = {}
    
    for area_name, net_file, trips_file, output_file in areas:
        try:
            print(f"\nProcessing {area_name}...")
            
            # Create output directories if they don't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Check if trip file exists
            if not os.path.exists(trips_file):
                print(f"❌ Trip file not found: {trips_file}")
                results[area_name] = f"❌ FAILED: Trip file not found"
                continue
            
            print(f"Using existing trip file: {trips_file}")
            successful_routes, metrics = generate_routes(net_file, trips_file, output_file, area_name)
            results[area_name] = f"✅ Generated {successful_routes} successful routes"
            print(f"✅ Generated {successful_routes} successful routes for {area_name}")
            
        except Exception as e:
            results[area_name] = f"❌ FAILED: {str(e)}"
            print(f"❌ Failed to build routes for {area_name}: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("A* ALGORITHM RUNNER COMPLETE")
    print("=" * 60)
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Results:")
    for area, result in results.items():
        print(f"  {area}: {result}")


if __name__ == "__main__":
    main() 