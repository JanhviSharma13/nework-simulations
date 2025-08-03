#!/usr/bin/env python3
"""
Test script to verify the fixed greedy simulation
"""

import sys
import os
sys.path.append('Greedy')

from greedy_agent import build_graph, greedy_path, run_simulation
import xml.etree.ElementTree as ET

def test_single_trip():
    """Test a single trip with greedy algorithm and simulation"""
    
    # Test with CP2 network
    net_file = "CP2/cp2.net.xml"
    trips_file = "A-Star/CP2/cp2_trips.xml"
    
    if not os.path.exists(net_file):
        print(f"Network file not found: {net_file}")
        return
    
    if not os.path.exists(trips_file):
        print(f"Trips file not found: {trips_file}")
        return
    
    print("Building graph...")
    G, net = build_graph(net_file)
    
    print(f"Using full graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Parse first trip
    tree = ET.parse(trips_file)
    root = tree.getroot()
    trips = root.findall("trip")
    
    if not trips:
        print("No trips found in file")
        return
    
    # Use first trip
    trip = trips[0]
    from_edge = trip.attrib["from"]
    to_edge = trip.attrib["to"]
    
    print(f"Testing trip: {from_edge} -> {to_edge}")
    
    # Run greedy algorithm
    edges, path_metrics = greedy_path(G, net, from_edge, to_edge)
    
    if edges and path_metrics['success']:
        print(f"✅ Path found with {len(edges)} edges")
        print(f"Total distance: {path_metrics['total_distance']:.2f}")
        print(f"Nodes expanded: {path_metrics['nodes_expanded']}")
        
        # Create route file for simulation
        routes_root = ET.Element("routes")
        ET.SubElement(routes_root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5",
                      length="5", minGap="2.5", maxSpeed="70", guiShape="passenger")
        
        veh = ET.SubElement(routes_root, "vehicle", id="test_vehicle", type="car", depart="0")
        ET.SubElement(veh, "route", edges=" ".join(edges))
        
        temp_route_file = "test_greedy_route.rou.xml"
        temp_tree = ET.ElementTree(routes_root)
        temp_tree.write(temp_route_file)
        
        print("Running SUMO simulation...")
        sim_metrics = run_simulation(net_file, temp_route_file)
        
        if sim_metrics['success']:
            print(f"✅ Simulation successful!")
            print(f"Total time: {sim_metrics['total_time']:.2f} seconds")
            print(f"Arrival success: {sim_metrics['arrival_success']}")
            print(f"Final edge: {sim_metrics['final_edge']}")
        else:
            print(f"❌ Simulation failed: {sim_metrics.get('error', 'Unknown error')}")
        
        # Clean up
        if os.path.exists(temp_route_file):
            os.remove(temp_route_file)
    else:
        print(f"❌ No path found: {path_metrics.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_single_trip() 