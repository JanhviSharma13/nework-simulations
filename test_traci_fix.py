#!/usr/bin/env python3
"""
Test script to verify the fixed TraCI simulation
"""

import sys
import os
sys.path.append('Greedy')

from greedy_agent import build_graph, greedy_path, run_simulation
import xml.etree.ElementTree as ET

def test_traci_simulation():
    """Test the fixed TraCI simulation with a single trip"""
    
    # Test with CP2 network
    net_file = "CP2/cp2.net.xml"
    trips_file = "A-Star/CP2/cp2_trips.xml"
    
    print("Building graph...")
    G, net = build_graph(net_file)
    
    # Parse first trip
    tree = ET.parse(trips_file)
    root = tree.getroot()
    trips = root.findall("trip")
    
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
        print(f"Route: {' -> '.join(edges[:5])}...{' -> '.join(edges[-5:])}")
        
        # Create route file for simulation
        routes_root = ET.Element("routes")
        ET.SubElement(routes_root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5",
                      length="5", minGap="2.5", maxSpeed="70", guiShape="passenger")
        
        veh = ET.SubElement(routes_root, "vehicle", id="test_vehicle", type="car", depart="0")
        ET.SubElement(veh, "route", edges=" ".join(edges))
        
        temp_route_file = "test_traci_route.rou.xml"
        temp_tree = ET.ElementTree(routes_root)
        temp_tree.write(temp_route_file)
        
        print("Running TraCI simulation...")
        sim_metrics = run_simulation(net_file, temp_route_file)
        
        if sim_metrics['success']:
            print(f"✅ TraCI simulation successful!")
            print(f"Total time: {sim_metrics['total_time']:.2f} seconds")
            print(f"Num steps: {sim_metrics['num_steps']}")
            print(f"Arrival success: {sim_metrics['arrival_success']}")
            print(f"Final edge: {sim_metrics['final_edge']}")
            print(f"Destination edge: {to_edge}")
            
            # Check if vehicle actually moved
            if sim_metrics['num_steps'] > 0:
                print(f"✅ Vehicle moved for {sim_metrics['num_steps']} steps")
            else:
                print(f"❌ Vehicle did not move")
                
        else:
            print(f"❌ TraCI simulation failed: {sim_metrics.get('error', 'Unknown error')}")
        
        # Clean up
        if os.path.exists(temp_route_file):
            os.remove(temp_route_file)
    else:
        print(f"❌ No path found: {path_metrics.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_traci_simulation() 