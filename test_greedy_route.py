#!/usr/bin/env python3
"""
Test script to verify the greedy route works with SUMO
"""

import sys
import os
sys.path.append('Greedy')

from greedy_agent import build_graph, greedy_path
import xml.etree.ElementTree as ET

def test_greedy_route():
    """Test a greedy route with SUMO"""
    
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
        print(f"Route: {' '.join(edges[:10])}...")  # Show first 10 edges
        
        # Create route file for simulation
        routes_root = ET.Element("routes")
        ET.SubElement(routes_root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5",
                      length="5", minGap="2.5", maxSpeed="70", guiShape="passenger")
        
        veh = ET.SubElement(routes_root, "vehicle", id="test_vehicle", type="car", depart="0")
        ET.SubElement(veh, "route", edges=" ".join(edges))
        
        temp_route_file = "test_greedy_route.rou.xml"
        temp_tree = ET.ElementTree(routes_root)
        temp_tree.write(temp_route_file)
        
        print(f"Created route file: {temp_route_file}")
        
        # Test with SUMO command line
        import subprocess
        sumo_cmd = [
            'sumo', '--net-file', net_file,
            '--route-files', temp_route_file,
            '--no-step-log', 'true',
            '--no-warnings', 'true',
            '--random', 'false',
            '--time-to-teleport', '300',
            '--ignore-route-errors', 'true',
            '--collision.action', 'none',
            '--tripinfo-output', 'tripinfo.xml'
        ]
        
        print(f"Running SUMO command: {' '.join(sumo_cmd)}")
        result = subprocess.run(sumo_cmd, capture_output=True, text=True)
        
        print(f"SUMO exit code: {result.returncode}")
        if result.stdout:
            print(f"SUMO stdout: {result.stdout}")
        if result.stderr:
            print(f"SUMO stderr: {result.stderr}")
        
        # Check if tripinfo was created
        if os.path.exists('tripinfo.xml'):
            print("✅ Tripinfo file created - route is valid!")
            os.remove('tripinfo.xml')
        else:
            print("❌ No tripinfo file - route may be invalid")
        
        # Clean up
        if os.path.exists(temp_route_file):
            os.remove(temp_route_file)
    else:
        print(f"❌ No path found: {path_metrics.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_greedy_route() 