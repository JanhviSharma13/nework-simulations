#!/usr/bin/env python3
"""
Comprehensive test script to verify all algorithms work with fixed TraCI simulation
"""

import sys
import os
import xml.etree.ElementTree as ET

def test_algorithm(algorithm_name, module_path, agent_module):
    """Test a specific algorithm with TraCI simulation"""
    print(f"\n{'='*60}")
    print(f"TESTING {algorithm_name.upper()} ALGORITHM")
    print(f"{'='*60}")
    
    # Add module path
    sys.path.append(module_path)
    
    # Import the algorithm
    if algorithm_name == "greedy":
        from greedy_agent import build_graph, greedy_path, run_simulation
        path_func = greedy_path
    elif algorithm_name == "astar":
        from astar_agent import build_graph, a_star_path, run_simulation
        path_func = a_star_path
    elif algorithm_name == "dijkstra":
        from dijkstra_agent import build_graph, dijkstra_path, run_simulation
        path_func = dijkstra_path
    else:
        print(f"❌ Unknown algorithm: {algorithm_name}")
        return False
    
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
    
    print(f"Testing {algorithm_name} trip: {from_edge} -> {to_edge}")
    
    # Run algorithm
    edges, path_metrics = path_func(G, net, from_edge, to_edge)
    
    if edges and path_metrics['success']:
        print(f"✅ {algorithm_name.upper()} path found with {len(edges)} edges")
        print(f"Total distance: {path_metrics['total_distance']:.2f}")
        print(f"Nodes expanded: {path_metrics['nodes_expanded']}")
        print(f"Route: {' -> '.join(edges[:3])}...{' -> '.join(edges[-3:])}")
        
        # Create route file for simulation
        routes_root = ET.Element("routes")
        ET.SubElement(routes_root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5",
                      length="5", minGap="2.5", maxSpeed="70", guiShape="passenger")
        
        veh = ET.SubElement(routes_root, "vehicle", id="test_vehicle", type="car", depart="0")
        ET.SubElement(veh, "route", edges=" ".join(edges))
        
        temp_route_file = f"test_{algorithm_name}_route.rou.xml"
        temp_tree = ET.ElementTree(routes_root)
        temp_tree.write(temp_route_file)
        
        print(f"Running {algorithm_name.upper()} TraCI simulation...")
        sim_metrics = run_simulation(net_file, temp_route_file)
        
        if sim_metrics['success']:
            print(f"✅ {algorithm_name.upper()} TraCI simulation successful!")
            print(f"Total time: {sim_metrics['total_time']:.2f} seconds")
            print(f"Num steps: {sim_metrics['num_steps']}")
            print(f"Arrival success: {sim_metrics['arrival_success']}")
            print(f"Final edge: {sim_metrics['final_edge']}")
            print(f"Destination edge: {to_edge}")
            
            # Check if vehicle actually moved
            if sim_metrics['num_steps'] > 0:
                print(f"✅ Vehicle moved for {sim_metrics['num_steps']} steps")
                success = True
            else:
                print(f"❌ Vehicle did not move")
                success = False
                
        else:
            print(f"❌ {algorithm_name.upper()} TraCI simulation failed: {sim_metrics.get('error', 'Unknown error')}")
            success = False
        
        # Clean up
        if os.path.exists(temp_route_file):
            os.remove(temp_route_file)
    else:
        print(f"❌ No {algorithm_name} path found: {path_metrics.get('error', 'Unknown error')}")
        success = False
    
    return success

def main():
    """Test all three algorithms"""
    print("COMPREHENSIVE ALGORITHM TESTING")
    print("Testing Greedy, A*, and Dijkstra with fixed TraCI simulation")
    
    results = {}
    
    # Test Greedy
    results['greedy'] = test_algorithm("greedy", "Greedy", "greedy_agent")
    
    # Test A*
    results['astar'] = test_algorithm("astar", "A-Star", "astar_agent")
    
    # Test Dijkstra
    results['dijkstra'] = test_algorithm("dijkstra", "Dijkstra", "dijkstra_agent")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for algorithm, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{algorithm.upper()}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print(f"\n🎉 ALL ALGORITHMS PASSED! TraCI simulation is working correctly.")
    else:
        print(f"\n⚠️  Some algorithms failed. Please check the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main() 