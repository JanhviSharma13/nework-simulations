#!/usr/bin/env python3
"""
A* Algorithm Implementation with Real SUMO/TraCI Simulation
"""

import networkx as nx
import sumolib
import traci
import tempfile
import xml.etree.ElementTree as ET
import os
import time
from typing import List, Dict, Tuple, Optional
import heapq
import math


def build_graph(net_file: str) -> Tuple[nx.DiGraph, sumolib.net.Net]:
    """
    Build NetworkX graph from SUMO network file
    """
    print(f"Building graph from {net_file}...")
    
    # Load SUMO network
    net = sumolib.net.readNet(net_file)
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add edges to graph
    for edge in net.getEdges():
        edge_id = edge.getID()
        from_node = edge.getFromNode().getID()
        to_node = edge.getToNode().getID()
        length = edge.getLength()
        
        G.add_edge(from_node, to_node, 
                   edge_id=edge_id, 
                   length=length,
                   weight=length)  # Use length as weight
    
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G, net


def heuristic_distance(G: nx.DiGraph, current: str, goal: str) -> float:
    """
    Calculate heuristic distance between current and goal nodes
    Uses Euclidean distance as admissible heuristic
    """
    try:
        # Get node coordinates from the graph
        current_coords = G.nodes[current].get('coords', (0, 0))
        goal_coords = G.nodes[goal].get('coords', (0, 0))
        
        # Calculate Euclidean distance
        dx = current_coords[0] - goal_coords[0]
        dy = current_coords[1] - goal_coords[1]
        return math.sqrt(dx*dx + dy*dy)
    except:
        # Fallback: return 0 if coordinates not available
        return 0.0


def a_star_path(G: nx.DiGraph, net: sumolib.net.Net, start_edge: str, goal_edge: str) -> Tuple[List[str], Dict]:
    """
    A* pathfinding algorithm implementation
    """
    print(f"Finding A* path from {start_edge} to {goal_edge}")
    
    # Get start and goal nodes from edges
    start_node = net.getEdge(start_edge).getFromNode().getID()
    goal_node = net.getEdge(goal_edge).getToNode().getID()
    
    # Initialize data structures
    open_set = [(0, start_node)]  # (f_score, node)
    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic_distance(G, start_node, goal_node)}
    closed_set = set()
    
    nodes_expanded = 0
    max_frontier_size = 1
    search_depth = 0
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        nodes_expanded += 1
        
        # Update search depth
        current_depth = len([current])
        temp = current
        while temp in came_from:
            temp = came_from[temp]
            current_depth += 1
        search_depth = max(search_depth, current_depth)
        
        # Check if we reached the goal
        if current == goal_node:
            # Reconstruct path
            path_nodes = []
            temp = current
            while temp in came_from:
                path_nodes.append(temp)
                temp = came_from[temp]
            path_nodes.append(start_node)
            path_nodes.reverse()
            
            # Convert node path to edge path
            edge_path = []
            total_distance = 0
            edges_visited = 0
            
            for i in range(len(path_nodes) - 1):
                from_node = path_nodes[i]
                to_node = path_nodes[i + 1]
                
                # Find edge between these nodes
                edge_data = G.get_edge_data(from_node, to_node)
                if edge_data:
                    edge_id = edge_data['edge_id']
                    edge_path.append(edge_id)
                    total_distance += edge_data['length']
                    edges_visited += 1
            
            # Add goal edge
            edge_path.append(goal_edge)
            total_distance += net.getEdge(goal_edge).getLength()
            edges_visited += 1
            
            return edge_path, {
                'success': True,
                'total_distance': total_distance,
                'nodes_expanded': nodes_expanded,
                'search_depth': search_depth,
                'max_frontier_size': max_frontier_size,
                'num_edges': len(edge_path),
                'edges_visited': edges_visited,
                'heuristic_used': 'Euclidean',
                'final_heuristic_cost': f_score.get(goal_node, 0),
                'final_total_cost': g_score.get(goal_node, 0)
            }
        
        closed_set.add(current)
        
        # Explore neighbors
        for neighbor in G.successors(current):
            if neighbor in closed_set:
                continue
            
            # Calculate tentative g score
            edge_data = G.get_edge_data(current, neighbor)
            if not edge_data:
                continue
                
            tentative_g = g_score[current] + edge_data['length']
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic_distance(G, neighbor, goal_node)
                
                # Add to open set if not already there
                if neighbor not in [node for _, node in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    max_frontier_size = max(max_frontier_size, len(open_set))
    
    # No path found
    return [], {
        'success': False,
        'error': 'No path found',
        'nodes_expanded': nodes_expanded,
        'search_depth': search_depth,
        'max_frontier_size': max_frontier_size
    }


def run_simulation(net_file: str, route_file: str) -> Dict:
    """
    Run SUMO simulation with TraCI and track vehicle movement
    """
    print(f"Running SUMO simulation with route file: {route_file}")
    
    # Parse route file to get edges
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    # Find vehicle route
    vehicle = root.find("vehicle")
    if vehicle is None:
        return {'success': False, 'error': 'No vehicle found in route file'}
    
    route_elem = vehicle.find("route")
    if route_elem is None:
        return {'success': False, 'error': 'No route found in vehicle'}
    
    edges = route_elem.get('edges', '').split()
    if not edges:
        return {'success': False, 'error': 'No edges found in route'}
    
    destination_edge = edges[-1]
    vehicle_id = vehicle.get('id', 'test_vehicle')
    
    # SUMO command
    sumo_cmd = [
        'sumo',
        '--net-file', net_file,
        '--route-files', route_file,
        '--no-step-log', 'true',
        '--no-warnings', 'true',
        '--random', 'false',
        '--time-to-teleport', '300',
        '--ignore-route-errors', 'true',
        '--collision.action', 'none'
    ]
    
    print(f"Starting SUMO simulation with command: {' '.join(sumo_cmd)}")
    
    try:
        # Start TraCI
        traci.start(sumo_cmd)
        
        # Initialize tracking variables
        start_time = 0
        num_steps = 0
        arrival_success = False
        final_edge = ''
        
        # Simulation loop
        while traci.simulation.getMinExpectedNumber() > 0:
            # Check if vehicle exists
            if vehicle_id in traci.vehicle.getIDList():
                try:
                    current_edge = traci.vehicle.getRoadID(vehicle_id)
                    final_edge = current_edge
                    
                    # Check if vehicle reached destination
                    if current_edge == destination_edge:
                        arrival_success = True
                        break
                        
                except traci.exceptions.TraCIException:
                    # Vehicle might have left the network
                    pass
            
            # Check for arrived vehicles
            arrived = traci.simulation.getArrivedIDList()
            if vehicle_id in arrived:
                arrival_success = True
                break
            
            # Step simulation
            traci.simulationStep()
            num_steps += 1
            
            # Timeout after 1000 steps
            if num_steps >= 1000:
                break
        
        # Get final simulation time
        total_time = traci.simulation.getTime()
        
        # Clean up
        traci.close()
        
        return {
            'success': True,
            'total_time': total_time,
            'num_steps': num_steps,
            'arrival_success': arrival_success,
            'final_edge': final_edge,
            'destination_edge': destination_edge
        }
        
    except Exception as e:
        try:
            traci.close()
        except:
            pass
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    # Test the implementation
    print("A* Algorithm Implementation Test")
    print("=" * 50)
    
    # Test with CP2 network
    net_file = "../CP2/cp2.net.xml"
    G, net = build_graph(net_file)
    
    # Test pathfinding
    start_edge = "24584377#4"
    goal_edge = "-1076782195#0"
    
    edges, path_metrics = a_star_path(G, net, start_edge, goal_edge)
    
    if edges and path_metrics['success']:
        print(f"✅ Path found with {len(edges)} edges")
        print(f"Total distance: {path_metrics['total_distance']:.2f}")
        print(f"Nodes expanded: {path_metrics['nodes_expanded']}")
        
        # Create test route file
        routes_root = ET.Element("routes")
        ET.SubElement(routes_root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5",
                      length="5", minGap="2.5", maxSpeed="70", guiShape="passenger")
        
        veh = ET.SubElement(routes_root, "vehicle", id="test_vehicle", type="car", depart="0")
        ET.SubElement(veh, "route", edges=" ".join(edges))
        
        temp_route_file = "test_astar_route.rou.xml"
        temp_tree = ET.ElementTree(routes_root)
        temp_tree.write(temp_route_file)
        
        # Run simulation
        sim_metrics = run_simulation(net_file, temp_route_file)
        
        if sim_metrics['success']:
            print(f"✅ Simulation successful!")
            print(f"Total time: {sim_metrics['total_time']:.2f} seconds")
            print(f"Num steps: {sim_metrics['num_steps']}")
            print(f"Arrival success: {sim_metrics['arrival_success']}")
            print(f"Final edge: {sim_metrics['final_edge']}")
        else:
            print(f"❌ Simulation failed: {sim_metrics.get('error', 'Unknown error')}")
        
        # Clean up
        if os.path.exists(temp_route_file):
            os.remove(temp_route_file)
    else:
        print(f"❌ No path found: {path_metrics.get('error', 'Unknown error')}") 