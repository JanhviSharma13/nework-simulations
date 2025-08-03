#!/usr/bin/env python3
"""
Dijkstra and A* Baseline Algorithms for SUMO Networks
Comprehensive implementation with detailed metrics logging
"""

import os
import sys
import xml.etree.ElementTree as ET
import pandas as pd
import networkx as nx
import heapq
import math
import time
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import traci
import sumolib

# Add sumo tools to path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)

class AStarNode:
    """Node class for A* algorithm"""
    def __init__(self, node_id: str, g_cost: float, h_cost: float, parent=None):
        self.node_id = node_id
        self.g_cost = g_cost  # Cost from start to this node
        self.h_cost = h_cost  # Heuristic cost to goal
        self.f_cost = g_cost + h_cost  # Total cost
        self.parent = parent
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost

class BaselineAlgorithms:
    def __init__(self):
        self.networks = {}
        self.graphs = {}
        self.node_positions = {}
        self.results = {
            'dijkstra': [],
            'astar': []
        }
    
    def load_network(self, net_file: str) -> bool:
        """Load SUMO network and build graph"""
        try:
            if sumolib:
                net = sumolib.net.readNet(net_file)
                self.networks[net_file] = net
                
                # Build NetworkX graph
                G = nx.DiGraph()
                
                # Add nodes and store positions
                for node in net.getNodes():
                    node_id = node.getID()
                    coords = node.getCoord()
                    G.add_node(node_id, pos=(coords[0], coords[1]))
                    self.node_positions[node_id] = (coords[0], coords[1])
                
                # Add edges with weights
                for edge in net.getEdges():
                    edge_id = edge.getID()
                    from_node = edge.getFromNode().getID()
                    to_node = edge.getToNode().getID()
                    length = edge.getLength()
                    
                    G.add_edge(from_node, to_node, 
                              edge_id=edge_id, 
                              length=length,
                              weight=length)
                
                self.graphs[net_file] = G
                print(f"Loaded network {net_file}: {len(G.nodes)} nodes, {len(G.edges)} edges")
                return True
                
            else:
                # Fallback: parse XML directly
                print(f"Parsing network XML: {net_file}")
                tree = ET.parse(net_file)
                root = tree.getroot()
                
                G = nx.DiGraph()
                
                # Parse nodes
                for node in root.findall('.//node'):
                    node_id = node.get('id')
                    x = float(node.get('x', 0))
                    y = float(node.get('y', 0))
                    G.add_node(node_id, pos=(x, y))
                    self.node_positions[node_id] = (x, y)
                
                # Parse edges
                for edge in root.findall('.//edge'):
                    edge_id = edge.get('id')
                    from_node = edge.get('from')
                    to_node = edge.get('to')
                    length = float(edge.get('length', 1.0))
                    
                    G.add_edge(from_node, to_node,
                              edge_id=edge_id,
                              length=length,
                              weight=length)
                
                self.graphs[net_file] = G
                print(f"Loaded network {net_file}: {len(G.nodes)} nodes, {len(G.edges)} edges")
                return True
                
        except Exception as e:
            print(f"Error loading network {net_file}: {e}")
            return False
    
    def get_edge_nodes(self, net_file: str, edge_id: str) -> Tuple[str, str]:
        """Get from_node and to_node for an edge"""
        if sumolib and net_file in self.networks:
            net = self.networks[net_file]
            edge = net.getEdge(edge_id)
            if edge:
                return edge.getFromNode().getID(), edge.getToNode().getID()
        
        # Fallback: use graph edges
        G = self.graphs[net_file]
        for u, v, data in G.edges(data=True):
            if data.get('edge_id') == edge_id:
                return u, v
        
        return None, None
    
    def calculate_heuristic(self, node1: str, node2: str, heuristic_type: str = "straight_line") -> float:
        """Calculate heuristic distance between two nodes"""
        if node1 not in self.node_positions or node2 not in self.node_positions:
            return 0.0
        
        pos1 = self.node_positions[node1]
        pos2 = self.node_positions[node2]
        
        if heuristic_type == "straight_line":
            return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        elif heuristic_type == "manhattan":
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        else:
            return 0.0
    
    def dijkstra_algorithm(self, net_file: str, start_node: str, goal_node: str) -> Dict:
        """Dijkstra algorithm implementation"""
        G = self.graphs[net_file]
        
        if start_node not in G or goal_node not in G:
            return {'success': False, 'error': 'Start or goal node not in graph'}
        
        # Initialize
        distances = {node: float('inf') for node in G.nodes()}
        distances[start_node] = 0
        previous = {}
        visited = set()
        nodes_expanded = 0
        
        # Priority queue: (distance, node)
        pq = [(0, start_node)]
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            nodes_expanded += 1
            
            if current_node == goal_node:
                # Reconstruct path
                path = []
                current = goal_node
                while current is not None:
                    path.append(current)
                    current = previous.get(current)
                path.reverse()
                
                # Calculate total distance and edge sequence
                total_distance = 0
                edge_sequence = []
                
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge_data = G[u][v]
                    total_distance += edge_data['length']
                    edge_sequence.append(edge_data['edge_id'])
                
                return {
                    'success': True,
                    'total_distance': total_distance,
                    'num_edges': len(edge_sequence),
                    'path': path,
                    'edge_sequence': edge_sequence,
                    'nodes_expanded': nodes_expanded,
                    'search_depth': len(path) - 1
                }
            
            # Explore neighbors
            for neighbor, edge_data in G[current_node].items():
                if neighbor not in visited:
                    new_dist = current_dist + edge_data['weight']
                    
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        previous[neighbor] = current_node
                        heapq.heappush(pq, (new_dist, neighbor))
        
        return {'success': False, 'error': 'No path found'}
    
    def astar_algorithm(self, net_file: str, start_node: str, goal_node: str, 
                       heuristic_type: str = "straight_line") -> Dict:
        """A* algorithm implementation with detailed metrics"""
        G = self.graphs[net_file]
        
        if start_node not in G or goal_node not in G:
            return {'success': False, 'error': 'Start or goal node not in graph'}
        
        # Initialize
        open_set = []
        closed_set = set()
        came_from = {}
        
        # Cost from start to current node
        g_score = {node: float('inf') for node in G.nodes()}
        g_score[start_node] = 0
        
        # Estimated total cost from start to goal through current node
        f_score = {node: float('inf') for node in G.nodes()}
        f_score[start_node] = self.calculate_heuristic(start_node, goal_node, heuristic_type)
        
        # Priority queue: (f_score, node)
        heapq.heappush(open_set, (f_score[start_node], start_node))
        
        # Metrics tracking
        nodes_expanded = 0
        max_frontier_size = 1
        edges_visited = set()
        
        while open_set:
            current_f, current_node = heapq.heappop(open_set)
            
            if current_node in closed_set:
                continue
            
            closed_set.add(current_node)
            nodes_expanded += 1
            
            if current_node == goal_node:
                # Reconstruct path
                path = []
                current = goal_node
                while current is not None:
                    path.append(current)
                    current = came_from.get(current)
                path.reverse()
                
                # Calculate metrics
                total_distance = 0
                edge_sequence = []
                
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge_data = G[u][v]
                    total_distance += edge_data['length']
                    edge_sequence.append(edge_data['edge_id'])
                    edges_visited.add(edge_data['edge_id'])
                
                final_heuristic_cost = self.calculate_heuristic(goal_node, goal_node, heuristic_type)
                final_total_cost = g_score[goal_node] + final_heuristic_cost
                
                return {
                    'success': True,
                    'total_distance': total_distance,
                    'num_edges': len(edge_sequence),
                    'path': path,
                    'edge_sequence': edge_sequence,
                    'nodes_expanded': nodes_expanded,
                    'max_frontier_size': max_frontier_size,
                    'edges_visited': len(edges_visited),
                    'search_depth': len(path) - 1,
                    'heuristic_used': heuristic_type,
                    'final_heuristic_cost': final_heuristic_cost,
                    'final_total_cost': final_total_cost
                }
            
            # Explore neighbors
            for neighbor, edge_data in G[current_node].items():
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score[current_node] + edge_data['weight']
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.calculate_heuristic(neighbor, goal_node, heuristic_type)
                    
                    if neighbor not in [node for _, node in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        edges_visited.add(edge_data['edge_id'])
            
            max_frontier_size = max(max_frontier_size, len(open_set))
        
        return {'success': False, 'error': 'No path found'}
    
    def run_simulation(self, net_file: str, route_file: str, area_name: str, 
                      algorithm: str, max_steps: int = 1000) -> Dict:
        """Run SUMO simulation for a route"""
        try:
            # Start SUMO
            sumo_cmd = [
                'sumo', '--net-file', net_file,
                '--route-files', route_file,
                '--no-step-log', 'true',
                '--no-warnings', 'true',
                '--random', 'false'
            ]
            
            traci.start(sumo_cmd)
            
            # Get vehicle ID
            vehicle_id = traci.vehicle.getIDList()[0] if traci.vehicle.getIDList() else None
            
            if not vehicle_id:
                traci.close()
                return {'success': False, 'error': 'No vehicle found'}
            
            # Track metrics
            start_time = time.time()
            num_steps = 0
            final_edge = None
            arrival_success = False
            
            while num_steps < max_steps:
                traci.simulationStep()
                num_steps += 1
                
                if not traci.vehicle.getIDList():  # Vehicle left simulation
                    break
                
                # Get current edge
                current_edge = traci.vehicle.getRoadID(vehicle_id)
                final_edge = current_edge
                
                # Check if arrived at destination
                if traci.vehicle.getRouteIndex(vehicle_id) == -1:
                    arrival_success = True
                    break
            
            total_time = time.time() - start_time
            traci.close()
            
            return {
                'success': True,
                'total_time': total_time,
                'num_steps': num_steps,
                'final_edge': final_edge,
                'arrival_success': arrival_success
            }
            
        except Exception as e:
            if 'traci' in locals():
                traci.close()
            return {'success': False, 'error': str(e)}
    
    def process_area(self, area_name: str, net_file: str, route_file: str, 
                    trips_data: List[Dict]) -> None:
        """Process all trips for an area using both algorithms"""
        print(f"\n=== Processing {area_name} ===")
        
        if not self.load_network(net_file):
            print(f"Failed to load network for {area_name}")
            return
        
        for i, trip in enumerate(trips_data):
            print(f"Processing trip {i+1}/{len(trips_data)}: {trip['from']} -> {trip['to']}")
            
            # Get start and goal nodes
            start_from, start_to = self.get_edge_nodes(net_file, trip['from'])
            goal_from, goal_to = self.get_edge_nodes(net_file, trip['to'])
            
            if not start_to or not goal_from:
                print(f"  Skipping: Invalid edge mapping")
                continue
            
            # Run Dijkstra
            dijkstra_result = self.dijkstra_algorithm(net_file, start_to, goal_from)
            
            if dijkstra_result['success']:
                # Run simulation for Dijkstra
                sim_result = self.run_simulation(net_file, route_file, area_name, "dijkstra")
                
                # Log Dijkstra results
                self.results['dijkstra'].append({
                    'episode': i,
                    'map_name': area_name,
                    'start_edge': trip['from'],
                    'goal_edge': trip['to'],
                    'success': 1,
                    'total_time': sim_result.get('total_time', 0),
                    'route_length': dijkstra_result['num_edges'],
                    'total_distance': dijkstra_result['total_distance'],
                    'final_edge': sim_result.get('final_edge', ''),
                    'num_steps': sim_result.get('num_steps', 0),
                    'arrival_success': sim_result.get('arrival_success', False),
                    'nodes_expanded': dijkstra_result['nodes_expanded'],
                    'search_depth': dijkstra_result['search_depth']
                })
                
                # Run A* with straight-line heuristic
                astar_result = self.astar_algorithm(net_file, start_to, goal_from, "straight_line")
                
                if astar_result['success']:
                    # Run simulation for A*
                    sim_result = self.run_simulation(net_file, route_file, area_name, "astar")
                    
                    # Log A* results
                    self.results['astar'].append({
                        'episode': i,
                        'map_name': area_name,
                        'start_edge': trip['from'],
                        'goal_edge': trip['to'],
                        'success': 1,
                        'total_time': sim_result.get('total_time', 0),
                        'route_length': astar_result['num_edges'],
                        'total_distance': astar_result['total_distance'],
                        'final_edge': sim_result.get('final_edge', ''),
                        'num_steps': sim_result.get('num_steps', 0),
                        'arrival_success': sim_result.get('arrival_success', False),
                        'heuristic_used': astar_result['heuristic_used'],
                        'total_nodes_expanded': astar_result['nodes_expanded'],
                        'max_frontier_size': astar_result['max_frontier_size'],
                        'final_heuristic_cost': astar_result['final_heuristic_cost'],
                        'final_total_cost': astar_result['final_total_cost'],
                        'search_depth': astar_result['search_depth'],
                        'num_edges_visited': astar_result['edges_visited']
                    })
                else:
                    print(f"  A* failed: {astar_result.get('error', 'Unknown error')}")
            else:
                print(f"  Dijkstra failed: {dijkstra_result.get('error', 'Unknown error')}")
    
    def save_results(self, output_dir: str = "."):
        """Save results to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save Dijkstra results
        if self.results['dijkstra']:
            dijkstra_df = pd.DataFrame(self.results['dijkstra'])
            dijkstra_file = os.path.join(output_dir, f"dijkstra_baseline_{timestamp}.csv")
            dijkstra_df.to_csv(dijkstra_file, index=False)
            print(f"Saved Dijkstra results to {dijkstra_file}")
        
        # Save A* results
        if self.results['astar']:
            astar_df = pd.DataFrame(self.results['astar'])
            astar_file = os.path.join(output_dir, f"astar_baseline_{timestamp}.csv")
            astar_df.to_csv(astar_file, index=False)
            print(f"Saved A* results to {astar_file}")

def main():
    """Main execution function"""
    algorithms = BaselineAlgorithms()
    
    # Define areas and their files - using original networks
    areas = {
        'CP2': {
            'net_file': 'CP2/cp2.net.xml',
            'route_file': 'CP2/cp2.rou.xml',
            'trips_file': 'CP2/cp2.trips.xml'
        },
        'CP': {
            'net_file': 'CP/cp.net.xml',
            'route_file': 'CP/cp.rou.xml',
            'trips_file': 'CP/cp.trips.xml'
        },
        'RakabGanj': {
            'net_file': 'Rakab-Ganj/rg.net.xml',
            'route_file': 'Rakab-Ganj/rg.rou.xml',
            'trips_file': 'Rakab-Ganj/rg.trips.xml'
        },
        'Safdarjung': {
            'net_file': 'Safdarjung/SE.net.xml',
            'route_file': 'Safdarjung/SE.rou.xml',
            'trips_file': 'Safdarjung/SE.trips.xml'
        },
        'ChandniChowk': {
            'net_file': 'Chandni-Chowk/ch.net.xml',
            'route_file': 'Chandni-Chowk/ch.rou.xml',
            'trips_file': 'Chandni-Chowk/ch.trips.xml'
        }
    }
    
    # Process each area
    total_start_time = time.time()
    
    for area_name, files in areas.items():
        if (os.path.exists(files['net_file']) and 
            os.path.exists(files['route_file']) and 
            os.path.exists(files['trips_file'])):
            
            print(f"\n{'='*50}")
            print(f"Processing {area_name}")
            print(f"{'='*50}")
            
            # Parse trips
            tree = ET.parse(files['trips_file'])
            root = tree.getroot()
            trips = []
            
            for trip in root.findall('.//trip'):
                trips.append({
                    'id': trip.get('id'),
                    'from': trip.get('from'),
                    'to': trip.get('to'),
                    'depart': float(trip.get('depart', 0))
                })
            
            print(f"Found {len(trips)} trips")
            
            # Process area
            algorithms.process_area(area_name, files['net_file'], files['route_file'], trips)
            
        else:
            print(f"Skipping {area_name}: Missing files")
            print(f"  Network: {files['net_file']} - {'Exists' if os.path.exists(files['net_file']) else 'Missing'}")
            print(f"  Route: {files['route_file']} - {'Exists' if os.path.exists(files['route_file']) else 'Missing'}")
            print(f"  Trips: {files['trips_file']} - {'Exists' if os.path.exists(files['trips_file']) else 'Missing'}")
    
    # Save results
    algorithms.save_results()
    
    total_time = time.time() - total_start_time
    print(f"\n{'='*50}")
    print("BASELINE ALGORITHMS COMPLETE")
    print(f"{'='*50}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Dijkstra results: {len(algorithms.results['dijkstra'])}")
    print(f"A* results: {len(algorithms.results['astar'])}")

if __name__ == "__main__":
    main() 