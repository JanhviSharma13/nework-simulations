import networkx as nx
import sumolib
from math import sqrt
import pandas as pd
import time
from datetime import datetime

def heuristic(coord1, coord2):
    """Fixed heuristic function - calculates straight-line distance"""
    return sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def build_graph(net_file):
    """Build graph from SUMO network file"""
    net = sumolib.net.readNet(net_file)
    G = nx.DiGraph()

    for edge in net.getEdges():
        if not edge.allows("passenger"):
            continue
        from_node = edge.getFromNode()
        to_node = edge.getToNode()
        edge_id = edge.getID()
        length = edge.getLength()

        G.add_edge(
            from_node.getID(),
            to_node.getID(),
            id=edge_id,
            weight=length,
            from_coord=from_node.getCoord(),
            to_coord=to_node.getCoord()
        )

    return G, net

def a_star_path(G, net, from_edge_id, to_edge_id):
    """A* pathfinding with detailed metrics"""
    try:
        from_node = net.getEdge(from_edge_id).getToNode().getID()
        to_node = net.getEdge(to_edge_id).getFromNode().getID()
    except:
        return None, {'success': False, 'error': 'Invalid edge IDs'}

    def h(n1, n2=to_node):
        try:
            c1 = net.getNode(n1).getCoord()
            c2 = net.getNode(n2).getCoord()
            return heuristic(c1, c2)
        except:
            return 0

    # A* implementation with detailed metrics
    open_set = []
    closed_set = set()
    came_from = {}
    
    # Cost from start to current node
    g_score = {node: float('inf') for node in G.nodes()}
    g_score[from_node] = 0
    
    # Estimated total cost from start to goal through current node
    f_score = {node: float('inf') for node in G.nodes()}
    f_score[from_node] = h(from_node)
    
    # Priority queue: (f_score, node)
    import heapq
    heapq.heappush(open_set, (f_score[from_node], from_node))
    
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
        
        if current_node == to_node:
            # Reconstruct path
            path = []
            current = to_node
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
                total_distance += edge_data['weight']
                edge_sequence.append(edge_data['id'])
                edges_visited.add(edge_data['id'])
            
            final_heuristic_cost = h(to_node)
            final_total_cost = g_score[to_node] + final_heuristic_cost
            
            return edge_sequence, {
                'success': True,
                'total_distance': total_distance,
                'num_edges': len(edge_sequence),
                'path': path,
                'edge_sequence': edge_sequence,
                'nodes_expanded': nodes_expanded,
                'max_frontier_size': max_frontier_size,
                'edges_visited': len(edges_visited),
                'search_depth': len(path) - 1,
                'heuristic_used': 'straight_line',
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
                f_score[neighbor] = tentative_g_score + h(neighbor)
                
                if neighbor not in [node for _, node in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    edges_visited.add(edge_data['id'])
        
        max_frontier_size = max(max_frontier_size, len(open_set))
    
    return [], {'success': False, 'error': 'No path found'}

def run_simulation(net_file, route_file, max_steps=1000):
    """Run SUMO simulation and collect metrics"""
    import traci
    import os
    import xml.etree.ElementTree as ET
    import tempfile
    
    temp_route_file = None
    
    try:
        # Create a proper route file with just one vehicle
        routes_root = ET.Element("routes")
        
        # Add vehicle type definition
        ET.SubElement(routes_root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5",
                      length="5", minGap="2.5", maxSpeed="70", guiShape="passenger")
        
        # Parse the original route file to get the vehicle route
        tree = ET.parse(route_file)
        root = tree.getroot()
        
        # Find the first vehicle and its route
        vehicle = root.find("vehicle")
        if vehicle is None:
            return {'success': False, 'error': 'No vehicle found in route file'}
        
        route = vehicle.find("route")
        if route is None:
            return {'success': False, 'error': 'No route found for vehicle'}
        
        edges = route.attrib.get("edges", "").split()
        if not edges:
            return {'success': False, 'error': 'No edges found in route'}
        
        # Create new vehicle with proper route
        veh = ET.SubElement(routes_root, "vehicle", id="test_vehicle", type="car", depart="0")
        ET.SubElement(veh, "route", edges=" ".join(edges))
        
        # Write temporary route file
        temp_route_file = tempfile.NamedTemporaryFile(mode='w', suffix='.rou.xml', delete=False)
        temp_tree = ET.ElementTree(routes_root)
        temp_tree.write(temp_route_file.name)
        temp_route_file.close()
        
        # Start SUMO (no GUI)
        sumo_cmd = [
            'sumo', '--net-file', net_file,
            '--route-files', temp_route_file.name,
            '--no-step-log', 'true',
            '--no-warnings', 'true',
            '--random', 'false',
            '--time-to-teleport', '300',  # Allow teleporting after 300 seconds
            '--ignore-route-errors', 'true',  # Ignore route validation errors
            '--collision.action', 'none'  # Ignore accident checks
        ]
        
        print(f"Starting SUMO simulation with command: {' '.join(sumo_cmd)}")
        traci.start(sumo_cmd)
        
        # Get vehicle ID
        vehicle_id = "test_vehicle"
        
        # Track metrics
        start_time = 0
        num_steps = 0
        final_edge = None
        arrival_success = False
        destination_edge = edges[-1] if edges else None
        
        # Run simulation
        while num_steps < max_steps:
            traci.simulationStep()
            num_steps += 1
            
            # Get current simulation time
            current_time = traci.simulation.getTime()
            
            # Check if vehicle exists
            if vehicle_id in traci.vehicle.getIDList():
                # Get current edge
                try:
                    current_edge = traci.vehicle.getRoadID(vehicle_id)
                    final_edge = current_edge
                    
                    # Check if we've reached the destination
                    if current_edge == destination_edge:
                        arrival_success = True
                        break
                        
                except traci.exceptions.TraCIException:
                    # Vehicle might have left the network
                    break
            else:
                # Check if vehicle has arrived (left the network)
                arrived_vehicles = traci.simulation.getArrivedIDList()
                if vehicle_id in arrived_vehicles:
                    # Vehicle completed its route
                    arrival_success = True
                    break
                
                # If vehicle doesn't exist and hasn't arrived, it might have been teleported
                if num_steps > 10:  # Give some time for vehicle to be inserted
                    break
        
        total_time = traci.simulation.getTime() - start_time
        traci.close()
        
        print(f"Simulation completed: total_time={total_time}, arrival_success={arrival_success}")
        
        return {
            'success': True,
            'total_time': total_time,
            'num_steps': num_steps,
            'final_edge': final_edge or '',
            'arrival_success': arrival_success
        }
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        
        if 'traci' in locals():
            try:
                traci.close()
            except:
                pass
        
        return {'success': False, 'error': str(e)}
    
    finally:
        # Clean up temp file
        if temp_route_file and os.path.exists(temp_route_file.name):
            try:
                os.unlink(temp_route_file.name)
            except:
                pass 