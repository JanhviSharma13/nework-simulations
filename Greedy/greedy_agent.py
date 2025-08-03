import networkx as nx
import sumolib
import pandas as pd
import time
from datetime import datetime
import heapq

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

def greedy_path(G, net, from_edge_id, to_edge_id, boundary_edges=None):
    """Greedy algorithm implementation"""
    try:
        from_node = net.getEdge(from_edge_id).getToNode().getID()
        to_node = net.getEdge(to_edge_id).getFromNode().getID()
    except:
        return None, {'success': False, 'error': 'Invalid edge IDs'}

    if from_node not in G or to_node not in G:
        return None, {'success': False, 'error': 'Start or goal node not in graph'}

    # Use the full graph for greedy search (no boundary filtering)
    G_search = G

    # Greedy algorithm: always choose the neighbor closest to goal
    current_node = from_node
    path = [current_node]
    edge_sequence = []
    total_distance = 0
    nodes_expanded = 0
    max_frontier_size = 1
    
    visited = set()
    visited.add(current_node)
    
    while current_node != to_node and nodes_expanded < 1000:  # Prevent infinite loops
        nodes_expanded += 1
        
        if current_node not in G_search:
            return [], {'success': False, 'error': 'Node not in graph'}
        
        # Find best neighbor (closest to goal)
        best_neighbor = None
        best_distance = float('inf')
        
        for neighbor in G_search[current_node]:
            if neighbor in visited:
                continue
                
            # Calculate distance to goal for this neighbor
            try:
                neighbor_coord = net.getNode(neighbor).getCoord()
                goal_coord = net.getNode(to_node).getCoord()
                
                # Straight-line distance to goal
                distance_to_goal = ((neighbor_coord[0] - goal_coord[0])**2 + 
                                  (neighbor_coord[1] - goal_coord[1])**2)**0.5
                
                if distance_to_goal < best_distance:
                    best_distance = distance_to_goal
                    best_neighbor = neighbor
            except:
                continue
        
        if best_neighbor is None:
            # No unvisited neighbors, backtrack or fail
            if len(path) > 1:
                path.pop()
                current_node = path[-1]
                continue
            else:
                return [], {'success': False, 'error': 'No path found'}
        
        # Move to best neighbor
        edge_data = G_search[current_node][best_neighbor]
        edge_sequence.append(edge_data['id'])
        total_distance += edge_data['weight']
        
        current_node = best_neighbor
        path.append(current_node)
        visited.add(current_node)
        
        max_frontier_size = max(max_frontier_size, len(G_search[current_node]))
    
    if current_node == to_node:
        return edge_sequence, {
            'success': True,
            'total_distance': total_distance,
            'num_edges': len(edge_sequence),
            'path': path,
            'edge_sequence': edge_sequence,
            'nodes_expanded': nodes_expanded,
            'search_depth': len(path) - 1,
            'max_frontier_size': max_frontier_size
        }
    else:
        return [], {'success': False, 'error': 'Goal not reached'}

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