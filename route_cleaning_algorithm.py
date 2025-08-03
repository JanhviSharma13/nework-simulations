#!/usr/bin/env python3
"""
Route Cleaning Algorithm for SUMO Networks
Validates trips using Dijkstra and creates cleaned route files
"""

import os
import sys
import xml.etree.ElementTree as ET
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

# Add sumo tools to path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    print("Warning: SUMO_HOME not set, trying to find sumolib...")

try:
    import sumolib
    import traci
except ImportError:
    print("Warning: sumolib/traci not available, using networkx fallback")
    sumolib = None

class RouteCleaner:
    def __init__(self):
        self.networks = {}
        self.graphs = {}
        self.valid_trips = {}
        self.dijkstra_reference = []
        
    def load_network(self, net_file: str) -> bool:
        """Load SUMO network and build graph"""
        try:
            if sumolib:
                # Use sumolib for better SUMO integration
                net = sumolib.net.readNet(net_file)
                self.networks[net_file] = net
                
                # Build NetworkX graph
                G = nx.DiGraph()
                
                # Add nodes
                for node in net.getNodes():
                    G.add_node(node.getID(), pos=(node.getCoord()[0], node.getCoord()[1]))
                
                # Add edges with weights
                for edge in net.getEdges():
                    edge_id = edge.getID()
                    # Use the correct sumolib API
                    from_node = edge.getFromNode().getID()
                    to_node = edge.getToNode().getID()
                    length = edge.getLength()
                    
                    G.add_edge(from_node, to_node, 
                              edge_id=edge_id, 
                              length=length,
                              weight=length)  # Use length as weight
                
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
    
    def parse_trips(self, trips_file: str) -> List[Dict]:
        """Parse trip file and extract start/goal pairs"""
        trips = []
        try:
            tree = ET.parse(trips_file)
            root = tree.getroot()
            
            for trip in root.findall('.//trip'):
                trip_data = {
                    'id': trip.get('id'),
                    'depart': float(trip.get('depart', 0)),
                    'from': trip.get('from'),
                    'to': trip.get('to'),
                    'type': trip.get('type', 'passenger')
                }
                trips.append(trip_data)
            
            print(f"Parsed {len(trips)} trips from {trips_file}")
            return trips
            
        except Exception as e:
            print(f"Error parsing trips file {trips_file}: {e}")
            return []
    
    def validate_trip_dijkstra(self, net_file: str, from_edge: str, to_edge: str) -> Optional[Dict]:
        """Validate trip using Dijkstra algorithm"""
        G = self.graphs[net_file]
        
        # Get start and goal nodes
        start_from, start_to = self.get_edge_nodes(net_file, from_edge)
        goal_from, goal_to = self.get_edge_nodes(net_file, to_edge)
        
        if not start_to or not goal_from:
            return None
        
        try:
            # Run Dijkstra from start_to to goal_from
            path = nx.shortest_path(G, start_to, goal_from, weight='weight')
            
            if path:
                # Calculate total distance
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
                    'dijkstra_route': ','.join(edge_sequence)
                }
            else:
                return {'success': False}
                
        except nx.NetworkXNoPath:
            return {'success': False}
        except Exception as e:
            print(f"Error in Dijkstra validation: {e}")
            return {'success': False}
    
    def clean_routes(self, area_name: str, net_file: str, trips_file: str, output_dir: str = "."):
        """Main cleaning function"""
        print(f"\n=== Cleaning routes for {area_name} ===")
        
        # Load network
        if not self.load_network(net_file):
            print(f"Failed to load network for {area_name}")
            return
        
        # Parse trips
        trips = self.parse_trips(trips_file)
        if not trips:
            print(f"No trips found for {area_name}")
            return
        
        # Validate each trip
        valid_trips = []
        start_time = time.time()
        
        for i, trip in enumerate(trips):
            if i % 10 == 0:
                print(f"Validating trip {i+1}/{len(trips)}...")
            
            result = self.validate_trip_dijkstra(net_file, trip['from'], trip['to'])
            
            if result and result['success']:
                valid_trips.append(trip)
                
                # Log to dijkstra reference
                self.dijkstra_reference.append({
                    'route_id': trip['id'],
                    'map_name': area_name,
                    'start_edge': trip['from'],
                    'goal_edge': trip['to'],
                    'total_distance_m': result['total_distance'],
                    'num_edges': result['num_edges'],
                    'dijkstra_route': result['dijkstra_route']
                })
        
        processing_time = time.time() - start_time
        print(f"Validation complete: {len(valid_trips)}/{len(trips)} trips valid")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        # Save cleaned route file
        cleaned_file = os.path.join(output_dir, f"{area_name.lower()}_dijkstra_cleaned.rou.xml")
        self.save_cleaned_routes(valid_trips, cleaned_file)
        
        # Save dijkstra reference
        ref_file = os.path.join(output_dir, f"dijkstra_reference_{area_name.lower()}.csv")
        df = pd.DataFrame(self.dijkstra_reference)
        df.to_csv(ref_file, index=False)
        print(f"Saved dijkstra reference to {ref_file}")
        
        return len(valid_trips)
    
    def save_cleaned_routes(self, valid_trips: List[Dict], output_file: str):
        """Save cleaned routes to XML file"""
        root = ET.Element('routes')
        
        # Add vehicle type
        vtype = ET.SubElement(root, 'vType')
        vtype.set('id', 'car')
        vtype.set('accel', '1.0')
        vtype.set('decel', '5.0')
        vtype.set('maxSpeed', '25')
        vtype.set('length', '5')
        vtype.set('sigma', '0.5')
        vtype.set('color', '1,0,0')
        
        # Add valid trips
        for trip in valid_trips:
            vehicle = ET.SubElement(root, 'vehicle')
            vehicle.set('id', trip['id'])
            vehicle.set('type', 'car')
            vehicle.set('depart', str(trip['depart']))
            
            route = ET.SubElement(vehicle, 'route')
            route.set('edges', f"{trip['from']} {trip['to']}")
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        print(f"Saved {len(valid_trips)} valid trips to {output_file}")

def main():
    """Main execution function"""
    cleaner = RouteCleaner()
    
    # Define areas and their files - using original networks instead of cleaned ones
    areas = {
        'CP2': {
            'net_file': 'CP2/cp2.net.xml',
            'trips_file': 'CP2/cp2.trips.xml'
        },
        'CP': {
            'net_file': 'CP/cp.net.xml',
            'trips_file': 'CP/cp.trips.xml'  # We'll need to check if this exists
        },
        'RakabGanj': {
            'net_file': 'Rakab-Ganj/rg.net.xml',
            'trips_file': 'Rakab-Ganj/rg.trips.xml'  # We'll need to check if this exists
        },
        'Safdarjung': {
            'net_file': 'Safdarjung/SE.net.xml',
            'trips_file': 'Safdarjung/SE.trips.xml'
        },
        'ChandniChowk': {
            'net_file': 'Chandni-Chowk/ch.net.xml',
            'trips_file': 'Chandni-Chowk/ch.trips.xml'
        }
    }
    
    # Process each area
    total_start_time = time.time()
    results = {}
    
    for area_name, files in areas.items():
        if os.path.exists(files['net_file']) and os.path.exists(files['trips_file']):
            print(f"\n{'='*50}")
            print(f"Processing {area_name}")
            print(f"{'='*50}")
            
            valid_count = cleaner.clean_routes(area_name, files['net_file'], files['trips_file'])
            results[area_name] = valid_count
        else:
            print(f"Skipping {area_name}: Missing files")
            print(f"  Network: {files['net_file']} - {'Exists' if os.path.exists(files['net_file']) else 'Missing'}")
            print(f"  Trips: {files['trips_file']} - {'Exists' if os.path.exists(files['trips_file']) else 'Missing'}")
    
    total_time = time.time() - total_start_time
    print(f"\n{'='*50}")
    print("CLEANING COMPLETE")
    print(f"{'='*50}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print("\nResults:")
    for area, count in results.items():
        print(f"  {area}: {count} valid trips")
    
    # Save combined dijkstra reference
    if cleaner.dijkstra_reference:
        combined_ref = pd.DataFrame(cleaner.dijkstra_reference)
        combined_ref.to_csv('dijkstra_reference_all_areas.csv', index=False)
        print(f"\nSaved combined reference to dijkstra_reference_all_areas.csv")

if __name__ == "__main__":
    main() 