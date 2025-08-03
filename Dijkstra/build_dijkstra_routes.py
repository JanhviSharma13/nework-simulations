import xml.etree.ElementTree as ET
from dijkstra_agent import build_graph, dijkstra_path, run_simulation
import pandas as pd
import time
from datetime import datetime

def parse_trips(trips_file):
    """Parse trip file and extract start/goal pairs"""
    tree = ET.parse(trips_file)
    root = tree.getroot()
    trips = []
    for trip in root.findall("trip"):
        trip_data = {
            'id': trip.attrib["id"],
            'depart': float(trip.attrib.get("depart", 0)),
            'from': trip.attrib["from"],
            'to': trip.attrib["to"],
            'type': trip.attrib.get("type", "passenger")
        }
        trips.append(trip_data)
    return trips

def generate_routes(net_file, trips_file, output_file, area_name):
    """Generate routes using Dijkstra with comprehensive logging"""
    print(f"Building graph for {area_name}...")
    G, net = build_graph(net_file)
    trips = parse_trips(trips_file)
    
    print(f"Processing {len(trips)} trips for {area_name}...")
    
    routes_root = ET.Element("routes")
    
    # Add vehicle type definition
    ET.SubElement(routes_root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5",
                  length="5", minGap="2.5", maxSpeed="70", guiShape="passenger")
    
    # Metrics collection
    metrics = []
    successful_routes = 0
    
    for i, trip in enumerate(trips):
        if i % 10 == 0:
            print(f"Processing trip {i+1}/{len(trips)}...")
        
        edges, path_metrics = dijkstra_path(G, net, trip['from'], trip['to'])
        
        # Always log metrics (success and failure)
        if edges and path_metrics['success']:
            # Create vehicle with route
            veh = ET.SubElement(routes_root, "vehicle", id=f"veh{i}", type="car", depart=str(trip['depart']))
            ET.SubElement(veh, "route", edges=" ".join(edges))
            
            successful_routes += 1
            
            # Run simulation for this route - CREATE SINGLE VEHICLE ROUTE FILE
            temp_route_file = f"temp_dijkstra_route_{i}.rou.xml"
            
            # Create a NEW route file with ONLY this vehicle
            sim_routes_root = ET.Element("routes")
            ET.SubElement(sim_routes_root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5",
                          length="5", minGap="2.5", maxSpeed="70", guiShape="passenger")
            
            sim_veh = ET.SubElement(sim_routes_root, "vehicle", id=f"veh{i}", type="car", depart=str(trip['depart']))
            ET.SubElement(sim_veh, "route", edges=" ".join(edges))
            
            temp_tree = ET.ElementTree(sim_routes_root)
            temp_tree.write(temp_route_file)
            
            sim_metrics = run_simulation(net_file, temp_route_file)
            
            # Combine metrics for successful route
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
                'nodes_expanded': path_metrics['nodes_expanded'],
                'search_depth': path_metrics['search_depth']
            }
            
            # Clean up temp file
            import os
            if os.path.exists(temp_route_file):
                os.remove(temp_route_file)
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
                'nodes_expanded': path_metrics.get('nodes_expanded', 0),
                'search_depth': 0,
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
        metrics_file = f"dijkstra_metrics_{area_name.lower()}_{timestamp}.csv"
        df = pd.DataFrame(metrics)
        df.to_csv(metrics_file, index=False)
        print(f"Saved metrics to {metrics_file}")
    
    return successful_routes, metrics

if __name__ == "__main__":
    """Main execution for Dijkstra algorithm"""
    print("=" * 60)
    print("DIJKSTRA ALGORITHM ROUTE GENERATOR")
    print("=" * 60)
    
    start_time = time.time()
    
    # Define areas to process
    areas = [
        ("CP2", "CP2/cp2.net.xml", "A-Star/CP2/cp2_trips.xml", "Dijkstra/CP2/cp2_dijkstra_routes.rou.xml"),
        ("ChandniChowk", "Chandni-Chowk/ch.net.xml", "A-Star/ChandniChowk/chandnichowk_trips.xml", "Dijkstra/ChandniChowk/chandnichowk_dijkstra_routes.rou.xml"),
        ("RakabGanj", "Rakab-Ganj/rg.net.xml", "A-Star/RakabGanj/rakabganj_trips.xml", "Dijkstra/RakabGanj/rakabganj_dijkstra_routes.rou.xml"),
        ("Safdarjung", "Safdarjung/SE.net.xml", "A-Star/Safdarjung/safdarjung_trips.xml", "Dijkstra/Safdarjung/safdarjung_dijkstra_routes.rou.xml")
    ]
    
    results = {}
    
    for area_name, net_file, trips_file, output_file in areas:
        try:
            print(f"\nProcessing {area_name}...")
            successful_routes, metrics = generate_routes(net_file, trips_file, output_file, area_name)
            results[area_name] = f"✅ Generated {successful_routes} successful routes"
            print(f"✅ Generated {successful_routes} successful routes for {area_name}")
        except Exception as e:
            results[area_name] = f"❌ FAILED: {str(e)}"
            print(f"❌ Failed to build routes for {area_name}: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("DIJKSTRA ALGORITHM RUNNER COMPLETE")
    print("=" * 60)
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Results:")
    for area, result in results.items():
        print(f"  {area}: {result}") 