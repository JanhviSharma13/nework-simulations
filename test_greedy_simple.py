import traci
import os
import sys
import random
from greedy_routing_agent import GreedyTravelTimeAgent
import xml.etree.ElementTree as ET

def get_valid_edges_from_route(rou_file):
    """Get valid edges from route file"""
    try:
        # Parse route file to get edges
        tree = ET.parse(rou_file)
        root = tree.getroot()
        
        # Get all trip elements
        trips = root.findall('.//trip')
        
        # Extract unique edges from trips
        edges = set()
        for trip in trips:
            from_edge = trip.get('from')
            to_edge = trip.get('to')
            if from_edge:
                edges.add(from_edge)
            if to_edge:
                edges.add(to_edge)
                
        return list(edges)
        
    except Exception as e:
        print(f"❌ Error getting valid edges from route file: {e}")
        return []

def test_single_area():
    """Test the greedy agent on a single area (Rakab Ganj)"""
    
    print("🧪 Testing Greedy Travel-Time Routing Agent")
    print("=" * 50)
    
    # Configuration
    config_file = "Rakab-Ganj/rg.sumocfg"
    rou_file = "Rakab-Ganj/rg.rou.xml"
    
    try:
        # Get valid edges from route file
        print("🔍 Getting valid edges from route file...")
        valid_edges = get_valid_edges_from_route(rou_file)
        
        if len(valid_edges) < 2:
            print(f"❌ Not enough valid edges. Found: {len(valid_edges)}")
            return
            
        # Use first two edges as start/goal
        start_edge = valid_edges[0]
        goal_edge = valid_edges[1]
        
        print(f"📍 Start: {start_edge}")
        print(f"🎯 Goal: {goal_edge}")
        print(f"📊 Total valid edges: {len(valid_edges)}")
        
        # Start SUMO
        print(f"🚀 Starting SUMO with config: {config_file}")
        sumo_cmd = ["sumo", "-c", config_file, "--no-warnings", "--no-step-log"]
        traci.start(sumo_cmd)
        
        # Create a route first
        route_id = "test_route"
        try:
            traci.route.add(route_id, [start_edge])
        except traci.TraCIException:
            pass  # Route might already exist
        
        # Create agent
        agent = GreedyTravelTimeAgent(vehicle_id="test_agent")
        agent.initialize_episode("Rakab Ganj", start_edge, goal_edge)
        
        # Add vehicle with route ID
        try:
            traci.vehicle.add(agent.vehicle_id, routeID=route_id, typeID="passenger")
        except:
            # Fallback: try without typeID
            try:
                traci.vehicle.add(agent.vehicle_id, routeID=route_id)
            except:
                # Last resort: try with just the edge
                traci.vehicle.add(agent.vehicle_id, start_edge)
        
        # Run simulation
        step = 0
        max_steps = 500
        
        print("🔄 Running simulation...")
        
        while step < max_steps:
            traci.simulationStep()
            
            # Make greedy decision
            reroute_performed = agent.greedy_decision_step()
            
            # Update num_steps
            agent.stats['num_steps'] = step
            
            # Check if vehicle reached goal
            if agent.stats['success']:
                print(f"✅ Success! Reached goal in {step} steps")
                break
                
            # Check if vehicle is stuck
            if step > 100 and agent.stats['stuck_time'] > 50:
                print(f"⚠️ Vehicle stuck for {agent.stats['stuck_time']:.1f}s")
                break
                
            step += 1
            
            # Print progress every 50 steps
            if step % 50 == 0:
                try:
                    current_edge = traci.vehicle.getRoadID(agent.vehicle_id)
                    print(f"   Step {step}: Current edge = {current_edge}")
                except:
                    print(f"   Step {step}: Vehicle not found")
        
        # Get final stats
        final_stats = agent.get_stats()
        
        print(f"\n📊 Final Results:")
        print(f"   Success: {final_stats['success']}")
        print(f"   Total Time: {final_stats['total_time']:.1f}s")
        print(f"   Route Length: {final_stats['route_length']} edges")
        print(f"   Decisions Made: {final_stats['decision_count']}")
        print(f"   Final Edge: {final_stats['final_edge']}")
        print(f"   Stuck Time: {final_stats['stuck_time']:.1f}s")
        print(f"   Congestion Events: {final_stats['congestion_event']}")
        
        # Save results
        agent.save_stats("test_greedy_results.csv")
        
        traci.close()
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        try:
            traci.close()
        except:
            pass

if __name__ == "__main__":
    test_single_area() 