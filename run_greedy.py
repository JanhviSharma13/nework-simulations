import traci
import os
import sys
import random
import subprocess
from datetime import datetime
from greedy_routing_agent import GreedyTravelTimeAgent
import xml.etree.ElementTree as ET

class GreedyRoutingTester:
    """
    Test the Greedy Travel-Time Routing Agent across all Delhi areas
    """
    
    def __init__(self):
        self.areas = {
            'rakabganj': {
                'config': 'Rakab-Ganj/rg.sumocfg',
                'rou_file': 'Rakab-Ganj/EDQL-RG/EDQL-RakabGanj/rakabganj_simple.rou.xml',
                'net_file': 'Rakab-Ganj/EDQL-RG/EDQL-RakabGanj/rakabganj_cleaned.net.xml',
                'name': 'Rakab Ganj'
            },
            'safdarjung': {
                'config': 'Safdarjung/SE.sumocfg', 
                'rou_file': 'Safdarjung/safdarjung_simple.rou.xml',
                'net_file': 'Safdarjung/safdarjung_cleaned.net.xml',
                'name': 'Safdarjung'
            },
            'cp2': {
                'config': 'CP2/cp2.sumocfg',
                'rou_file': 'Rakab-Ganj/EDQL-RG/EDQL-CP2/cp2_simple.rou.xml', 
                'net_file': 'Rakab-Ganj/EDQL-RG/EDQL-CP2/cp2_cleaned.net.xml',
                'name': 'CP2'
            },
            'chandnichowk': {
                'config': 'Chandni-Chowk/ch.sumocfg',
                'rou_file': 'Chandni-Chowk/chandnichowk_simple.rou.xml',
                'net_file': 'Chandni-Chowk/chandnichowk_cleaned.net.xml',
                'name': 'Chandni Chowk'
            }
        }
        
        self.results_file = f"greedy_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
    def get_valid_edges_from_network(self, net_file):
        """Get valid edges from network file"""
        try:
            # Start TraCI to get edge information from network
            sumo_cmd = ["sumo", "-n", net_file, "--no-warnings", "--no-step-log"]
            traci.start(sumo_cmd)
            
            # Get all edges
            all_edges = traci.edge.getIDList()
            
            # Filter out internal edges (containing ':')
            valid_edges = [edge for edge in all_edges if ':' not in edge]
            
            traci.close()
            return valid_edges
            
        except Exception as e:
            print(f"❌ Error getting valid edges from network file: {e}")
            return []
    
    def run_single_test(self, area_key, start_edge, goal_edge, max_steps=1000):
        """
        Run a single test with the greedy agent
        
        Args:
            area_key (str): Area identifier
            start_edge (str): Starting edge
            goal_edge (str): Goal edge
            max_steps (int): Maximum simulation steps
            
        Returns:
            dict: Test results
        """
        area_config = self.areas[area_key]
        net_file = area_config['net_file']
        area_name = area_config['name']
        
        try:
            # Start SUMO with network file directly
            sumo_cmd = ["sumo", "-n", net_file, "--no-warnings", "--no-step-log"]
            traci.start(sumo_cmd)
            
            # Create a simple route with just the start edge
            route_id = f"test_route_{random.randint(1000, 9999)}"
            try:
                traci.route.add(route_id, [start_edge])
            except traci.TraCIException:
                pass  # Route might already exist
            
            # Create agent
            agent = GreedyTravelTimeAgent(vehicle_id="greedy_agent")
            agent.initialize_episode(area_name, start_edge, goal_edge)
            
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
            
            while step < max_steps:
                traci.simulationStep()
                
                # Make greedy decision
                reroute_performed = agent.greedy_decision_step()
                
                # Update num_steps
                agent.stats['num_steps'] = step
                
                # Check if vehicle reached goal
                if agent.stats['success']:
                    break
                    
                # Check if vehicle is stuck
                if step > 100 and agent.stats['stuck_time'] > 50:
                    break
                    
                step += 1
            
            # Get final stats
            final_stats = agent.get_stats()
            
            # Save results
            agent.save_stats(self.results_file)
            
            traci.close()
            return final_stats
            
        except Exception as e:
            print(f"❌ Error in test: {e}")
            try:
                traci.close()
            except:
                pass
            return None
    
    def run_area_tests(self, area_key, num_tests=10):
        """
        Run multiple tests for a specific area
        
        Args:
            area_key (str): Area identifier
            num_tests (int): Number of tests to run
        """
        area_config = self.areas[area_key]
        area_name = area_config['name']
        rou_file = area_config['rou_file']
        
        print(f"\n🎯 Running {num_tests} tests for {area_name}")
        print("=" * 60)
        
        # Get valid edges
        valid_edges = self.get_valid_edges_from_network(area_config['net_file'])
        
        if len(valid_edges) < 2:
            print(f"❌ Not enough valid edges for {area_name}. Found: {len(valid_edges)}")
            return
        
        successful_tests = 0
        total_time = 0.0
        total_decisions = 0
        
        for test_num in range(1, num_tests + 1):
            print(f"\n📋 Test {test_num}/{num_tests}")
            print("-" * 30)
            
            # Select random start/goal edges
            start_edge = random.choice(valid_edges)
            goal_edge = random.choice(valid_edges)
            
            # Ensure start and goal are different
            while goal_edge == start_edge:
                goal_edge = random.choice(valid_edges)
            
            print(f"🚀 Testing {area_name}: {start_edge} → {goal_edge}")
            
            # Run test
            result = self.run_single_test(area_key, start_edge, goal_edge)
            
            if result:
                successful_tests += 1
                total_time += result['total_time']
                total_decisions += result['decision_count']
        
        # Print summary
        success_rate = (successful_tests / num_tests) * 100
        avg_time = total_time / successful_tests if successful_tests > 0 else 0
        avg_decisions = total_decisions / successful_tests if successful_tests > 0 else 0
        
        print(f"\n📊 {area_name} Summary:")
        print(f"   Success Rate: {success_rate:.1f}% ({successful_tests}/{num_tests})")
        print(f"   Avg Time: {avg_time:.1f}s")
        print(f"   Avg Decisions: {avg_decisions:.1f}")
    
    def run_all_areas(self, tests_per_area=5):
        """Run tests for all areas"""
        print("🚀 Greedy Travel-Time Routing Agent Testing")
        print("=" * 80)
        print(f"📊 Results will be saved to: {self.results_file}")
        
        for area_key in self.areas.keys():
            self.run_area_tests(area_key, tests_per_area)
        
        print(f"\n✅ All tests completed! Results saved to: {self.results_file}")
        
        # Check if results file exists
        if os.path.exists(self.results_file):
            print(f"✅ Results file found: {self.results_file}")
        else:
            print(f"❌ Results file not found: {self.results_file}")
    
    def analyze_results(self):
        """Analyze the results from the CSV file"""
        if not os.path.exists(self.results_file):
            print(f"❌ Results file not found: {self.results_file}")
            return
        
        print(f"\n📊 Analyzing results from: {self.results_file}")
        print("=" * 50)
        
        # Read and analyze CSV
        import pandas as pd
        try:
            df = pd.read_csv(self.results_file)
            
            print(f"📈 Total tests: {len(df)}")
            print(f"✅ Success rate: {(df['success'].sum() / len(df)) * 100:.1f}%")
            print(f"⏱️ Avg travel time: {df['total_time'].mean():.1f}s")
            print(f"🛣️ Avg route length: {df['route_length'].mean():.1f} edges")
            print(f"🎯 Avg decisions: {df['decision_count'].mean():.1f}")
            
            # Area-wise analysis
            print(f"\n📊 Area-wise Analysis:")
            for area in df['map_name'].unique():
                area_df = df[df['map_name'] == area]
                success_rate = (area_df['success'].sum() / len(area_df)) * 100
                avg_time = area_df['total_time'].mean()
                print(f"   {area}: {success_rate:.1f}% success, {avg_time:.1f}s avg time")
                
        except Exception as e:
            print(f"❌ Error analyzing results: {e}")

def main():
    """Main function"""
    tester = GreedyRoutingTester()
    
    # Phase 1: Test on Rakab Ganj first
    print("🎯 Phase 1: Testing on Rakab Ganj")
    print("=" * 60)
    tester.run_area_tests('rakabganj', 3)
    
    # Phase 2: Test on all areas
    print("\n🎯 Phase 2: Testing on all areas")
    tester.run_all_areas(3)
    
    # Phase 3: Analyze results
    tester.analyze_results()

if __name__ == "__main__":
    main() 