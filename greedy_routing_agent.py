import traci
import traci.constants as tc
import numpy as np
import csv
import os
import time
from datetime import datetime
from collections import defaultdict
import xml.etree.ElementTree as ET

class GreedyTravelTimeAgent:
    """
    Greedy Travel-Time Routing Agent
    
    Selects the next edge at each decision point based on minimum estimated 
    travel time to the destination using SUMO and TraCI.
    """
    
    def __init__(self, vehicle_id="greedy_agent", reroute_threshold=50):
        self.vehicle_id = vehicle_id
        self.reroute_threshold = reroute_threshold  # meters from edge end to trigger reroute
        self.stats = {
            'episode': 1,
            'map_name': '',
            'start_edge': '',
            'goal_edge': '',
            'success': 0,
            'total_time': 0.0,
            'route_length': 0,
            'total_distance': 0.0,
            'decision_count': 0,
            'final_edge': '',
            'stuck_time': 0.0,
            'congestion_event': 0,
            'route_edges': [],
            'reroute_timestamps': [],
            'decision_latency_avg': 0.0,
            'internal_edges_skipped': 0,
            # Add missing metrics from user specification
            'num_steps': 0,
            'num_edges_visited': 0,
            'arrival_success': 0
        }
        self.start_time = None
        self.last_position = None
        self.stuck_start_time = None
        self.decision_times = []
        self.visited_edges = set()  # Track unique edges visited
        
    def initialize_episode(self, map_name, start_edge, goal_edge):
        """Initialize a new episode with start and goal edges"""
        self.stats['map_name'] = map_name
        self.stats['start_edge'] = start_edge
        self.stats['goal_edge'] = goal_edge
        self.stats['route_edges'] = [start_edge]
        self.stats['reroute_timestamps'] = []
        self.stats['decision_count'] = 0
        self.stats['internal_edges_skipped'] = 0
        self.stats['num_steps'] = 0
        self.stats['num_edges_visited'] = 0
        self.stats['arrival_success'] = 0
        self.decision_times = []
        self.start_time = traci.simulation.getTime()
        self.last_position = None
        self.stuck_start_time = None
        self.visited_edges = {start_edge}  # Initialize with start edge
        
        print(f"🚀 Starting greedy routing: {start_edge} → {goal_edge}")
        
    def greedy_decision_step(self):
        """
        Make a greedy routing decision based on current position and goal
        
        Returns:
            bool: True if reroute was performed, False otherwise
        """
        if not traci.vehicle.getIDList():
            return False
            
        if self.vehicle_id not in traci.vehicle.getIDList():
            return False
            
        current_edge = traci.vehicle.getRoadID(self.vehicle_id)
        current_pos = traci.vehicle.getLanePosition(self.vehicle_id)
        
        try:
            edge_length = traci.lane.getLength(current_edge + "_0")
        except:
            # If lane doesn't exist, use a default length
            edge_length = 100
            
        # Check if we're near the end of current edge (decision point)
        if edge_length - current_pos > self.reroute_threshold:
            return False
            
        # Record decision start time
        decision_start = time.time()
        
        # Get current position and goal
        goal_edge = self.stats['goal_edge']
        
        # Check if we've reached the goal
        if current_edge == goal_edge:
            self.stats['success'] = 1
            self.stats['final_edge'] = current_edge
            print(f"✅ Reached goal: {goal_edge}")
            return False
            
        # Find best next edge using greedy approach
        best_next_edge = self._find_best_next_edge(current_edge, goal_edge)
        
        if best_next_edge and best_next_edge != current_edge:
            # Perform reroute
            self._perform_reroute(current_edge, best_next_edge, goal_edge)
            
            # Track visited edges
            self.visited_edges.add(best_next_edge)
            
            # Record decision metrics
            decision_time = time.time() - decision_start
            self.decision_times.append(decision_time)
            self.stats['decision_count'] += 1
            self.stats['reroute_timestamps'].append(traci.simulation.getTime())
            
            print(f"🔄 Reroute at {current_edge} → {best_next_edge} (took {decision_time:.3f}s)")
            return True
            
        return False
    
    def _find_best_next_edge(self, current_edge, goal_edge):
        """
        Find the best next edge using greedy travel-time estimation
        
        Args:
            current_edge (str): Current edge ID
            goal_edge (str): Goal edge ID
            
        Returns:
            str: Best next edge ID
        """
        try:
            # Get outgoing edges from current edge
            outgoing_edges = self._get_outgoing_edges(current_edge)
            
            if not outgoing_edges:
                return None
                
            best_edge = None
            best_time = float('inf')
            
            for next_edge in outgoing_edges:
                # Skip internal edges (containing ':')
                if ':' in next_edge:
                    self.stats['internal_edges_skipped'] += 1
                    continue
                    
                # Estimate travel time to goal from this edge
                estimated_time = self._estimate_travel_time_to_goal(next_edge, goal_edge)
                
                if estimated_time < best_time:
                    best_time = estimated_time
                    best_edge = next_edge
                    
            return best_edge
            
        except Exception as e:
            print(f"❌ Error finding best next edge: {e}")
            return None
    
    def _get_outgoing_edges(self, edge_id):
        """Get all outgoing edges from the given edge"""
        try:
            # Get all edges
            all_edges = traci.edge.getIDList()
            
            # For simplicity, return all edges except the current one
            # In a real implementation, you'd use proper network topology
            outgoing = []
            for other_edge in all_edges:
                if other_edge != edge_id and ':' not in other_edge:
                    # Simple heuristic: assume edges are connected if they share common patterns
                    if self._edges_are_connected(edge_id, other_edge):
                        outgoing.append(other_edge)
                        
            return outgoing
            
        except Exception as e:
            print(f"❌ Error getting outgoing edges: {e}")
            return []
    
    def _edges_are_connected(self, edge1, edge2):
        """
        Check if two edges are connected (simplified implementation)
        In a real scenario, you'd use proper graph analysis
        """
        # Simplified: assume edges are connected if they share common characters
        # This is a placeholder - in reality you'd use proper graph analysis
        common_chars = len(set(edge1) & set(edge2))
        return common_chars > 0 and len(edge1) > 3 and len(edge2) > 3
    
    def _estimate_travel_time_to_goal(self, from_edge, goal_edge):
        """
        Estimate travel time from current edge to goal
        
        Args:
            from_edge (str): Starting edge
            goal_edge (str): Goal edge
            
        Returns:
            float: Estimated travel time in seconds
        """
        try:
            # Get current travel time for the edge
            current_time = traci.edge.getTraveltime(from_edge)
            
            # Get edge length
            try:
                edge_length = traci.lane.getLength(from_edge + "_0")
            except:
                edge_length = 100  # Default length
                
            # Get current speed limit
            try:
                max_speed = traci.lane.getMaxSpeed(from_edge + "_0")
            except:
                max_speed = 13.89  # Default speed (50 km/h)
                
            # Estimate based on current conditions
            if current_time > 0:
                estimated_time = current_time
            else:
                # Fallback to speed-based estimation
                estimated_time = edge_length / max_speed if max_speed > 0 else 60
                
            # Add congestion penalty if edge is congested
            try:
                vehicle_count = traci.edge.getLastStepVehicleNumber(from_edge)
                if vehicle_count > 5:  # Congestion threshold
                    estimated_time *= 1.5
                    self.stats['congestion_event'] = 1
            except:
                pass
                
            return estimated_time
            
        except Exception as e:
            print(f"❌ Error estimating travel time: {e}")
            return 60.0  # Default fallback
    
    def _perform_reroute(self, current_edge, next_edge, goal_edge):
        """Perform the actual reroute using TraCI"""
        try:
            # Create new route: current → next → goal
            new_route = [current_edge, next_edge]
            
            # Try to find a path to goal from next_edge
            try:
                # Use TraCI's built-in routing to find path to goal
                route_to_goal = traci.simulation.findRoute(next_edge, goal_edge)
                if route_to_goal and route_to_goal.edges:
                    new_route.extend(route_to_goal.edges)
            except:
                # Fallback: just add goal edge
                new_route.append(goal_edge)
                
            # Update vehicle route
            traci.vehicle.setRoute(self.vehicle_id, new_route)
            
            # Update stats
            self.stats['route_edges'].extend(new_route[1:])  # Skip current edge
            
        except Exception as e:
            print(f"❌ Error performing reroute: {e}")
    
    def update_stats(self):
        """Update episode statistics"""
        if not traci.vehicle.getIDList() or self.vehicle_id not in traci.vehicle.getIDList():
            return
            
        current_time = traci.simulation.getTime()
        
        # Update total time
        if self.start_time:
            self.stats['total_time'] = current_time - self.start_time
            
        # Update route length
        self.stats['route_length'] = len(self.stats['route_edges'])
        
        # Update total distance
        try:
            total_distance = 0
            for edge in self.stats['route_edges']:
                try:
                    total_distance += traci.lane.getLength(edge + "_0")
                except:
                    total_distance += 100  # Default length
            self.stats['total_distance'] = total_distance
        except:
            pass
            
        # Update num_edges_visited (unique edges)
        self.stats['num_edges_visited'] = len(self.visited_edges)
        
        # Update arrival_success (same as success)
        self.stats['arrival_success'] = self.stats['success']
        
        # Check for stuck vehicle
        try:
            current_pos = traci.vehicle.getLanePosition(self.vehicle_id)
            if self.last_position is not None:
                if abs(current_pos - self.last_position) < 1.0:  # No progress
                    if self.stuck_start_time is None:
                        self.stuck_start_time = current_time
                else:
                    if self.stuck_start_time is not None:
                        self.stats['stuck_time'] += current_time - self.stuck_start_time
                        self.stuck_start_time = None
                        
            self.last_position = current_pos
        except:
            pass
        
        # Update final edge
        if traci.vehicle.getIDList():
            try:
                self.stats['final_edge'] = traci.vehicle.getRoadID(self.vehicle_id)
            except:
                pass
            
        # Calculate average decision latency
        if self.decision_times:
            self.stats['decision_latency_avg'] = np.mean(self.decision_times)
    
    def get_stats(self):
        """Get current statistics"""
        self.update_stats()
        return self.stats.copy()
    
    def save_stats(self, filename):
        """Save statistics to CSV file"""
        stats = self.get_stats()
        
        # Prepare CSV data
        csv_data = {
            'episode': stats['episode'],
            'map_name': stats['map_name'],
            'start_edge': stats['start_edge'],
            'goal_edge': stats['goal_edge'],
            'success': stats['success'],
            'total_time': stats['total_time'],
            'route_length': stats['route_length'],
            'total_distance': stats['total_distance'],
            'decision_count': stats['decision_count'],
            'final_edge': stats['final_edge'],
            'stuck_time': stats['stuck_time'],
            'congestion_event': stats['congestion_event'],
            'route_edges': ','.join(stats['route_edges']),
            'reroute_timestamps': ','.join(map(str, stats['reroute_timestamps'])),
            'decision_latency_avg': stats['decision_latency_avg'],
            'internal_edges_skipped': stats['internal_edges_skipped'],
            'num_steps': stats['num_steps'],
            'num_edges_visited': stats['num_edges_visited'],
            'arrival_success': stats['arrival_success']
        }
        
        # Write to CSV
        file_exists = os.path.exists(filename)
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = csv_data.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
                
            writer.writerow(csv_data)
            
        print(f"📊 Stats saved to: {filename}") 