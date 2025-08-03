import os
import numpy as np
import traci
import gymnasium as gym
from gymnasium import spaces

# 👇 Local SUMO starter function (replaces failed import)
def start_sumo(net_file, route_file, use_gui, max_steps, port=8813):
    sumo_binary = "sumo-gui" if use_gui else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-n", net_file,
        "-r", route_file,
        "--step-length", "1",
        "--duration-log.disable", "true",
        "--no-warnings", "true",
        "--log", "NUL",
        # "--remote-port", str(port)
    ]
    traci.start(sumo_cmd)
    return traci

class EDQLRGEnv(gym.Env):
    def __init__(self, net_file, route_file, use_gui=False, max_steps=2000):
        super().__init__()
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.step_count = 0
        self.vehicle_id = "rl_agent0"
        
        # For Rakab-Ganj network, we'll use a subset of edges for routing
        # These are sample edges from the network analysis
        self.edge_list = [
            "-1049540403#0", "-1049540403#1", "-1049540404",
            "-1060463404#0", "-1060463404#1", "-1060463405#0",
            "-1060463405#1", "-1060463405#2", "-1060463406#0",
            "-1060463406#1", "-1068084352#0", "-1068084352#2",
            "-1068084352#3", "-1072954052#0", "-1072954052#1",
            "-1072954052#2", "-1072954053#0", "-1072954053#1"
        ]

        # Action space: discrete control (0-7 = combinations of speed & lane)
        self.action_space = spaces.Discrete(8)

        # Observation: [speed, lane_pos, lane_index, edge_progress]
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)

        self.traci = None

    def reset(self, seed=None, options=None):
        if self.traci:
            self.traci.close()

        self.step_count = 0
        self.traci = start_sumo(
            net_file=self.net_file,
            route_file=self.route_file,
            use_gui=self.use_gui,
            max_steps=self.max_steps
        )

        print("Starting SUMO for Rakab-Ganj network...")

        # For Rakab-Ganj, we'll use a simpler approach - just use the first available edge
        # and let the vehicle stay on that edge for training
        available_edges = self.traci.edge.getIDList()
        
        # Find a simple edge that allows passenger vehicles
        simple_edge = None
        for edge in available_edges[:50]:  # Check first 50 edges
            try:
                lanes = self.traci.edge.getLaneNumber(edge)
                if lanes > 0:
                    simple_edge = edge
                    break
            except:
                continue
        
        if simple_edge is None:
            print("Warning: No suitable edge found, using first edge")
            simple_edge = available_edges[0]
        
        # Create a simple route with just one edge
        self.traci.route.add("r0", [simple_edge])
        print(f"Defining route r0 with edge: {simple_edge}")

        # Add vehicle with proper typeID
        try:
            self.traci.vehicle.add(self.vehicle_id, "r0", typeID="passenger")
        except:
            # If passenger type fails, try default type
            try:
                self.traci.vehicle.add(self.vehicle_id, "r0", typeID="DEFAULT_VEHTYPE")
            except:
                # Last resort - add without specifying type
                self.traci.vehicle.add(self.vehicle_id, "r0")
        
        # Advance simulation by one step to properly place the vehicle
        self.traci.simulationStep()
        
        print("Vehicle position:", self.traci.vehicle.getPosition(self.vehicle_id))
        print(f"Adding {self.vehicle_id} manually...")
        
        # Check vehicle status after placement
        if self.vehicle_id in self.traci.vehicle.getIDList():
            current_edge = self.traci.vehicle.getRoadID(self.vehicle_id)
            current_speed = self.traci.vehicle.getSpeed(self.vehicle_id)
            print(f"Vehicle successfully placed on edge: {current_edge}, speed: {current_speed:.2f}")
        else:
            print("ERROR: Vehicle not found in simulation after placement!")

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # Check if vehicle is still in the simulation
        if self.vehicle_id not in self.traci.vehicle.getIDList():
            # Vehicle completed its route - this is normal and expected
            return self._get_obs(), 0.0, True  # Neutral reward for completion

        try:
            speed = 5 + (action % 4) * 5  # 5, 10, 15, 20
            lane = (action // 4) % 2     # lane 0 or 1

            # Only try to change lane and speed if vehicle still exists
            if self.vehicle_id in self.traci.vehicle.getIDList():
                # Get current vehicle state before applying actions
                current_edge = self.traci.vehicle.getRoadID(self.vehicle_id)
                current_speed = self.traci.vehicle.getSpeed(self.vehicle_id)
                
                # Debug: Print vehicle status every 100 steps (less frequent for larger network)
                if self.step_count % 100 == 0:
                    print(f"Step {self.step_count}: Vehicle on edge {current_edge}, speed {current_speed:.2f}")
                
                self.traci.vehicle.changeLane(self.vehicle_id, lane, 25)
                self.traci.vehicle.setSpeed(self.vehicle_id, speed)
                    
        except Exception as e:
            print(f"[ERROR] Failed to apply action {action}: {e}")

        # Advance simulation
        self.traci.simulationStep()

        obs = self._get_obs()
        reward = self._get_reward()
        done = self._is_done()

        return obs, reward, done

    def _get_obs(self):
        try:
            speed = self.traci.vehicle.getSpeed(self.vehicle_id)
            lane_pos = self.traci.vehicle.getLanePosition(self.vehicle_id)
            lane_index = self.traci.vehicle.getLaneIndex(self.vehicle_id)
            
            # Add edge progress as additional observation
            edge_progress = 0.0
            try:
                edge_length = self.traci.edge.getLength(self.traci.vehicle.getRoadID(self.vehicle_id))
                edge_progress = lane_pos / edge_length if edge_length > 0 else 0.0
            except:
                edge_progress = 0.0
                
            return np.array([speed, lane_pos, lane_index, edge_progress], dtype=np.float32)
        except traci.TraCIException:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _get_reward(self):
        try:
            # If vehicle has reached the destination (completed route)
            if self.vehicle_id not in self.traci.vehicle.getIDList():
                return 100.0  # Large positive reward for completing the route

            # Get mean waiting time for the vehicle
            mwt = self.traci.vehicle.getWaitingTime(self.vehicle_id)
            
            # Reward based on waiting time thresholds
            if mwt <= 50:
                return -5
            elif 50 < mwt <= 100:
                return -10
            elif 100 < mwt <= 150:
                return -20
            elif 150 < mwt <= 200:
                return -30
            else:
                return -100  # Severe penalty for very long waiting times
                
        except traci.TraCIException:
            # Vehicle has disappeared or other TraCI error
            return -100

    def _is_done(self):
        if self.step_count >= self.max_steps:
            return True
        if self.vehicle_id not in self.traci.vehicle.getIDList():
            # Vehicle completed its route - this is a successful completion
            return True
        return False

    def close(self):
        if self.traci:
            self.traci.close()
            self.traci = None
        print("Done. Closing SUMO.") 