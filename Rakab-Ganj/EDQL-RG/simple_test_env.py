import os
import numpy as np
import traci
import gymnasium as gym
from gymnasium import spaces

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
    ]
    traci.start(sumo_cmd)
    return traci

class SimpleRGEnv(gym.Env):
    def __init__(self, net_file, route_file, use_gui=False, max_steps=1000):
        super().__init__()
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.step_count = 0
        self.vehicle_id = "rl_agent0"

        # Action space: discrete control (0-7 = combinations of speed & lane)
        self.action_space = spaces.Discrete(8)

        # Observation: [speed, lane_pos, lane_index]
        self.observation_space = spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32)

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

        # Get available edges
        available_edges = self.traci.edge.getIDList()
        print(f"Total edges available: {len(available_edges)}")
        
        # Find an edge that allows vehicles
        suitable_edge = None
        for edge in available_edges[:100]:  # Check first 100 edges
            try:
                # Check if this edge has any lanes
                lane_count = self.traci.edge.getLaneNumber(edge)
                if lane_count > 0:
                    # Check if any lane allows vehicles
                    for lane_idx in range(lane_count):
                        lane_id = f"{edge}_{lane_idx}"
                        try:
                            # Try to get lane info
                            lane_length = self.traci.lane.getLength(lane_id)
                            if lane_length > 0:
                                suitable_edge = edge
                                break
                        except:
                            continue
                    if suitable_edge:
                        break
            except:
                continue
        
        if suitable_edge is None:
            print("No suitable edge found, using first edge")
            suitable_edge = available_edges[0]
        
        print(f"Using edge: {suitable_edge}")
        
        # Create a simple route with just this edge
        self.traci.route.add("r0", [suitable_edge])
        
        # Try to add vehicle without specifying type
        try:
            self.traci.vehicle.add(self.vehicle_id, "r0")
            print("Vehicle added successfully")
        except Exception as e:
            print(f"Failed to add vehicle: {e}")
            # Return a default observation
            return np.array([0.0, 0.0, 0.0], dtype=np.float32), {}

        # Advance simulation by one step to properly place the vehicle
        self.traci.simulationStep()
        
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
                
                # Debug: Print vehicle status every 50 steps
                if self.step_count % 50 == 0:
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
            return np.array([speed, lane_pos, lane_index], dtype=np.float32)
        except traci.TraCIException:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

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

def test_simple_env():
    """Test the simple Rakab-Ganj environment"""
    print("Testing Simple Rakab-Ganj Environment...")
    
    try:
        # Initialize environment
        env = SimpleRGEnv(
            net_file="rg.net.xml",
            route_file="rg.rou.xml",
            use_gui=False,
            max_steps=100
        )
        
        print("Environment initialized successfully!")
        
        # Test reset
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial observation: {obs}")
        
        # Test a few steps
        for step in range(10):
            # Random action
            action = np.random.randint(0, 8)
            obs, reward, done = env.step(action)
            
            print(f"Step {step + 1}:")
            print(f"  Action: {action}")
            print(f"  Observation: {obs}")
            print(f"  Reward: {reward}")
            print(f"  Done: {done}")
            
            if done:
                print("Episode completed!")
                break
        
        env.close()
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_env() 