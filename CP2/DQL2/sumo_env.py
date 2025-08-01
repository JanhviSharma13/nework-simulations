import os
import sys
import numpy as np
import traci
import sumolib
import gymnasium as gym
from gymnasium import spaces

# Set SUMO_HOME
if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = r"C:\Users\Janhvi\Downloads\sumo\sumo-1.23.1"
tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

# SUMO binaries
SUMO_BINARY = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")
SUMO_GUI_BINARY = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui")


class SumoEnv(gym.Env):
    def __init__(self, sumo_cfg, use_gui=True, max_steps=1000):
        print("[env] 🔧 __init__() called")
        self.sumo_cfg = sumo_cfg
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.step_count = 0

        self.vehicle_id = "rl_agent0"

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)

        self.edge_ids = self._load_edges()
        self.action_space = spaces.Discrete(len(self.edge_ids))
        self.edge_id_map = {i: eid for i, eid in enumerate(self.edge_ids)}
        self.reverse_edge_map = {eid: i for i, eid in enumerate(self.edge_ids)}

        self.sumo_cmd = [SUMO_GUI_BINARY if self.use_gui else SUMO_BINARY, "-c", self.sumo_cfg, "--start"]

    def _load_edges(self):
        net_file = self.sumo_cfg.replace(".sumocfg", ".net.xml")
        net = sumolib.net.readNet(net_file)
        return [edge.getID() for edge in net.getEdges() if not edge.getID().startswith(":")]

    def reset(self, seed=None, options=None):
        print("[env] 🔄 reset() called")
        self.close()

        traci.start(self.sumo_cmd)
        self.step_count = 0

        traci.simulationStep()

        # ✅ Add the vehicle at start of simulation
        vehicle_ids = traci.vehicle.getIDList()
        if self.vehicle_id not in vehicle_ids:
         raise RuntimeError(f"Vehicle '{self.vehicle_id}' not found in route file. Ensure it exists in rl_agent.rou.xml.")
        print(f"[env] 🚗 Vehicle '{self.vehicle_id}' entered simulation.")

        return self._get_state(), {}

    def _get_state(self):
        edge = traci.vehicle.getRoadID(self.vehicle_id)
        edge_idx = self.reverse_edge_map.get(edge, -1)
        speed = traci.vehicle.getSpeed(self.vehicle_id)
        try:
            route = traci.vehicle.getRoute(self.vehicle_id)
            destination = route[-1]
            distance = traci.vehicle.getDrivingDistance(self.vehicle_id, destination, 0)
        except:
            distance = 0.0
        return (edge_idx, 0, speed, distance)

    def step(self, action_idx):
        self.step_count += 1

        try:
            edge_id = self.edge_id_map[action_idx]
        except KeyError:
            print(f"❌ Invalid action index: {action_idx}")
            return self._get_state(), -100.0, True, {}

        try:
            self._apply_action(edge_id)
        except Exception as e:
            print(f"❌ Error applying action: {e}")
            return self._get_state(), -10.0, False, {}

        traci.simulationStep()

        state = self._get_state()
        done = self._is_done(state)
        reward = self._get_reward(state, done)

        return state, reward, done, {}

    def _apply_action(self, edge_id):
        current_edge = traci.vehicle.getRoadID(self.vehicle_id)

        if edge_id == current_edge:
            print(f"⚠️ Already on edge {edge_id}, skipping.")
            return

        try:
           route = traci.simulation.findRoute(current_edge, edge_id).edges
        except traci.exceptions.TraCIException:
            print(f"❌ findRoute failed: no connection from {current_edge} to {edge_id}")
            raise

        clean_route = tuple(e for e in route if not e.startswith(":"))
        if not clean_route or clean_route[-1] != edge_id:
            raise ValueError(f"❌ Invalid route to {edge_id}. No connection or not reachable.")

        new_route = tuple(traci.vehicle.getRoute(self.vehicle_id)) + clean_route
        traci.vehicle.setRoute(self.vehicle_id, new_route)
        print(f"✅ Route updated with {edge_id} → {clean_route}")

    def _get_reward(self, state, done):
        _, _, _, distance = state
        if done:
            return 100.0
        return max(0.0, 2500 - distance)

    def _is_done(self, state):
        _, _, speed, distance = state
        return (
            self.step_count >= self.max_steps
            or distance <= 5.0
            or speed < 0.1
            or self.vehicle_id not in traci.vehicle.getIDList()
        )

    def close(self):
        if traci.isLoaded():
            traci.close()


if __name__ == "__main__":
    env = SumoEnv(
        sumo_cfg="cp2.sumocfg",
        use_gui=True,
        max_steps=1000
    )

    state, _ = env.reset()
    print("Initial state:", state)

    for step in range(env.max_steps):
        action_idx = np.random.randint(env.action_space.n)
        state, reward, done, _ = env.step(action_idx)
        print(f"Step {step}: State = {state}, Reward = {reward}")
        if done:
            print("✅ Episode finished.")
            break
