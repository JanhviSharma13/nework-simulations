import os
import random
import traci
import sumolib


class SumoEnvironment:
    def __init__(self, sumo_binary, config_file, net_file, max_steps=500, gui=False):
        print("[env] 🔧 __init__() called")
        self.sumo_binary = sumo_binary
        self.config_file = config_file
        self.net_file = net_file
        self.max_steps = max_steps
        self.gui = gui
        self.step_count = 0
        self.vehicle_id = "rl_agent"

        print("[env] 🔌 Reading network...")
        self.net = sumolib.net.readNet(net_file)
        self.all_edges = [e.getID() for e in self.net.getEdges() if not e.isSpecial()]
        print(f"[env] ✅ Loaded network with {len(self.all_edges)} usable edges")

    def _start_sumo(self):
        sumo_cmd = [self.sumo_binary, "-c", self.config_file]
        if self.gui:
            sumo_cmd += ["--start", "--quit-on-end"]  # 👈 added --quit-on-end
        traci.start(sumo_cmd)
        print("[traci] ✅ SUMO started via TraCI.")

    def _get_valid_edges(self):
        return self.all_edges

    def get_valid_actions(self, current_edge):
        """Return all directly reachable edges from current_edge, skipping internal ones."""
        if current_edge.startswith(":"):
            print(f"[get_valid_actions] Skipping internal edge: {current_edge}")
            return []

        try:
            edge_obj = self.net.getEdge(current_edge)
            outgoing = edge_obj.getOutgoing()
            return [e.getID() for e in outgoing if not e.getID().startswith(":")]
        except KeyError:
            print(f"[get_valid_actions] Edge not found in network: {current_edge}")
            return []

    def reset(self):
        print("[reset] Starting SUMO via TraCI...")
        if traci.isLoaded():
            traci.close()

        self._start_sumo()

        self.step_count = 0
        self.vehicle_id = "rl_agent"
        attempts = 0
        max_attempts = 100

        edge_list = self._get_valid_edges()
        print(f"[reset] Total candidate edges: {len(edge_list)}")

        while attempts < max_attempts:
            from_edge = random.choice(edge_list)
            actions = self.get_valid_actions(from_edge)
            if not actions:
                attempts += 1
                continue
            to_edge = random.choice(actions)

            try:
                route_id = "route_rl"
                traci.route.add(route_id, [from_edge, to_edge])
                traci.vehicle.add(self.vehicle_id, route_id, typeID="car")
                traci.vehicle.setColor(self.vehicle_id, (255, 0, 0))
                print(f"[reset] ✅ Vehicle added from {from_edge} → {to_edge}")

                traci.simulationStep()
                return from_edge
            except Exception as e:
                print(f"[reset] ❌ Attempt {attempts + 1} failed: {e}")
                attempts += 1

        print("[reset] ❌ Failed to place vehicle after multiple attempts.")
        return None

    def step(self, action):
        try:
            traci.vehicle.changeTarget(self.vehicle_id, action)
            print(f"[step] Vehicle target changed to {action}")
        except traci.TraCIException as e:
            print(f"[step] ❌ Failed to change target: {e}")
            self.close()
            return None, -100, True, {}

        traci.simulationStep()
        self.step_count += 1

        if self.step_count >= self.max_steps:
            print("[step] Max steps reached. Ending episode.")
            self.close()
            return None, 0, True, {}

        try:
            pos = traci.vehicle.getRoadID(self.vehicle_id)

            if pos not in self.all_edges:
                print(f"[step] ⚠️ Vehicle moved to internal edge {pos}. Ending episode.")
                self.close()
                return None, -100, True, {}

            reward = -1
            done = pos == action
            print(f"[step] Vehicle is on {pos}, target is {action}, done={done}")

            if done:
                self.close()
            return pos, reward, done, {}

        except traci.TraCIException as e:
            print(f"[step] ❌ Vehicle vanished or unreachable: {e}")
            self.close()
            return None, -100, True, {}

    def close(self):
        if traci.isLoaded():
            print("[close] 🔒 Closing TraCI and shutting SUMO down.")
            traci.close()
