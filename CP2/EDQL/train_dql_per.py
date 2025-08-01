import os
import random
import traci
import sumolib
from typing import Optional, List, Tuple, Dict
import time

class SumoEnvironment:
    def __init__(self, sumo_binary: str, config_file: str, net_file: str, 
                 max_steps: int = 500, gui: bool = False):
        if not os.path.exists(net_file):
            raise FileNotFoundError(f"Network file not found: {net_file}")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")

        self.sumo_binary = sumo_binary
        self.config_file = config_file
        self.net_file = net_file
        self.max_steps = max_steps
        self.gui = gui
        self.step_count = 0
        self.vehicle_id = "rl_agent"
        self.destination_edge = None
        self.start_edge = None

        self.net = sumolib.net.readNet(net_file)
        self.all_edges = [e.getID() for e in self.net.getEdges() if not e.isSpecial()]
        self._analyze_network()
        self._init_traci()

    def _analyze_network(self):
        print("\n=== NETWORK ANALYSIS ===")
        print(f"Total edges: {len(self.all_edges)}")
        connected_pairs = 0
        test_edges = self.all_edges[:min(10, len(self.all_edges))]
        for start in test_edges:
            for dest in test_edges:
                if start == dest:
                    continue
                try:
                    path, _ = self.net.getShortestPath(start, dest)
                    if path:
                        connected_pairs += 1
                        print(f"  {start} → {dest}: {len(path)} edges")
                except Exception:
                    continue
        print(f"Connected pairs: {connected_pairs}/100")
        print("=======================\n")

    def _init_traci(self):
        if traci.isLoaded():
            traci.close()

        sumo_cmd = [
            self.sumo_binary,
            "-c", self.config_file,
            "--no-step-log",
            "--no-warnings",
            "--duration-log.disable",
            "--quit-on-end"
        ]

        if self.gui:
            sumo_cmd.append("--start")

        try:
            traci.start(sumo_cmd)
            print("[SUMO] TraCI started successfully")
        except Exception as e:
            print(f"[SUMO] Failed to start: {e}")
            raise RuntimeError("SUMO initialization failed") from e

    def get_valid_actions(self, current_edge: str) -> List[str]:
        if current_edge is None or current_edge.startswith(":"):
            print(f"[get_valid_actions] Invalid or internal edge: {current_edge}")
            return []

        try:
            edge_obj = self.net.getEdge(current_edge)
            outgoing_edges = [
                e.getID() for e in edge_obj.getOutgoing()
                if not e.getID().startswith(":")
            ]
            return outgoing_edges
        except Exception as e:
            print(f"[get_valid_actions] Error on edge {current_edge}: {e}")
            return []

    def reset(self) -> Optional[str]:
        self._init_traci()
        self.step_count = 0

        valid_edges = [e for e in self.all_edges if self.get_valid_actions(e)]
        if not valid_edges:
            print("No valid edges with outgoing connections")
            return None

        for _ in range(10):
            self.start_edge = random.choice(valid_edges)
            self.destination_edge = random.choice([e for e in valid_edges if e != self.start_edge])

            try:
                route_edges, _ = self.net.getShortestPath(self.net.getEdge(self.start_edge), self.net.getEdge(self.destination_edge))
                if not route_edges:
                    continue
                route_id = f"rl_route_{int(time.time())}"
                route_edge_ids = [e.getID() for e in route_edges]
                traci.route.add(route_id, route_edge_ids)
                traci.vehicle.add(
                    vehID=self.vehicle_id,
                    routeID=route_id,
                    typeID="rl_car",
                    departLane="best",
                    departSpeed="0"
                )
                traci.simulationStep()
                if self.vehicle_id in traci.vehicle.getIDList():
                    current_edge = traci.vehicle.getRoadID(self.vehicle_id)
                    print(f"Selected route: {self.start_edge} → {self.destination_edge}")
                    print(f"Vehicle placed at {current_edge}")
                    if current_edge is None or current_edge.startswith(":"):
                        print("Reset led to internal or unknown edge — skipping episode")
                        return None
                    return current_edge
            except Exception as e:
                print(f"[reset] Error: {e}")
                continue

        print("Reset failed after multiple route attempts")
        return None

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        if self.vehicle_id not in traci.vehicle.getIDList():
            return None, -100, True, {"error": "vehicle_lost"}

        try:
            traci.vehicle.setRoute(self.vehicle_id, [action])
            traci.simulationStep()
            self.step_count += 1
            current_edge = traci.vehicle.getRoadID(self.vehicle_id)
            done = False
            reward = 0

            try:
                waiting_time = traci.edge.getWaitingTime(current_edge)
            except:
                waiting_time = 0

            if current_edge == self.destination_edge:
                reward = 100
                done = True
            elif waiting_time <= 50:
                reward = -5
            elif waiting_time <= 100:
                reward = -10
            elif waiting_time <= 150:
                reward = -20
            elif waiting_time <= 200:
                reward = -30
            else:
                reward = -100

            if self.step_count >= self.max_steps:
                done = True

            return current_edge, reward, done, {
                "waiting_time": waiting_time,
                "step": self.step_count
            }

        except Exception as e:
            print(f"[step] Error: {e}")
            return None, -100, True, {"error": str(e)}

    def close(self):
        if traci.isLoaded():
            traci.close()
            print("SUMO closed")