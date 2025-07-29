import traci
import traci.constants as tc
import os
import time

sumo_binary = "sumo"  # or "sumo-gui"
net_file = "cp2.net.xml"
route_file = "cp2.rou.xml"

def run_simulation():
    traci.start([sumo_binary, "-n", net_file, "-r", route_file])
    print("Simulation started...")
    step = 0
    arrived = []

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        arrived += traci.simulation.getArrivedIDList()
        if step % 50 == 0:
            print(f"Step {step}, vehicles arrived: {len(arrived)}")

    print(f"Simulation complete. Total arrived: {len(arrived)}")
    traci.close()

if __name__ == "__main__":
    run_simulation()
