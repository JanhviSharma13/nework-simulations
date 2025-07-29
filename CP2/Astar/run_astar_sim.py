import traci
import os
import subprocess
from build_astar_routes import generate_routes
from generate_trips import generate_random_trips
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
# ---- Paths
net_file = "../cp2.net.xml"
trips_file = "astar_trips.xml"
rou_file = "astar_routes.rou.xml"
output_plot = "trip_scatter.png"

# ---- Step 1: Generate trips and routes
print("Generating random trips...")
generate_random_trips(net_file, trips_file)

print("Building routes with A*...")
generate_routes(net_file, trips_file, rou_file)

# ---- Step 2: Run simulation in GUI mode
print("Running simulation in SUMO-GUI...")
sumo_cmd = ["sumo-gui", "-n", net_file, "-r", rou_file]
traci.start(sumo_cmd)

# ---- Step 3: Track vehicle data
departures = {}
arrivals = {}
distances = {}

step = 0
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    
    for veh_id in traci.simulation.getDepartedIDList():
        departures[veh_id] = traci.simulation.getTime()
        route = traci.vehicle.getRoute(veh_id)
        if route and len(route) > 1:
            dist = 0
            for edge_id in route:
                try:
                    dist += traci.lane.getLength(edge_id + "_0")
                except:
                    pass  # ignore if edge doesn't exist
            distances[veh_id] = dist


    for veh_id in traci.simulation.getArrivedIDList():
        arrivals[veh_id] = traci.simulation.getTime()

    step += 1

traci.close()

# ---- Step 4: Calculate trip durations and plot
trip_times = []
trip_dists = []

for veh_id in arrivals:
    if veh_id in departures and veh_id in distances:
        time = arrivals[veh_id] - departures[veh_id]
        dist = distances[veh_id]
        trip_times.append(time)
        trip_dists.append(dist)
trip_dists = np.array(trip_dists)
trip_times = np.array(trip_times)
# ---- Step 5: Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(trip_dists, trip_times, color='blue', alpha=0.6)
m, b = np.polyfit(trip_dists, trip_times, 1)
plt.plot(trip_dists, m * trip_dists + b, color="black", linestyle="--", label="Best Fit Line")
plt.xlabel("Trip Distance (meters)")
plt.ylabel("Trip Time (seconds)")
plt.title("Trip Time vs. Distance (A* Routes)")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_plot)
print(f"✅ Saved scatter plot: {output_plot}")
