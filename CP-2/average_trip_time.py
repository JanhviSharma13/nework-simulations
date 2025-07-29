import traci
import sumolib

sumo_binary = "sumo"  # use "sumo-gui" for GUI
sumocfg_file = "cp.sumocfg"

print("[INFO] Starting SUMO simulation...")
traci.start([sumo_binary, "-c", sumocfg_file])

depart_times = {}
arrival_times = {}

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    for veh_id in traci.vehicle.getIDList():
        if veh_id not in depart_times:
            depart_times[veh_id] = traci.vehicle.getDeparture(veh_id)

    for veh_id in traci.simulation.getArrivedIDList():
        arrival_times[veh_id] = traci.simulation.getTime()

traci.close()

# Compute average trip time
total_time = 0
count = 0
for veh_id in arrival_times:
    if veh_id in depart_times:
        total_time += (arrival_times[veh_id] - depart_times[veh_id])
        count += 1

if count > 0:
    avg_trip_time = total_time / count
    print(f"[INFO] Average trip time: {avg_trip_time:.2f} seconds over {count} completed trips.")
else:
    print("[WARN] No completed trips.")
