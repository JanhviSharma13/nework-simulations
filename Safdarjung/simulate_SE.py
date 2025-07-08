import traci
import sumolib

sumoBinary = "sumo"
sumoConfig = "SE.sumocfg"

traci.start([sumoBinary, "-c", sumoConfig])

step = 0
while step < 1000:
    traci.simulationStep()
    print("Vehicles on road:", traci.vehicle.getIDList())
    step += 1

traci.close()