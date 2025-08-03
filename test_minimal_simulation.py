#!/usr/bin/env python3
"""
Minimal test to verify SUMO simulation works
"""

import xml.etree.ElementTree as ET
import os
import tempfile

def test_minimal_simulation():
    """Test SUMO simulation with a minimal route"""
    
    # Create a minimal route file
    routes_content = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="70" guiShape="passenger"/>
    <vehicle id="test_vehicle" type="car" depart="0">
        <route edges="24584377#5 24584377#6 1354558617 24584383#0 24584383#1 24584383#2 24584383#3 24584383#4 80926764#0 80926764#1 80926764#2 695569947 758753881#1 758753882 590609994#0 590609994#1 572511730 447141148 24583327#0 24583327#1 24583327#2 24583327#3 24583327#4 24583327#5 24583327#6 24583327#7 24583327#8 24583327#9 24583327#10 24583327#11 24583327#12 -1066619123 74851428#0 74851428#1 74851428#2 74851428#3 74851428#4 74851428#5 563791811#0 563791811#1 563791811#2 563791811#3 563791811#4 563791811#5 563791811#6 563791811#7 563791811#8 563679820#0 563679820#1 563679820#2 563679820#3 563679820#4 563679820#5 563679820#6 563679820#7 563679820#8 563679820#9 563679820#10 563679820#11 563679820#12 563679820#13 563679820#14 563679820#15 563679820#16 563679820#17 665921722 1312753971#0 1312753971#1 553596950 74851427 164782722 -265975966#5 1312753969#0 1076782194#0"/>
    </vehicle>
</routes>"""
    
    # Write route file
    with open("test_minimal_route.rou.xml", "w") as f:
        f.write(routes_content)
    
    print("Created minimal route file")
    
    # Test SUMO command
    try:
        import traci
        
        sumo_cmd = [
            'sumo', '--net-file', 'CP2/cp2.net.xml',
            '--route-files', 'test_minimal_route.rou.xml',
            '--no-step-log', 'true',
            '--no-warnings', 'true',
            '--random', 'false',
            '--time-to-teleport', '300',
            '--ignore-route-errors', 'true',
            '--ignore-accidents', 'true'
        ]
        
        print(f"Starting SUMO with command: {' '.join(sumo_cmd)}")
        traci.start(sumo_cmd)
        
        print("SUMO started successfully!")
        print(f"Vehicle list: {traci.vehicle.getIDList()}")
        
        # Run simulation
        step = 0
        while step < 100:
            traci.simulationStep()
            step += 1
            
            if "test_vehicle" in traci.vehicle.getIDList():
                try:
                    edge = traci.vehicle.getRoadID("test_vehicle")
                    print(f"Step {step}: Vehicle on edge {edge}")
                except:
                    print(f"Step {step}: Vehicle left network")
                    break
            else:
                print(f"Step {step}: Vehicle not found")
                break
        
        total_time = traci.simulation.getTime()
        traci.close()
        print(f"Simulation completed! Total time: {total_time}")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    if os.path.exists("test_minimal_route.rou.xml"):
        os.remove("test_minimal_route.rou.xml")

if __name__ == "__main__":
    test_minimal_simulation() 