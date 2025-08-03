#!/usr/bin/env python3
"""
Test script to debug SUMO simulation issues
"""

import xml.etree.ElementTree as ET
import os
import sys

def test_simulation():
    """Test SUMO simulation with a simple route"""
    
    # Create a simple test network and route
    net_content = """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">
    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,100.00,100.00" origBoundary="-10000000000.00,-10000000000.00,10000000000.00,10000000000.00" projParameter="!"/>
    <edge id="edge1" from="node1" to="node2" priority="1">
        <lane id="edge1_0" index="0" speed="13.89" length="100.00" shape="0.00,0.00 100.00,0.00"/>
    </edge>
    <edge id="edge2" from="node2" to="node3" priority="1">
        <lane id="edge2_0" index="0" speed="13.89" length="100.00" shape="100.00,0.00 200.00,0.00"/>
    </edge>
    <junction id="node1" type="dead_end" x="0.00" y="0.00" incLanes="" intLanes="" shape="0.00,0.00"/>
    <junction id="node2" type="priority" x="100.00" y="0.00" incLanes="edge1_0" intLanes="" shape="100.00,0.00">
        <request index="0" response="00" foes="00" cont="0"/>
    </junction>
    <junction id="node3" type="dead_end" x="200.00" y="0.00" incLanes="edge2_0" intLanes="" shape="200.00,0.00"/>
    <connection from="edge1" to="edge2" fromLane="0" toLane="0" via=":node2_0_0" dir="r" state="M"/>
    <connection from="edge1" to="edge2" fromLane="0" toLane="0" dir="r" state="M"/>
</net>"""
    
    route_content = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="70" guiShape="passenger"/>
    <vehicle id="test_vehicle" type="car" depart="0">
        <route edges="edge1 edge2"/>
    </vehicle>
</routes>"""
    
    # Write test files
    with open("test_net.net.xml", "w") as f:
        f.write(net_content)
    
    with open("test_route.rou.xml", "w") as f:
        f.write(route_content)
    
    print("Created test network and route files")
    
    # Test SUMO command
    try:
        import traci
        
        sumo_cmd = [
            'sumo', '--net-file', 'test_net.net.xml',
            '--route-files', 'test_route.rou.xml',
            '--no-step-log', 'true',
            '--no-warnings', 'true',
            '--random', 'false',
            '--time-to-teleport', '300'
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
        
        traci.close()
        print("Simulation completed successfully!")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    for file in ["test_net.net.xml", "test_route.rou.xml"]:
        if os.path.exists(file):
            os.remove(file)

if __name__ == "__main__":
    test_simulation() 