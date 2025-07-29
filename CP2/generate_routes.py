import xml.etree.ElementTree as ET
import sumolib
import random

def parse_trips(trip_file):
    tree = ET.parse(trip_file)
    root = tree.getroot()
    trips = []
    for trip in root.findall("trip"):
        from_edge = trip.attrib["from"]
        to_edge = trip.attrib["to"]
        trips.append((from_edge, to_edge))
    return trips

def generate_routes(net_file, trips, output_file):
    net = sumolib.net.readNet(net_file)
    with open(output_file, "w") as f:
        f.write("<routes>\n")
        f.write('<vType id="car" accel="1.0" decel="5.0" maxSpeed="25" length="5" sigma="0.5" color="1,0,0"/>\n')
        for i, (from_edge_id, to_edge_id) in enumerate(trips):
            from_edge = net.getEdge(from_edge_id)
            to_edge = net.getEdge(to_edge_id)
            try:
                path_edges, _ = net.getShortestPath(from_edge, to_edge)
                edge_str = " ".join([e.getID() for e in path_edges])
                if edge_str:
                    f.write(f'<vehicle id="veh{i}" type="car" depart="{i * 2}">\n')
                    f.write(f'  <route edges="{edge_str}"/>\n')
                    f.write(f'</vehicle>\n')
            except:
                continue
        f.write("</routes>\n")

if __name__ == "__main__":
    trips = parse_trips("cp2.trips.xml")
    generate_routes("cp2.net.xml", trips, "cp2.rou.xml")
    print("Custom routes written to cp2.rou.xml")
