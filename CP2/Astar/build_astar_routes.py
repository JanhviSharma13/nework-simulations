import xml.etree.ElementTree as ET
from astar_agent import build_graph, a_star_path

def parse_trips(trips_file):
    tree = ET.parse(trips_file)
    root = tree.getroot()
    trips = []
    for trip in root.findall("trip"):
        from_edge = trip.attrib["from"]
        to_edge = trip.attrib["to"]
        trips.append((from_edge, to_edge))
    return trips

def generate_routes(net_file, trips_file, output_file):
    G, net = build_graph(net_file)
    trips = parse_trips(trips_file)

    routes_root = ET.Element("routes")

    # ✅ Add vehicle type definition
    ET.SubElement(routes_root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5",
                  length="5", minGap="2.5", maxSpeed="70", guiShape="passenger")

    for i, (from_edge, to_edge) in enumerate(trips):
        edges = a_star_path(G, net, from_edge, to_edge)
        if not edges:
            continue
        veh = ET.SubElement(routes_root, "vehicle", id=f"veh{i}", type="car", depart=str(i*2))
        ET.SubElement(veh, "route", edges=" ".join(edges))

    tree = ET.ElementTree(routes_root)
    tree.write(output_file)
    print(f"Saved {output_file}")
