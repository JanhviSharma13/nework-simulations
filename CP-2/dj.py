import sumolib
import networkx as nx
import random
from lxml import etree
print("[INFO] Loading SUMO network...")
net = sumolib.net.readNet("cp.net.xml")

# Build a NetworkX graph from SUMO edges
G = nx.DiGraph()
for edge in net.getEdges():
    if edge.allows("passenger"):
        from_node = edge.getFromNode().getID()
        to_node = edge.getToNode().getID()
        edge_id = edge.getID()
        length = edge.getLength()
        G.add_edge(from_node, to_node, id=edge_id, weight=length)

print(f"[INFO] NetworkX graph created with {len(G.nodes)} nodes and {len(G.edges)} edges.")

# Get all node IDs from the graph
all_nodes = list(G.nodes)

# Number of OD pairs to test (adjust as needed)
num_pairs = 100
print(f"[INFO] Generating {num_pairs} random origin-destination pairs...")
od_pairs = []
while len(od_pairs) < num_pairs:
    o, d = random.sample(all_nodes, 2)
    if nx.has_path(G, o, d):
        od_pairs.append((o, d))

print(f"[INFO] Running Dijkstra on {len(od_pairs)} pairs...")
results = []
from_node_ids = set(edge.getFromNode().getID() for edge in net.getEdges())
to_node_ids = set(edge.getToNode().getID() for edge in net.getEdges())

results = []
for i, (origin, dest) in enumerate(od_pairs):
    try:
        path = nx.shortest_path(G, origin, dest, weight='weight')
        path_length = nx.shortest_path_length(G, origin, dest, weight='weight')
        edge_ids = []
        valid_path = True

        for u, v in zip(path[:-1], path[1:]):
            if 'id' not in G[u][v]:
                valid_path = False
                break
            edge_ids.append(G[u][v]['id'])

        if valid_path:
            results.append((origin, dest, path_length, edge_ids))
        else:
            print(f"[WARN] Skipping invalid edge path: {origin} → {dest}")

    except Exception as e:
        print(f"[WARN] Dijkstra failed for {origin} → {dest}: {e}")
print(f"[INFO] {len(results)} successful Dijkstra routes generated.")

# Optional: write to a file (e.g., CSV or XML)
with open("dijkstra_results_cp.txt", "w") as f:
    for origin, dest, length, edges in results:
        f.write(f"{origin} -> {dest} | Length: {length:.2f} m | Edges: {edges}\n")

print("[DONE] Results written to dijkstra_results_cp.txt")
print("[INFO] Generating dijkstra_filtered.rou.xml for SUMO...")

root = etree.Element("routes")
vtype = etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", sigma="0.5", length="5", maxSpeed="25")

for i, (origin, dest, length, edge_ids) in enumerate(results):
    vehicle = etree.SubElement(root, "vehicle", id=str(i), type="car", depart=str(i))
    route = etree.SubElement(vehicle, "route", edges=" ".join(edge_ids))

tree = etree.ElementTree(root)
tree.write("dijkstra_filtered.rou.xml", pretty_print=True, xml_declaration=True, encoding="UTF-8")

print("[DONE] dijkstra_filtered.rou.xml written.")