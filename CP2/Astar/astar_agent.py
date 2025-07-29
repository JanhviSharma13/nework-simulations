import networkx as nx
import sumolib
from math import sqrt

def heuristic(coord1, coord2):
    return sqrt((coord1[0] - coord1[0]) ** 2 + (coord2[1] - coord2[1]) ** 2)

def build_graph(net_file):
    net = sumolib.net.readNet(net_file)
    G = nx.DiGraph()

    for edge in net.getEdges():
        if not edge.allows("passenger"):
            continue
        from_node = edge.getFromNode()
        to_node = edge.getToNode()
        edge_id = edge.getID()
        length = edge.getLength()

        G.add_edge(
            from_node.getID(),
            to_node.getID(),
            id=edge_id,
            weight=length,
            from_coord=from_node.getCoord(),
            to_coord=to_node.getCoord()
        )

    return G, net

def a_star_path(G, net, from_edge_id, to_edge_id):
    from_node = net.getEdge(from_edge_id).getToNode().getID()
    to_node = net.getEdge(to_edge_id).getFromNode().getID()

    def h(n1, n2=to_node):
        c1 = net.getNode(n1).getCoord()
        c2 = net.getNode(n2).getCoord()
        return heuristic(c1, c2)

    try:
        path = nx.astar_path(G, from_node, to_node, heuristic=h, weight="weight")
    except nx.NetworkXNoPath:
        return []

    edge_path = []
    for i in range(len(path) - 1):
        edge_data = G.get_edge_data(path[i], path[i+1])
        edge_path.append(edge_data["id"])
    return edge_path
