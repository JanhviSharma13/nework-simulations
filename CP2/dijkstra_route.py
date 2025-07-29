import networkx as nx
import xml.etree.ElementTree as ET
import random

def build_graph(net_file):
    G = nx.DiGraph()
    tree = ET.parse(net_file)
    root = tree.getroot()
    for edge in root.findall("edge"):
        if 'function' in edge.attrib:
            continue
        from_node = edge.attrib['from']
        to_node = edge.attrib['to']
        edge_id = edge.attrib['id']
        for lane in edge.findall("lane"):
            length = float(lane.attrib['length'])
            G.add_edge(from_node, to_node, weight=length, edge_id=edge_id)
    return G

def get_edge_mapping(net_file):
    edge_map = {}
    tree = ET.parse(net_file)
    root = tree.getroot()
    for edge in root.findall("edge"):
        if 'function' in edge.attrib:
            continue
        from_node = edge.attrib['from']
        to_node = edge.attrib['to']
        edge_id = edge.attrib['id']
        edge_map[(from_node, to_node)] = edge_id
    return edge_map

def compute_route(G, edge_map, src, tgt):
    try:
        path_nodes = nx.dijkstra_path(G, source=src, target=tgt, weight="weight")
    except:
        return []

    edge_route = []
    for i in range(len(path_nodes) - 1):
        edge_id = edge_map.get((path_nodes[i], path_nodes[i + 1]))
        if edge_id:
            edge_route.append(edge_id)
        else:
            return []
    return edge_route

def get_valid_node_pairs(G, num_pairs=100):
    nodes = list(G.nodes())
    pairs = set()
    attempts = 0
    while len(pairs) < num_pairs and attempts < num_pairs * 20:
        src, tgt = random.sample(nodes, 2)
        if src != tgt and nx.has_path(G, src, tgt):
            pairs.add((src, tgt))
        attempts += 1
    return list(pairs)
