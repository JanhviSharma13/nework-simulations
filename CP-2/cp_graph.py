import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

# Load SUMO .net.xml as a networkx-compatible MultiDiGraph
Cp_G = ox.graph_from_xml("cp.osm", simplify=False, retain_all=True)

# Convert to undirected graph
G_u = Cp_G.to_undirected()

# Basic metrics
print("Nodes:", len(G_u.nodes))
print("Edges:", len(G_u.edges))
print("Avg node degree:", sum(dict(G_u.degree()).values()) / len(G_u.nodes))
#print("Clustering coefficient:", nx.average_clustering(G_u))
#print("Assortativity:", nx.degree_pearson_correlation_coefficient(G_u))
print("Graph density:", nx.density(G_u))

# Plot and save the graph
ox.plot_graph(
    Cp_G,
    node_size=10,
    edge_color='black',
    bgcolor='white',
    node_color='red',
    show=True,   # ← THIS shows the plot in a popup window
    close=False
)
