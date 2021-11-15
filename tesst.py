from networkx.algorithms.shortest_paths import weighted
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from networkx.algorithms.approximation.steinertree import steiner_tree
edges = []
for i in range(5):
    edgesi = []
    for j in range(5):
        if(j !=i):
            edgesi.append(j)
    edges.append(edgesi)


G = nx.Graph()
for i in range(len(edges)):
    for j in range(len(edges[i])):
        G.add_edge(i, edges[i][j], weight=1)

pos = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1), 4: (0.5, 2.0)}
terminal_nodes =  (1,3,4)
nx.draw_networkx_nodes(G,pos, node_size=3000, nodelist=G.nodes, node_color="tab:red")
nx.draw_networkx_edges(G,pos, alpha=0.5, width=6, edge_color="red")
H = steiner_tree(G,terminal_nodes, weight="weight")
nx.draw_networkx_nodes(G,pos, node_size=3000, nodelist=H.nodes, node_color="tab:blue")
nx.draw_networkx_edges(H,pos, alpha=0.5, width=6, edge_color="blue")


ax = plt.gca()
ax.margins(0.11)
plt.tight_layout()
plt.axis("off")
plt.show()
