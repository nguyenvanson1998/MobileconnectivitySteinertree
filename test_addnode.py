from utils import distance
from phase2al import get_relay_between_2_node
from utils import distance
import networkx as nx
import time


start  = time.time()
list1 = [(20.0, 84.0), (20.0, 90.0), (8.0, 78.0), (14.0, 78.0)]


G1 = nx.Graph()
for i in range(len(list1) -1):
    for j in range(i+1, len(list1)):
        if distance(list1[i], list1[j]) <= 2*4 +0.001:
            G1.add_edge(i, j, weight = 1)

print("--- %s seconds ---" % (time.time() - start))
print(G1.edges)