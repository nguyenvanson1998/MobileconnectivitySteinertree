from os import terminal_size
import numpy as np
import networkx as nx
import math
from networkx.algorithms.shortest_paths import weighted
from networkx.algorithms.approximation.steinertree import steiner_tree
import matplotlib.pyplot as plt
eps = np.finfo(float).eps

def read_file(in_path: str):
    with open(in_path, "r") as f:
        W, H = [float(x) for x in f.readline().split()]
        BS = tuple([float(x) for x in f.readline().split()])
        M = int(f.readline())
        R = float(f.readline())
        K = int(f.readline())
        cars = []
        for k in range(K):
            for m in range(M):
                cars.append(tuple([float(x) for x in f.readline().split()] + [m, k]))
        return W,H, BS, M, R, K, cars

def solve(in_path: str):
    NUM_NODE = 0
    W, H, BS, M, R, K, cars = read_file(in_path)
    print('End Reading 1')
    D = R
    Uw = math.ceil(W/D)
    Uh = math.ceil(H/D)

    def cid_to_center(cid: int):
        cy = cid // Uw
        cx = cid - Uw * cy

        lx = cx * D
        rx = min((cx + 1)*D, W)
        ly = cy*D
        ry = min((cy + 1)*D, H)
        return tuple([(lx + rx)/2, (ly+ry)/2] +[cid])
    points = []
    for cid in range(Uw*Uh):
        NUM_NODE += 1
        points.append(cid_to_center(cid))
    def point_to_cid(point):
        cy = math.ceil(point[0]/ D)
        cx = math.ceil(point[1]/D)
        return cx + Uw*cy
    points.append(BS+ tuple([NUM_NODE]))
    NUM_NODE +=1
    for i in range(len(cars)):
        points.append(cars[i] + tuple([NUM_NODE]))
        NUM_NODE +=1
    print("Done get i")


    def get_edges_of_terminal(index:int, edges):
        cid = point_to_cid(points[index])
        cy = cid // Uw
        cx = cid - Uw * cy
        edges_node = []
        if cx  - 1 >= 0 :
            print("hahahahaha")
        


    def get_edges_of_anchor(cid:int):
        edge_nodes = []
        cy = cid // Uw
        cx = cid - Uw * cy
        if cx >0 and cx < Uw - 1:
            if cy >0 and cy < Uh -1:
                edge_nodes.append(points[cid -1])
                edge_nodes.append(points[cid +1])
                edge_nodes.append(points[cid +Uw])
                edge_nodes.append(points[cid -Uw])
            elif cy == 0:
                edge_nodes.append(points[cid -1])
                edge_nodes.append(points[cid +1])
                edge_nodes.append(points[cid +Uw])
            else:
                edge_nodes.append(points[cid -1])
                edge_nodes.append(points[cid +1])
                edge_nodes.append(points[cid -Uw])
        elif cx == 0:
            if cy >0 and cy < Uh -1:
                edge_nodes.append(points[cid +1])
                edge_nodes.append(points[cid +Uw])
                edge_nodes.append(points[cid -Uw])
            elif cy == 0:
                edge_nodes.append(points[cid +1])
                edge_nodes.append(points[cid +Uw])
            else:
                edge_nodes.append(points[cid +1])
                edge_nodes.append(points[cid -Uw])
        else:
            if cy >0 and cy < Uh -1:
                edge_nodes.append(points[cid -1])
                edge_nodes.append(points[cid +Uw])
                edge_nodes.append(points[cid -Uw])
            elif cy == 0:
                edge_nodes.append(points[cid -1])
                edge_nodes.append(points[cid +Uw])
            else:
                edge_nodes.append(points[cid -1])
                edge_nodes.append(points[cid -Uw])
        return edge_nodes
    

            
            





                
        


    def connectable(cid1: int, cid2: int):
        if cid1 == cid2:
            return False
        if min(cid1, cid2) > Uw*Uh and points[cid1][3] != points[cid2][3]:
            return False
        p1 = points[cid1]
        p2 = points[cid2]
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 <= 4*R*R + 0.001

    def get_edges(cid:int):
        edge_nodes = []
        for i in range(len(points)):
            if i != cid and connectable(cid, i):
                edge_nodes.append(i)
        return edge_nodes

    edges = []
    for i in range(len(points)):
        edges.append(get_edges(i))
    

    # build model
    pos = {}
    for i in range(len(points)):
        pos[i] = (points[i][0], points[i][1])
    print("Done reading File")
    G = nx.Graph()
    for i in range(len(edges)):
        for j in range(len(edges[i])):
            G.add_edge(i, edges[i][j], weight=1)
    terminal_nodes = [points[i][-1] for i in range(Uw*Uh,len(pos))]
    H = steiner_tree(G,terminal_nodes, weight="weight")
    print(len(H.nodes) - len(cars) -1)
    # nx.draw_networkx_nodes(G,pos, node_size=3000, nodelist=G.nodes, node_color="tab:red")
    # nx.draw_networkx_edges(G,pos, alpha=0.5, width=6, edge_color="red")
    # nx.draw_networkx_nodes(G,pos, node_size=3000, nodelist=H.nodes, node_color="tab:blue")
    # nx.draw_networkx_edges(H,pos, alpha=0.5, width=6, edge_color="blue")
    # ax = plt.gca()
    # ax.margins(0.11)
    # plt.tight_layout()
    # plt.axis("off")
    # plt.show()

if __name__ == '__main__':
    #solve('./Test/Test2_10_24_50.inp')
    solve('./Testmip/Test5.inp')
