from unittest import result

from tables import Description
from phase1al import solve
from clustering import get_optimal_numcluster, visualize_2d
import itertools
from utils import check_inside, distance
from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.context import Context
import networkx as nx
import logging
import math
import time
from tqdm import tqdm
from networkx.algorithms.approximation.steinertree import steiner_tree

logging.getLogger('file_handler').addHandler(logging.NullHandler())
logging.getLogger('file_handler').propagate = False


def anchor_to_terminal_of_grid(ter, minX, minY, maxX, maxY, grid_size):
    x = round(min( round((ter[0] - minX)/grid_size)*grid_size + minX, maxX),2)
    y = round(min(round((ter[1] - minY)/grid_size)*grid_size + minY, maxY),2)
    return (x,y)

def edge_cost(v1:list, v2:list):
    min_dis = 1e9
    idx_v1 = 0
    idx_v2 = 0
    for i in range(len(v1)):
        for j in range(len(v2)):
            if distance(v1[i], v2[j]) < min_dis:
                min_dis = distance(v1[i], v2[j])
                idx_v1 = i
                idx_v2 = j
    return [min_dis, i, j]




def steiner_tree_1(terminals, R):
    # grid_size = math.floor(R*math.sqrt(2))
    grid_size = math.floor(R*2)
    #print("The grid size = ", grid_size)
    minX = math.floor(min(terminals, key = lambda x: x[0])[0])
    maxX = round((((max(terminals, key = lambda x: x[0])[0] - minX)//grid_size +1)*grid_size + minX))
    minY = math.floor(min(terminals, key = lambda x: x[1])[1])
    maxY = round((((max(terminals, key = lambda x: x[1])[1] - minY)//grid_size +1)*grid_size + minY))
    grid = None
    grid_dim = [minX, maxX, minY, maxY]
    n_type = 4
    # print("Minx = {0}, MaxX = {1}, MinY = {2}, MaxY = {3}".format(minX, maxX, minY, maxY) )

    # create a squreGrid using GraphFactory
    graph = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)
    # print("The node in graph")
    # print(list(graph.get_nodes()))
    # print("Size of graph = ", len(list(graph.get_nodes())))   
    terminal_in_Grids = []
    for ter in terminals:
        terminal_in_Grids.append(anchor_to_terminal_of_grid(ter,minX,minY,maxX,maxY,grid_size))
    terminal_in_Grids = list(set(terminal_in_Grids))
    # print("Terminal in Grids = ", terminal_in_Grids)

    # print("the node not in grid = ")
    # for node in terminal_in_Grids:
    #     if node not in list(graph.get_nodes()):
    #         print(node +" haha")
    
    
    if(len(terminal_in_Grids) <2):
        return terminal_in_Grids
    else:
        #debug: print the terminal_in_Grids list
        # print(modified_ter)

        #create the context:
        context = Context(graph,terminal_in_Grids)
        # run and store results for S star heuristic search
        context.run('S*-MM')
        results = context.return_solutions()
        #print(results)
        res = list(set(itertools.chain.from_iterable(results['path'])))
        return res



def get_relay_between_2_node(nodeA, nodeB, R):
    cos_alpha = (nodeB[0] - nodeA[0])/distance(nodeA, nodeB)
    sin_alpha = (nodeB[1] - nodeA[1])/distance(nodeA, nodeB)
    list_node =[]
    old_sensor = nodeA
    while distance(old_sensor, nodeB) >= 2*R -0.001:
        x = old_sensor[0] + cos_alpha*2*R
        y = old_sensor[1] + sin_alpha*2*R
        list_node.append([x,y])
        old_sensor = [x,y]

    return list_node





def cluster_set(anchor_points):
    model_cluster, num_cluster = get_optimal_numcluster(anchor_points)
    labels = model_cluster.labels_
    return labels, num_cluster


def solvep2(in_path:str):

    start = time.time()
    anchor_points,  W, H, BS, M, R, K, cars = solve(in_path)
    # print(' len anchor = ', len(anchor_points))
    anchor_points = [BS] + anchor_points
    relaynode = []
    if len(anchor_points)/2 < 4000:
        # dont clustering
        relaynode = steiner_tree_1(anchor_points, R)
        # relaynode += anchor_points
    else:
        labels, num_cluster = cluster_set(anchor_points)
        #clustering anchor_points:
        set_cluster = []
        for i in range(num_cluster):
            point_in_clusters = []
            for j in range(len(anchor_points)):
                if labels[j] == i:
                    point_in_clusters.append(anchor_points[j])
            set_cluster.append(point_in_clusters)
        
        # print("Setcluster = ", set_cluster)
        

        set_node_in_each_clusters  = [steiner_tree_1(points, R) for points in set_cluster]
        for i in range(len(set_node_in_each_clusters)-1):
            for j in range(i+1, len(set_node_in_each_clusters)):
                if(len( set(set_node_in_each_clusters[i]).intersection(set(set_node_in_each_clusters[j])) ) >0):
                    set_node_in_each_clusters[i] = set_node_in_each_clusters[i] + set_node_in_each_clusters[j]
                    set_node_in_each_clusters[j] = []

        # checking empty list
        for setx in set_node_in_each_clusters:
            if (len(setx) ==0):
                set_node_in_each_clusters.remove(setx)
        

        # connecting all list:
        #Building the complete graph

        G = nx.Graph()
        edge_G = {}

        G.add_nodes_from(range(len(set_node_in_each_clusters)))
        for i in range(len(set_node_in_each_clusters) -1):
            for j in range(i+1, len(set_node_in_each_clusters)):
                info = edge_cost(set_node_in_each_clusters[i],set_node_in_each_clusters[j] )
                edge_G[(i,j)] = (info[1], info[2])
                G.add_edge(i,j,weight = info[0])
        # Concat the cluster to each other:
        T = nx.minimum_spanning_tree(G)
        add_node = []
        for edge in T.edges:
            a,b = edge_G[edge]
            nodeA = set_node_in_each_clusters[edge[0]][a]
            nodeB = set_node_in_each_clusters[edge[1]][b]
            add_node = add_node + get_relay_between_2_node(nodeA, nodeB, R)
        



        for set_node in set_node_in_each_clusters:
            add_node += set_node
    
        relaynode = add_node
    #print('relay ', relaynode)

    end_time = time.time() - start

    # remove redudant node
    all_node = cars + anchor_points + relaynode
    #all_node = cars +anchor_points+ relaynode


    G1 = nx.Graph()
    for i in range(len(all_node)-1):
        for j in range(i+1, len(all_node)):
            if j >= len(cars) or j>=len(cars):
                if distance(all_node[i], all_node[j]) <= 2*R + 0.1:
                    G1.add_edge(i,j, weight=1)
            else:
                if all_node[i][3] == all_node[j][3]  and distance(all_node[i], all_node[j]) <= 2*R + 0.1 :
                    G1.add_edge(i,j, weight=1)
    #print(nx.number_connected_components(G1))
    # print(G1.edges)
    ter = [idx for idx in range(len(cars)+1)]
    H = steiner_tree(G1,ter, weight="weight")
    return len(H.nodes) - len(cars),end_time
    # final_node = []
    # for i in H.nodes:
    #     final_node.append(all_node[i])
      
    # return  len(final_node), end_time

f = open("./result_get_new.txt", "a")
for i in tqdm(range(5,34),desc=" Instance: "):
    print("\nStart test {}".format(i))
    avg_node, avg_time = 0,0
    f.write("_________TestCase: {}______________\n".format(i))
    best = 1000000
    result = []
    num_runs = 30
    for j in tqdm(range(num_runs), desc = "Iter: "):
        num_node, end_time = solvep2('./Testnew/'+ str(i)+'.inp')
        if num_node < best:
            best = num_node
        result.append(num_node)
        avg_node += num_node/num_runs
        avg_time += end_time/num_runs
    print('num.Node = {} '.format(avg_node))
    print('avg_time = {} '.format(avg_time))
    print('best = {}'.format(best))
    f.write('')
    f.write('best num node = {}\n'.format(best))
    f.write('num.Node = {} \n'.format(avg_node))
    f.write('avg_time = {} \n'.format(avg_time))
    f.write(f"The nodes:{result} \n")
    f.flush()
    print("Done........")

f.close()
