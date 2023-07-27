from unittest import result

# from tables import Description
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
from hdbscan_cls import hdbscan_clustering
from multiprocessing import Pool
import warnings
import  fcntl

warnings.filterwarnings('ignore')
# Tạo logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# Tạo file handler và thiết lập mức độ ghi log
file_handler = logging.FileHandler('log_file.log')
file_handler.setLevel(logging.DEBUG)

# Tạo formatter để định dạng thông điệp ghi log
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Thiết lập formatter cho file handler
file_handler.setFormatter(formatter)

# Thêm file handler vào logger
logger.addHandler(file_handler)
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
    return [min_dis, idx_v1, idx_v2]




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
        list_node.append((x,y))
        old_sensor = (x,y)

    return list_node

def energy(no_receive, no_transmit, R):
    E_elec = 50*1e-9
    E_freespace = 10*1e-12
    # E_da = 5*1e-12
    K = 525*8
    return no_receive*(E_elec)*K + no_transmit*(K*E_elec + K*E_freespace*R*R)

def networks_E_consumption(Node_transfers, R):
    list_energy = []
    for node_index, values in Node_transfers.items():
        no_receive, no_transmit = values
        energy_consumption = energy(no_receive, no_transmit, R)
        list_energy.append(energy_consumption)

    # list energy   
    return list_energy, sum(list_energy), max(list_energy) - min(list_energy)
    



def cluster_set(anchor_points):
    model_cluster, num_cluster = get_optimal_numcluster(anchor_points)
    labels = model_cluster.labels_
    return labels, num_cluster


def solvep2(in_path:str):

    start = time.time()
    anchor_points,  W, H, BS, M, R, K, cars = solve(in_path)
    # print(anchor_points)
    # print(' len anchor = ', len(anchor_points))
    anchor_points =  [BS] + anchor_points 
    relaynode = []
    if len(anchor_points)/2 < 1:
        # dont clustering
        relaynode = steiner_tree_1(anchor_points, R)
        # relaynode += anchor_points
    else:
        node_infos = [[anchor[0], anchor[1]] for anchor in anchor_points]
        labels, num_cluster = hdbscan_clustering(node_infos)
        # print(set(labels))
        # return 0
        #clustering anchor_points:
        set_cluster = []
        for j in range(len(anchor_points)):
            if labels[j] == -1:
                set_cluster.append([anchor_points[j]])

        for i in range(0,num_cluster -1):
            point_in_clusters = []
            for j in range(len(anchor_points)):
                if labels[j] == i:
                    point_in_clusters.append(anchor_points[j])
            set_cluster.append(point_in_clusters)
        
        # print("Setcluster = ", set_cluster)
        # return 0
        # sum_list = []
        # for list_ in set_cluster:
        #     sum_list = sum_list + list_

        # print(f"This is len of sumlist {len(sum_list)}")
        # print(f"This is len of achorpoints {len(anchor_points)}")
        
        set_node_in_each_clusters  =  [steiner_tree_1(points, R) for points in set_cluster]
        # for i, anchor in enumerate(anchor_points):
        #     check = False
        #     for relaynodes in set_node_in_each_clusters:
        #         for relay in relaynodes:
        #             if distance(anchor, relay) <= 2*R +0.1:
        #                 check = True
        #                 break
        #     if check == False:
        #         print(f"\n Ancho {i} not connected")
        # for i, list_point in enumerate(set_cluster):
        #     check = False
        #     for point in list_point:
        #         for p1 in set_node_in_each_clusters[i]:
        #             if distance(point, p1)<= 2*R+0.1:
        #                 check = True
        #                 break
        #     if check ==False:
        #         print(f"Not connected in set_node_in_each_clusters[{i}]")
        #     else:
        #         print(f"set_node_in_each_clusters[{i}] is connected")

        # convert to tuple
        # print(set_node_in_each_clusters)
        for i in range(len(set_node_in_each_clusters)-1):
            for j in range(i+1, len(set_node_in_each_clusters)):
                if(len( set(set_node_in_each_clusters[i]).intersection(set(set_node_in_each_clusters[j])) ) >0):
                    set_node_in_each_clusters[i] = set_node_in_each_clusters[i] + set_node_in_each_clusters[j]
                    set_node_in_each_clusters[j] = []

        # checking empty list
        set_node_in_each_clusters = [sublist for sublist in set_node_in_each_clusters if sublist]
        # for setx in set_node_in_each_clusters:
        #     if (len(setx) ==0):
        #         set_node_in_each_clusters.remove(setx)      
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
        # print(set_node_in_each_clusters)
        add_node = []
        for edge in T.edges:
            a,b = edge_G[(edge)]
            nodeA = set_node_in_each_clusters[edge[0]][a]
            nodeB = set_node_in_each_clusters[edge[1]][b]
            add_node = add_node + get_relay_between_2_node(nodeA, nodeB, R)

            



        for set_node in set_node_in_each_clusters:
            add_node += set_node
    
        relaynode = add_node
    
    # for i, anchor in enumerate(anchor_points):
    #     check = False
    #     for relay in relaynode:
    #         if distance(anchor, relay) <= 2*R +0.1:
    #             check = True
    #             break
    #     if check == False:
    #         print(f"\n Ancho {i} not connected")

    # x = anchor_points + relaynode
    # G_c = nx.Graph()
    # for i in range(len(x) -1):
    #     for j in range(i+1, len(x)):
    #         if distance(x[i], x[j]) <= 2*R + 0.1:
    #             G_c.add_edge(i,j, weight=1)
    # print("______________________________________________")            
    # print(nx.number_connected_components(G_c))

    end_time = time.time() - start

    # remove redudant node
    all_node = cars + anchor_points + relaynode
    #all_node = cars +anchor_points+ relaynode


    G1 = nx.Graph()
    for i in range(len(all_node)-1):
        for j in range(i+1, len(all_node)):
            if i >= len(cars) or j>=len(cars):
                if distance(all_node[i], all_node[j]) <= 2*R + 0.1:
                    G1.add_edge(i,j, weight=1)
            else:
                if all_node[i][3] == all_node[j][3]  and distance(all_node[i], all_node[j]) <= 2*R + 0.1 :
                    G1.add_edge(i,j, weight=1)
    #print(nx.number_connected_components(G1))
    # print(G1.edges)
    ter = [idx for idx in range(len(cars)+1)]
    H = steiner_tree(G1,ter, weight="weight")

    ##############  Caculate the energy ################################
    car_ids = list(range(len(cars)))
    Base = len(cars)
    def shortest_paths(G, X, V):
        paths = []
        for x in X:
            shortest_path = nx.shortest_path(G, source=x, target=V)
            paths.append(shortest_path)
        return paths
    result = shortest_paths(H, car_ids, Base)
    Node_transfer = {}
    for path in result:
        for i,idx in enumerate(path[:-1]):
            if idx > len(cars):
                if idx not in Node_transfer:
                    if i ==0:
                        Node_transfer[idx] = [0, 1]
                    else:
                        Node_transfer[idx] = [1, 1]
                else:
                    if i ==0:
                        Node_transfer[idx][1] += 1
                    else:
                        Node_transfer[idx][1] += 1
                        Node_transfer[idx][0] += 1

    list_energy, total_e, denta_e = networks_E_consumption(Node_transfer, R)
    # print(f"total_e = {total_e}")
    # print(f"denta_e = {denta_e}")



    return len(H.nodes) - len(cars),total_e, denta_e ,end_time
    # final_node = []
    # for i in H.nodes:
    #     final_node.append(all_node[i])
      
    # return  len(final_node), end_time


def solver(i):

    print("\nStart test {}".format(i))
    avg_node, avg_time,avg_total_e,avg_denta_e = 0,0,0,0
    best,best_totale,best_denta_e = 1000000,100000,100000
    result = []
    num_runs = 20
    for j in tqdm(range(num_runs), desc = "Iter: "):
        num_node, total_e, denta_e ,end_time = solvep2('./Testnew/'+ str(i)+'.inp')
        # num_node = solvep2(f'./Testnew/{i}.inp')
        # print(num_node)
        if num_node < best:
            best = num_node
            best_totale = total_e
            best_denta_e = denta_e
        result.append(num_node)
        avg_total_e += total_e/num_runs
        avg_denta_e += denta_e/num_runs
        avg_node += num_node/num_runs
        avg_time += end_time/num_runs
    print(f"Avg node = {avg_node}, best node = {best}, best total energy = {best_totale}, best avg_e = {best_denta_e} avg_time = {avg_time}")
    # print('num.Node = {} '.format(avg_node))
    # print('avg_time = {} '.format(avg_time))
    # print('best = {}'.format(best))
    f = open("./results/result_data_2.txt", "a")
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    f.write("_________TestCase: {}______________\n".format(i))
    f.write('')
    f.write('best num node = {}\n'.format(best))
    f.write('num.Node = {} \n'.format(avg_node))
    f.write('total_e = {} \n'.format(avg_total_e))
    f.write('denta_e = {} \n'.format(avg_denta_e))
    f.write('avg_time = {} \n'.format(avg_time))
    f.write(f"The nodes:{result} \n")
    f.flush()
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    print("Done........")

    f.close()
    return i

if __name__ == '__main__':
    list_instance = list(range(6,34))
    with Pool(30) as p:
        results = list(tqdm(p.imap(solver, list_instance), total=len(list_instance)))