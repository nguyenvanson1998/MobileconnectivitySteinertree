# clustering using hierarchy
from contextlib import nullcontext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc

from matplotlib.pyplot import figure as fig
from matplotlib import pyplot as plt
import matplotlib

def visualize_tree(list_points):
    arr = []
    for p in list_points:
        arr.append([p.x, p.y])
    dendogram = shc.dendrogram(shc.linkage(arr, method = 'ward'))
    return dendogram

def visualize_2d(data, labels):
    # plot clusters
    union_labels = list(set(labels))
    x = np.asarray([dat[0] for dat in data])
    y = np.asarray([dat[1] for dat in data])


    for i in union_labels:
        plt.scatter(x[labels == i], y[labels == i], label=f'cluster_{i}')


    plt.legend()
    plt.show()




def agglomerative_clustering(points, K):
    scaler = StandardScaler()
    points_scaler = scaler.fit_transform(points)
    X_normalized = normalize(points_scaler)

    ac = AgglomerativeClustering(n_clusters= K, linkage= 'ward').fit(X_normalized)
    sid_score = silhouette_score(X_normalized, ac.labels_)

    #print("Numberpoints = "+ str(len(points))  +f" Siddihose = " + str(sid_score) + '   Numclusster = ' + str(K))
    return ac, sid_score


    


def get_optimal_numcluster(points):
    best_model, best_score = agglomerative_clustering(points, 2)
    best_k = 2

    end = min(6, int(len(points)/2))
    for i in range(2, end):
        ac, score = agglomerative_clustering(points, i)
        if(score > best_score):
            best_model = ac
            best_score = score
            best_k = i
    return best_model, best_k




    


