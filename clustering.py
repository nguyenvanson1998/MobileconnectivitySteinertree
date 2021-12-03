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


def visualize(list_points):
    arr = []
    for p in list_points:
        arr.append([p.x, p.y])
    dendogram = shc.dendrogram(shc.linkage(arr, method = 'ward'))
    return dendogram


def agglomerative_clustering(points, K):
    scaler = StandardScaler()
    points_scaler = scaler.fit_transform(points)
    X_normalized = normalize(points_scaler)

    ac = AgglomerativeClustering(n_clusters= K, linkage= 'ward').fit(X_normalized)
    sid_score = silhouette_score(X_normalized, ac.labels_)
    return ac, sid_score


    


def get_optimal_numcluster(points):
    best_model, best_score = agglomerative_clustering(points, 2)
    best_k = 2
    for i in range(3, 9):
        ac, score = agglomerative_clustering(points, i)
        if(score > best_score):
            best_score = ac
            best_score = score
            best_k = i
    return best_model, best_k




    


