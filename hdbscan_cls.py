from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import hdbscan
import random as rd
import numpy as np


def hdbscan_clustering(points):
    scaler = StandardScaler()
    points_scaler = scaler.fit_transform(points)
    X_normalized = normalize(points_scaler)
    min_cluster_size = np.random.randint(len(points)//40, len(points)//25+1)
    if min_cluster_size <2:
        min_cluster_size = 2
    if min_cluster_size < 5:
        min_samples = 2
    else:
        min_samples = 15
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(X_normalized)
    labels = clusterer.labels_
    # get number cluster
    numcluster = len(set(labels))
    return clusterer.labels_, numcluster
    

############ testing ###############
# if __name__ =='__main__':
#     points = [[1,1], [4,2], [5,3], [10,10], [16,11], [14,12]]
#     x, number = hdbscan_clustering(points)
#     print(x)
