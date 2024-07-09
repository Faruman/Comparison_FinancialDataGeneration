from .base import metric

import numpy as np
from sklearn.metrics import pairwise_distances

from clusteringMethods.distances import jiangCheungDistance


class dunnIndex(metric):
    def __init__(self, distance_metric):
        super().__init__(distance_metric)

    def calculate(self, X, labels, centroids = None):
        X = np.array(X).astype(float)
        self.distance_metric.reset_global_vars()
        if centroids is None:
            centroids = self._calculate_cluster_centers(X, labels)
        numer = float('inf')
        denom = 0
        for c_i in centroids.keys():  # for each cluster
            for c_j in centroids.keys():  # for each cluster
                if c_i != c_j:
                    for p1 in X[np.where(labels == c_i)]:
                        for p2 in X[np.where(labels == c_j)]:
                            interdis = self.distance_metric.calculate(p1, p2, X= X)
                            numer = min(numer, interdis)
            for p1 in X[np.where(labels == c_i)]:
                for p3 in X[np.where(labels == c_i)]:
                    intradis = self.distance_metric.calculate(p1, p3, X= X)
                    denom = max(denom, intradis)
        return numer / denom


if __name__ == '__main__':
    dummy_array = np.array([[0, 0, 0, 0, 0],
                            [1, 0, 1, 0, 1],
                            [0, 0, 0, 0, 1],
                            [1, 1, 1, 0, 0]])
    dummy_labels = np.array([0, 1, 0, 1])
    mymetric = dunnIndex(jiangCheungDistance)
    print(mymetric.calculate(dummy_array, dummy_labels))