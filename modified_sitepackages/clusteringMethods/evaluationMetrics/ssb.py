from .base import metric

import numpy as np
from sklearn.metrics import pairwise_distances

from clusteringMethods.distances import jiangCheungDistance

# implements an easy sepreation metric, the sum of squares between clusters
class ssb(metric):
    def __init__(self, distance_metric):
        super().__init__(distance_metric)

    def calculate(self, X, labels, centroids = None):
        X = np.array(X).astype(float)
        self.distance_metric.reset_global_vars()
        if self.distance_metric.need_complete_data:
            self.distance_metric.preset_x(X)
        if centroids is None:
            centroids = self._calculate_cluster_centers(X, labels)
        score = 0
        K = len(np.unique(labels))
        for c_i in centroids.keys():
            m_i = np.sum(labels == c_i)
            for c_j in centroids.keys():
                if c_i != c_j:
                    score += (m_i / K) / self.distance_metric.calculate(centroids[c_i], centroids[c_j], X= X) ** 2
        return score

if __name__ == '__main__':
    dummy_array = np.array([[0, 0, 0, 0, 0],
                            [1, 0, 1, 0, 1],
                            [0, 0, 0, 0, 1],
                            [1, 1, 1, 0, 0]])
    dummy_labels = np.array([0, 1, 0, 1])
    mymetric = ssb(jiangCheungDistance)
    mymetric.calculate(dummy_array, dummy_labels)