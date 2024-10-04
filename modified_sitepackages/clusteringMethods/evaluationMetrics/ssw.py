from .base import metric

import numpy as np
from sklearn.metrics import pairwise_distances

from clusteringMethods.distances import jiangCheungDistance

# implements an easy cohesion metric, the sum of squares within clusters
class ssw(metric):
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
        for c_i in centroids.keys():
            m_i = np.sum(labels == c_i)
            c_idx = np.where(labels == c_i)
            score += (1/(2*m_i)) * sum(sum(pairwise_distances(X[c_idx], metric=self.distance_metric.calculate)**2))
        return score

if __name__ == '__main__':
    dummy_array = np.array([[0, 0, 0, 0, 0],
                            [1, 0, 1, 0, 1],
                            [0, 0, 0, 0, 1],
                            [1, 1, 1, 0, 0]])
    dummy_labels = np.array([0, 1, 0, 1])
    mymetric = ssw(jiangCheungDistance)
    mymetric.calculate(dummy_array, dummy_labels)