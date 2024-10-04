from .base import metric

import numpy as np

from clusteringMethods.distances import jiangCheungDistance

class daviesBouldinIndex(metric):
    def __init__(self, distance_metric):
        super().__init__(distance_metric)

    def calculate(self, X, labels, centroids = None):
        X = np.array(X).astype(float)
        self.distance_metric.reset_global_vars()
        if centroids is None:
            centroids = self._calculate_cluster_centers(X, labels)
        c = len(np.unique(labels))
        score = 0
        within_cluster_distances = {}
        for c_i in centroids.keys():
            within_cluster_distances[c_i] = 0
            for x in X[np.where(labels == c_i)]:
                within_cluster_distances[c_i] += self.distance_metric.calculate(x, centroids[c_i], X= X) ** 2

        for c_i in centroids.keys():
            max_score = 0
            for c_j in centroids.keys():
                if c_i != c_j:
                    score = (within_cluster_distances[c_i] + within_cluster_distances[c_j]) / self.distance_metric.calculate(centroids[c_i], centroids[c_j], X= X)
                    if score > max_score:
                        max_score = score
            score += max_score
        return score / c

if __name__ == '__main__':
    dummy_array = np.array([[0, 0, 0, 0, 0],
                            [1, 0, 1, 0, 1],
                            [0, 0, 0, 0, 1],
                            [1, 1, 1, 0, 0]])
    dummy_labels = np.array([0, 1, 0, 1])
    mymetric = daviesBouldinIndex(jiangCheungDistance)
    print(mymetric.calculate(dummy_array, dummy_labels))