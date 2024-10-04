import numpy as np
from sklearn.metrics import pairwise_distances
import seaborn as sns
from matplotlib import pyplot as plt

class metric(object):
    def __init__(self, distance_metric):
        self.score = None
        self.distance_metric = distance_metric()

    def calculate(self, X, labels):
        raise NotImplementedError()

    def _calculate_distance_matrix(self, X_cluster, X_complete):
        if self.distance_metric.need_complete_data:
            self.distance_metric.preset_x(X_complete)
        return pairwise_distances(X_cluster, metric=self.distance_metric.calculate)

    def _calculate_cluster_centers(self, X, labels):
        cluster_centers = {}
        for label in np.unique(labels):
            label_idx = np.where(labels == label)
            X_label = X[label_idx]
            dist_matrix = self._calculate_distance_matrix(X_label, X)
            X_idx = np.sum(dist_matrix, axis=1).argmin()
            cluster_centers[label] = X_label[X_idx]
        return cluster_centers
