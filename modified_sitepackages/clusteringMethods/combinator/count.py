from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from scipy.optimize import linear_sum_assignment
from .base import combinator
import numpy as np


class count_combinator(combinator):
    def __init__(self):
        super().__init__()

    def combine(self, cluster_centers, scores= None):
        self.optim_cluster_center = cluster_centers[0]
        k_sub = 1
        for k, cluster_center in enumerate(cluster_centers[1:]):
            k += k_sub
            num_cluster_centers = cluster_center.shape[1]
            cluster_center_count = np.zeros(cluster_center.shape[1])
            optim_centroid_count = np.zeros(cluster_center.shape[1])
            cluster_center_idx, cluster_center_count[:len(cluster_center_idx)] = np.unique(cluster_center.argmax(axis= 1), return_counts= True)
            cluster_center_idx = np.append(cluster_center_idx, np.where(~np.in1d(np.arange(num_cluster_centers), cluster_center_idx))[0])
            optim_cluster_center_idx, optim_centroid_count[:len(optim_cluster_center_idx)] = np.unique(self.optim_cluster_center.argmax(axis= 1), return_counts= True)
            optim_cluster_center_idx = np.append(optim_cluster_center_idx, np.where(~np.in1d(np.arange(num_cluster_centers), optim_cluster_center_idx))[0])
            cluster_center_idx = cluster_center_idx[np.argsort(cluster_center_count)][::-1]
            optim_cluster_center_idx = optim_cluster_center_idx[np.argsort(optim_centroid_count)][::-1]
            self.optim_cluster_center = self.optim_cluster_center[:, optim_cluster_center_idx] * ((k - 1) / k) + cluster_center[:, cluster_center_idx] * 1 / k
        return self.optim_cluster_center