from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from scipy.optimize import linear_sum_assignment
from .base import combinator


class hungarian_combinator(combinator):
    def __init__(self):
        super().__init__()

    def combine(self, cluster_centers, scores= None):
        self.optim_cluster_center = cluster_centers[0]
        k_sub = 1
        for k, centroid in enumerate(cluster_centers[1:]):
            k += k_sub
            centroids_dists = manhattan_distances(self.optim_cluster_center, centroid) + 0.000001
            _, centroid_idx = linear_sum_assignment(centroids_dists)
            self.optim_cluster_center = self.optim_cluster_center * ((k - 1) / k) + centroid[centroid_idx] * 1 / k
        return self.optim_cluster_center