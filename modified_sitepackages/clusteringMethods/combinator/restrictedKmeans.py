import numpy as np
from k_means_constrained import KMeansConstrained
from .base import combinator


class restrictedKmeans_combinator(combinator):
    def __init__(self):
        super().__init__()

    def combine(self, cluster_centers, scores= None):
        self.model = KMeansConstrained(n_clusters= len(cluster_centers[0]), size_min= len(cluster_centers), size_max= len(cluster_centers), random_state=0)
        self.model.fit(np.concatenate(cluster_centers, axis=0))
        self.optim_cluster_center = self.model.cluster_centers_
        return self.optim_cluster_center

    def build_model(self):
        if self.model:
            return self.model
        else:
            raise ValueError("Combine needs to be called first")