import numpy as np
from scipy.spatial.distance import cdist
from .base import metric
from kneed import KneeLocator

class elbow_metric(metric):
    def __init__(self):
        super().__init__()
        self.optimum_type = "min"

    def calculate(self, X, labels):
        centroids = self._calculate_cluster_centers(X, labels)
        return sum(np.min(cdist(X, centroids, "euclidean"), axis=1)) / len(X)

    def _find_optimum(self, scores, scores_idx):
        # TODO: Implement Error Handler for kneeLocator
        i = 0
        opt_idx = None
        while i < 10 and opt_idx is None:
            opt_idx = KneeLocator(scores_idx, scores, S=1-(0.1*i), curve="convex", direction="decreasing").knee
            i += 1
        if opt_idx is None:
            raise ValueError("No optimum found")
        else:
            return opt_idx, scores[np.where(scores_idx == opt_idx)]
