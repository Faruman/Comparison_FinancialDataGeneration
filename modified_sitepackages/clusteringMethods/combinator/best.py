import numpy as np
from .base import combinator


class best_combinator(combinator):
    def __init__(self, type= "max"):
        super().__init__()
        self.type = type

    def combine(self, cluster_centers, scores):
        if self.type == "max":
            self.optim_cluster_center = cluster_centers[np.argmax(scores)]
        else:
            self.optim_cluster_center = cluster_centers[np.argmin(scores)]
        return self.optim_cluster_center