import numpy as np
from .base import metric
from c_index import calc_cindex_clusterSim_implementation, pdist_array

class cindex_metric(metric):
    def __init__(self):
        super().__init__()
        self.optimum_type = "max"

    def calculate(self, X, labels):
        return calc_cindex_clusterSim_implementation(pdist_array(X), labels)

    def _find_optimum(self, scores, scores_idx):
        idx = np.argmax(scores)
        return scores_idx[idx], scores[idx]
