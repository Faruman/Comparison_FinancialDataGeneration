import numpy as np
from .base import metric
from sklearn.metrics import silhouette_score

class silhouette_metric(metric):
    def __init__(self):
        super().__init__()
        self.optimum_type = "max"

    def calculate(self, X, labels):
        return silhouette_score(X, labels)

    def _find_optimum(self, scores, scores_idx):
        idx = np.argmax(scores)
        return scores_idx[idx], scores[idx]
