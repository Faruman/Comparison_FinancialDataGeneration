from .base import metric

import numpy as np
from sklearn.metrics import silhouette_score

from clusteringMethods.distances import jiangCheungDistance

class silhouetteIndex(metric):
    def __init__(self, distance_metric):
        super().__init__(distance_metric)

    def calculate(self, X, labels):
        X = np.array(X).astype(float)
        self.distance_metric.reset_global_vars()
        if self.distance_metric.need_complete_data:
            self.distance_metric.preset_x(X)
        return silhouette_score(X, labels, metric=self.distance_metric.calculate)

if __name__ == '__main__':
    dummy_array = np.array([[0, 0, 0, 0, 0],
                            [1, 0, 1, 0, 1],
                            [0, 0, 0, 0, 1],
                            [1, 1, 1, 0, 0]])
    dummy_labels = np.array([0, 1, 0, 1])
    mymetric = silhouetteIndex(jiangCheungDistance)
    print(mymetric.calculate(dummy_array, dummy_labels))