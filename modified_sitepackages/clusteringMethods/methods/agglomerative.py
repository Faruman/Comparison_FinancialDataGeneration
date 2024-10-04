import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from .base import clusteringMethod
from ..combinator import best_combinator, hungarian_combinator, count_combinator

class agglomerative(clusteringMethod):
    def __init__(self, iterations= 100, num_cluster_range= range(2, 16), metric= "elbow", optimum_type= "montecarlo", linkage= "ward", combinator= "hungarian", random_state= 42, **kwargs):
        super().__init__(metric, random_state)
        self.iterations = iterations
        self.num_cluster_range = num_cluster_range
        self.mc_num_clusters = np.array([])
        self.mc_labels = []
        self.combinator = self._get_combinator(combinator)
        self.nonbinary = None
        self.optimum_type = optimum_type
        self.linkage = linkage

        self.use_for_preperation = True
        self.use_for_clustering = False

    def fit(self, X, **kwargs):
        for num_clusters in tqdm(self.num_cluster_range):
            for i in range(self.iterations):
                t_X = np.concatenate((np.arange(X.shape[0]).reshape(-1, 1), X), axis=1)
                np.random.seed(self.random_state + i)
                np.random.shuffle(t_X)
                labels_idx = t_X[:, 0]
                t_X = t_X[:, 1:]
                model = AgglomerativeClustering(n_clusters=num_clusters, linkage= self.linkage)
                labels = model.fit_predict(t_X)
                labels_oh = np.zeros((labels.size, num_clusters))
                labels_oh[np.arange(labels.size), labels] = 1
                self.mc_labels.append(np.expand_dims(labels_oh[np.argsort(labels_idx)], axis=0))
                self.mc_num_clusters = np.append(self.mc_num_clusters, num_clusters)
                self.metric.calculate_step(t_X, labels, num_clusters)
        self.num_clusters, score = self.metric.find_optimum(optimum_type= self.optimum_type)
        self.labels = self.combinator.combine(np.concatenate([x for i, x in enumerate(self.mc_labels) if (self.mc_num_clusters == self.num_clusters)[i]]), self.metric.scores[self.metric.scores_idx == self.num_clusters])
        self.labels = np.argmax(self.labels, axis=1)
        self.model = KNeighborsClassifier(n_neighbors=1)
        self.model.fit(X, self.labels)
        self.cluster_centers = NearestCentroid().fit(X, self.labels).centroids_

    def predict(self, X):
        if self.model:
            X = self.preprocess(X, transform_to= "onehot")
            return self.model.predict(X)
        else:
            ValueError("model not fitted")

    def _get_combinator(self, combinator: str):
        if combinator.lower() == "hungarian":
            return hungarian_combinator()
        elif combinator.lower() == "best":
            return best_combinator(type= self.metric.optimum_type)
        elif combinator.lower() == "count":
            return count_combinator()
        else:
            raise NotImplementedError()
