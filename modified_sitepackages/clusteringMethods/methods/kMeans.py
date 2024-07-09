import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from .base import clusteringMethod
from ..combinator import best_combinator, hungarian_combinator, restrictedKmeans_combinator

class kMeans(clusteringMethod):
    def __init__(self, iterations= 100, num_cluster_range= range(2, 16), metric= "elbow", combinator= "best", optimum_type= "montecarlo", random_state= 42, **kwargs):
        super().__init__(metric, random_state)
        self.iterations = iterations
        self.num_cluster_range = num_cluster_range
        self.mc_clusters = []
        self.mc_num_clusters = np.array([])
        self.combinator = self._get_combinator(combinator)
        self.optimum_type = optimum_type

        self.use_for_preperation = True
        self.use_for_clustering = True

    def fit(self, X, **kwargs):
        for num_clusters in tqdm(self.num_cluster_range):
            for i in range(self.iterations):
                model = KMeans(n_clusters=num_clusters, n_init=3, max_iter=1000, random_state= self.random_state + i)
                model.fit(X)
                labels = model.predict(X)
                self.mc_clusters.append(np.expand_dims(model.cluster_centers_, axis= 0))
                self.mc_num_clusters = np.append(self.mc_num_clusters, num_clusters)
                self.metric.calculate_step(X, labels, num_clusters)
        self.num_clusters, score = self.metric.find_optimum(optimum_type= self.optimum_type)
        self.cluster_centers = self.combinator.combine(np.concatenate([x for i, x in enumerate(self.mc_clusters) if (self.mc_num_clusters == self.num_clusters)[i]]), self.metric.scores[self.metric.scores_idx == self.num_clusters])
        self.model = KMeans(n_clusters= int(self.num_clusters), n_init=1, max_iter=1000, init= self.cluster_centers)
        self.model.fit(self.cluster_centers)

    def _fit_with_n_clusters(self, X, num_clusters, cluster_centers= None, **kwargs):
        self.num_clusters = num_clusters
        X = self.preprocess(X, transform_to="onehot")
        if cluster_centers is not None:
            cluster_centers = self.preprocess(cluster_centers, transform_to="onehot", fit=False)
            self.model = KMeans(n_clusters=int(self.num_clusters), n_init=1, max_iter=1000, init=cluster_centers)
            self.model.fit(X)
        else:
            for i in range(self.iterations):
                model = KMeans(n_clusters= int(self.num_clusters), n_init=3, max_iter=1000, random_state=self.random_state + i)
                model.fit(X)
                labels = model.predict(X)
                self.mc_clusters.append(np.expand_dims(model.cluster_centers_, axis=0))
                self.mc_num_clusters = np.append(self.mc_num_clusters, num_clusters)
                self.metric.calculate_step(X, labels, num_clusters)
            self.cluster_centers = self.combinator.combine(np.concatenate([x for i, x in enumerate(self.mc_clusters) if (self.mc_num_clusters == self.num_clusters)[i]]), self.metric.scores[self.metric.scores_idx == self.num_clusters])
            self.model = KMeans(n_clusters=int(self.num_clusters), n_init=3, max_iter=1000, init=self.cluster_centers)
            self.model.fit(self.cluster_centers)
        self.cluster_centers = self.model.cluster_centers_

    def predict(self, X):
        if self.model:
            X = self.preprocess(X, transform_to= "onehot", fit= False)
            return self.model.predict(X)
        else:
            ValueError("model not fitted")

    def _get_combinator(self, combinator: str):
        if combinator.lower() == "hungarian":
            return hungarian_combinator()
        elif combinator.lower() == "restrictedkmeans":
            return restrictedKmeans_combinator()
        elif combinator.lower() == "best":
            return best_combinator()
        else:
            raise NotImplementedError()
