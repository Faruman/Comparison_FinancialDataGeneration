
from ..methods.base import clusteringMethod

class dualClustering(object):
    def __init__(self, preperation_method:clusteringMethod, clustering_method:clusteringMethod, prepare_cluster_centers:bool = False):
        if hasattr(preperation_method, 'use_for_preperation') and preperation_method.use_for_preperation:
            self.preperation_method = preperation_method
        else:
            raise ValueError("clusteringMethod choosen for preperation is not suitable")
        if hasattr(clustering_method, 'use_for_clustering') and clustering_method.use_for_clustering:
            self.clustering_method = clustering_method
        else:
            raise ValueError("clusteringMethod choosen for clustering is not suitable")
        self.prepare_cluster_centers = prepare_cluster_centers
        self.num_clusters = None
        self.cluster_centers = None

    def fit(self, X, **kwargs):
        self.preperation_method.fit(X, **kwargs)
        self.num_clusters = self.preperation_method.num_clusters
        if self.prepare_cluster_centers:
            self.cluster_centers = self.preperation_method.get_cluster_centers()
            self.clustering_method._fit_with_n_clusters(X, self.num_clusters, cluster_centers= self.cluster_centers, **kwargs)
        else:
            self.clustering_method._fit_with_n_clusters(X, self.num_clusters, **kwargs)
        self.cluster_centers = self.clustering_method.get_cluster_centers()

    def predict(self, X):
        return self.clustering_method.predict(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def get_params(self):
        return {"num_clusters": self.num_clusters, "cluster_centers": self.cluster_centers}