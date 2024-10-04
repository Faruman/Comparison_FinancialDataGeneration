from sklearn.cluster import KMeans

class combinator:
    def __init__(self):
        self.optim_cluster_center = None
        self.model = None

    def combine(self, cluster_centers, scores):
        raise NotImplementedError()

    def build_model(self):
        if self.optim_cluster_center:
            self.model = KMeans(n_clusters=2, init= self.optim_cluster_center, n_init=1)
            self.model.fit(self.optim_cluster_center)
            return self.model
        else:
            raise ValueError("Combine needs to be called first")
