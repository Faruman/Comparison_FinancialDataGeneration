import numpy as np
from .base import metric

class gapstatistic_metric(metric):
    def __init__(self, clustering_algo, n_refs=5):
        super().__init__()
        self.clustering_algo = clustering_algo
        self.nrefs = n_refs
        self.optimum_type = "max"

    def _calculate(self, X, labels):
        # Holder for reference dispersion results
        ref_dispersions = np.zeros(self.nrefs)

        # Compute the range of each feature
        X = np.asarray(X)
        a, b = X.min(axis=0, keepdims=True), X.max(axis=0, keepdims=True)

        # For n_references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(self.nrefs):
            # Create new random reference set uniformly over the range of each feature
            random_data = self._rs.random_sample(size=X.shape) * (b - a) + a

            # Fit to it, getting the centroids and labels, and add to accumulated reference dispersions array.
            labels = self.clustering_algo.fit_predict(random_data)
            centroids = self.clustering_algo.cluster_centers_
            dispersion = self._calculate_dispersion(
                X=random_data, labels=labels, centroids=centroids
            )
            ref_dispersions[i] = dispersion

        # Fit cluster to original data and create dispersion calc.
        centroids = self._calculate_cluster_centers(X=X, labels=labels)
        dispersion = self._calculate_dispersion(X=X, labels=labels, centroids=centroids)

        # Calculate gap statistic
        ref_log_dispersion = np.mean(np.log(ref_dispersions))
        log_dispersion = np.log(dispersion)
        gap_value = ref_log_dispersion - log_dispersion
        # compute standard deviation
        sdk = np.sqrt(np.mean((np.log(ref_dispersions) - ref_log_dispersion) ** 2.0))
        sk = np.sqrt(1.0 + 1.0 / self.nrefs) * sdk

        # Calculate Gap* statistic
        # by "A comparison of Gap statistic definitions with and
        # with-out logarithm function"
        # https://core.ac.uk/download/pdf/12172514.pdf
        gap_star = np.mean(ref_dispersions) - dispersion
        sdk_star = np.sqrt(np.mean((ref_dispersions - dispersion) ** 2.0))
        sk_star = np.sqrt(1.0 + 1.0 / self.nrefs) * sdk_star
        return gap_value, ref_dispersions.std(), sdk, sk, gap_star, sk_star

    def calculate(self, X, labels):
        gap_value,_, _, _, _, _ = self._calculate(X, labels)
        return gap_value

    def _find_optimum(self, scores, scores_idx):
        idx = np.argmax(scores)
        return scores_idx[idx], scores[idx]
