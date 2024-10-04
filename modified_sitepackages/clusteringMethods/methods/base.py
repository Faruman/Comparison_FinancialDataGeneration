import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# import selectionMetrics
from ..selectionMetrics import cindex_metric, silhouette_metric, elbow_metric, gapstatistic_metric

class clusteringMethod():
    def __init__(self, metric: str, random_state):
        self.num_clusters = None
        self.cluster_centers = None
        self.model = None
        self.random_state = random_state
        self.metric = self._get_metric_calc(metric)
        self.nonbinary = None
        self.X_columns = None
        self.transform_to = None

    def preprocess(self, X, transform_to= "onehot", fit= True, categorical_array= np.array([])):
        self.transform_to = transform_to
        self.categorical_array = categorical_array
        if isinstance(X, pd.DataFrame):
            self.X_columns = X.columns
            X = X.values
        if transform_to == "onehot":
            if fit:
                self.nonbinary = np.apply_along_axis(lambda x: len(np.unique(x)), 0, X) > 2
                self.column_order = np.append(np.argwhere(~self.nonbinary), np.argwhere(self.nonbinary))
            X = np.concatenate((X[:, ~self.nonbinary], self._encodeOneHot(X[:, self.nonbinary], fit= fit)), axis= 1)
        elif transform_to == "categorical":
            if len(categorical_array) >0:
                if len(np.concatenate(categorical_array)) > X.shape[1]:
                    raise ValueError("too many columns in categorical array")
                elif len(np.concatenate(categorical_array)) < X.shape[1]:
                    raise ValueError("too few columns in categorical array")
                temp = np.empty((len(X), 0))
                for categorical in categorical_array:
                    temp = np.concatenate((temp, np.expand_dims(np.apply_along_axis(lambda x: x.dot(2**np.arange(x.size)[::-1]), 1, X[:, categorical]), axis= 1)), axis= 1)
                X = temp.astype(int)
        return X

    def reverse_preprocess(self, X):
        if self.transform_to is not None:
            if self.transform_to == "onehot":
                X = np.concatenate((X[:, :sum(~self.nonbinary)], self._decodeOneHot(X[:, sum(~self.nonbinary):])), axis= 1)
                X = X[:, self.column_order]
            elif self.transform_to == "categorical":
                if len(self.categorical_array) > 0:
                    temp = np.empty((len(X), 0))
                    for i, categorical in enumerate(self.categorical_array):
                        temp = np.concatenate((temp, np.apply_along_axis(lambda x: np.array(list(np.base_repr(int(x)).zfill(categorical.size))).astype(int), 1, np.expand_dims(X[:, i],axis= 1))), axis=1)
                    X = temp
            if self.X_columns is not None:
                X = pd.DataFrame(X, columns= self.X_columns)
            return X
        else:
            raise ValueError(".preprocess needs to be run before it can be reversed")


    def fit(self, X, **kwargs):
        raise NotImplementedError()

    def _fit_with_n_clusters(self, X, num_clusters, **kwargs):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def get_params(self):
        if hasattr(self, "use_for_preperation") and hasattr(self, "use_for_clustering"):
            return {"num_clusters": self.num_clusters, "cluster_centers": self.cluster_centers,"use_for_preperation": self.use_for_preperation, "use_for_clustering": self.use_for_clustering}
        elif hasattr(self, "use_for_preperation"):
            return {"num_clusters": self.num_clusters, "cluster_centers": self.cluster_centers,"use_for_preperation": self.use_for_preperation}
        elif hasattr(self, "use_for_clustering"):
            return {"num_clusters": self.num_clusters, "cluster_centers": self.cluster_centers,"use_for_clustering": self.use_for_clustering}
        else:
            return {"num_clusters": self.num_clusters, "cluster_centers": self.cluster_centers}

    def get_cluster_centers(self):
        if self.cluster_centers is not None:
            return self.reverse_preprocess(self.cluster_centers)
        else:
            raise ValueError("model not fitted")

    def _calculate_cluster_centers(self, X, labels):
        sidx = np.argsort(labels)
        labels = labels[sidx]
        X = X[sidx]
        X = np.split(X, np.flatnonzero(labels[1:] != labels[:-1])+1)
        return np.array([np.mean(x, axis=0) for x in X])

    def _encodeOneHot(self, X, fit= True):
        if X.size > 0:
            if fit:
                self.dataEncoder = ColumnTransformer(transformers= [('encoder', OneHotEncoder(), list(range(X.ndim + 1)))], remainder='passthrough')
                return self.dataEncoder.fit_transform(X)
            else:
                return self.dataEncoder.transform(X)
        else:
            return X

    def _decodeOneHot(self, X):
        if X.size > 0:
            return self.dataEncoder.named_transformers_['encoder'].inverse_transform(X)
        else:
            return X

    def _encodeLabel(self, X):
        self.dataEncoder = ColumnTransformer(transformers= [('encoder', LabelEncoder(), list(range(X.ndim + 1)))], remainder='passthrough')
        return self.dataEncoder.fit_transform(X)

    def _get_metric_calc(self, metric: str):
        if metric.lower() == "elbow":
            return elbow_metric()
        elif metric.lower() == "cindex":
            return cindex_metric()
        elif metric.lower() == "silhouette":
            return silhouette_metric()
        elif metric.lower() == "gapstatistic":
            return gapstatistic_metric(clustering_algo= self.model.copy())
        else:
            raise NotImplementedError()


