import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

class metric(object):
    def __init__(self):
        self.score = None
        self.scores = np.array([])
        self.scores_idx = np.array([])
        self.calculator = None
        self.optimum = None
        self.optimums = []
        self.optimum_type = None
        self.type = None

    def calculate(self, X, labels):
        raise NotImplementedError()

    def calculate_step(self, X, labels, idx= None):
        self.scores = np.append(self.scores, self.calculate(X, labels))
        if idx:
            self.scores_idx = np.append(self.scores_idx, idx)
        else:
            self.scores_idx = np.append(self.scores_idx, len(self.scores))

    def find_optimum(self, optimum_type = "average"):
        self.optimum_type = optimum_type
        if self.optimum_type == "average":
            self.scores, self.scores_idx = self._avg_scores()
            self.optimum, self.score = self._find_optimum(self.scores, self.scores_idx)
            return self._find_optimum(self.scores, self.scores_idx)
        elif self.optimum_type == "montecarlo":
            scores, scores_idx = self._group_scores()
            self.optimums = []
            for i in range(scores.shape[1]):
                self.optimums.append(self._find_optimum(scores[:, i], scores_idx[:, i])[0])
            optimums_idx, optimums_count = np.unique(np.array([x for x in self.optimums if x != None]), return_counts=True)
            self.optimum = optimums_idx[np.argmax(optimums_count)]
            self.score = np.mean(scores[np.where(scores_idx == self.optimums[np.argmax(optimums_count)])])
            return self.optimum, self.score
        else:
            raise NotImplementedError()

    def _find_optimum(self, scores, scores_idx):
        raise NotImplementedError()

    def _calculate_cluster_centers(self, X, labels):
        sidx = np.argsort(labels)
        labels = labels[sidx]
        X = X[sidx]
        X = np.split(X, np.flatnonzero(labels[1:] != labels[:-1]) + 1)
        return np.array([np.mean(x, axis=0) for x in X])

    def plot(self):
        if self.scores is None:
            raise ValueError("No scores calculated")
        elif self.optimum is None:
            raise ValueError("No optimum calculated")
        else:
            if self.optimum_type == "average":
                g = sns.lineplot(x=self.scores_idx, y=self.scores, color= 'b')
                g.axvline(self.optimum, color='r', linestyle='--')
                return g
            elif self.optimum_type == "montecarlo":
                fig, axs = plt.subplots(ncols=2)
                iterations = int(len(self.scores_idx)/len(np.unique(self.scores_idx)))
                sns.lineplot(x=self.scores_idx, y=self.scores, hue=np.array(list(range(iterations))*int(len(self.scores_idx)/iterations)), ax= axs[0])
                sns.histplot(self.optimums, ax= axs[1])

            else:
                raise NotImplementedError()

    def _group_scores(self):
        sidx = np.argsort(self.scores_idx)
        scores_idx = self.scores_idx[sidx]
        scores = self.scores[sidx]
        scores = np.split(scores, np.flatnonzero(scores_idx[1:] != scores_idx[:-1]) + 1)
        scores_idx = np.split(scores_idx, np.flatnonzero(scores_idx[1:] != scores_idx[:-1]) + 1)
        scores = np.concatenate(np.expand_dims(scores, axis=0))
        scores_idx = np.concatenate(np.expand_dims(scores_idx, axis=0))
        return scores, scores_idx

    def _avg_scores(self):
        scores, scores_idx = self._group_scores()
        scores = scores.mean(axis= 1)
        scores_idx = scores_idx.mean(axis= 1)
        return scores, scores_idx
