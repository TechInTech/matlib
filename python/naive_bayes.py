import numpy as np
from base import BaseEstimator


class NaiveBayes(BaseEstimator):

    def __init__(self, dispersed=True):
        self.dispersed = dispersed

    def fit(self, X, y):
        self._setup_input(X, y)
        if self.dispersed:
            unique = np.unique(self.y)
            ulen = len(unique)
            # dt = np.dtype(dict(names=['yv', 'yp'], formats=[self.y.dtype.name, 'f32']))
            PY = {}
            PXY = {}
            xu = []
            for i in range(self.n_features):
                xu.append(np.unique(self.X[:, i]))
            for yi in unique:
                PY[yi] = np.mean(self.y == yi)
                PXY[yi] = []
            # PY = np.array(PY.items(), dtype=dt)
            for yi in unique:
                for j in range(self.n_features):
                    x = self.X[self.y == yi, j]
                    features = {}
                    for xi in xu[j]:
                        features[xi] = (np.sum(x == xi) + 1.0) / \
                            (len(x) + len(xu[j]) * 1.0)
                    PXY[yi].append(features)
            self.py = PY
            self.pxy = PXY
        else:
            unique = np.unique(self.y)
            self.means = {}
            self.covs = {}
            self.priors = {}
            for yi in unique:
                X_yi = self.X[self.y == yi]
                self.means[yi] = X_yi.mean(axis=0)
                self.covs[yi] = X_yi.var(axis=0)
                self.priors[yi] = X_yi.shape[0] / self.y.shape[0]

    def _predict(self, X):
        if self.dispersed:
            prob = {}
            for yi in np.unique(self.y):
                prob[yi] = np.log(self.py[yi])
                for i in range(self.n_features):
                    prob[yi] += np.log(self.pxy[yi][i][X[i]])
            return max(prob, key=prob.get)
        else:
            prob = {}
            for yi in np.unique(self.y):
                prob[yi] = np.log(self.priors[yi])
                prob[yi] += self._pdf(X, yi).sum()
            return max(prob, key=prob.get)

    def _pdf(self, X, yi):
        return np.exp(-(X - self.means[yi])**2 / (self.covs[yi] * 2)) / np.sqrt(2 * np.pi * self.covs[yi])
