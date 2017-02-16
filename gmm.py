import numpy as np
from base import BaseEstimator
from scipy.stats import multivariate_normal
import random
from kmeans import KMeans


class GMM(BaseEstimator):
    y_required = False

    def __init__(self, init='random', n_clusters=2, max_iters=1000, torlerance=1e-3):
        self.max_iters = max_iters
        self.n_clusters = n_clusters
        self.init = init
        self.torlerance = torlerance
        self.covs = []
        self.mean = []

    def fit(self, X):
        self._setup_input(X)
        self.weights = np.ones(shape=(1, self.n_clusters))
        if self.init == 'random':
            # (k, n)
            self.mean = [self.X[x] for x in random.sample(
                range(self.n_samples), self.n_clusters)]
            # (k, n, n)
            self.covs = [np.cov(self.X.T) for _ in range(self.n_clusters)]
        elif self.init == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_clusters,
                            max_iters=self.max_iters // 3, init='++')
            kmeans.fit(self.X)
            self.assignments = kmeans.predict()
            self.mean = kmeans.centroids
            self.covs = []
            for i in np.unique(self.assignments):
                self.weights[0][int(i)] = (self.assignments == i).sum()
                self.covs.append(np.cov(self.X[self.assignments == i].T))
        else:
            raise NotImplementedError()

        self.weights = self.weights / self.weights.sum()
        # (m, k)
        self.probs = np.zeros(
            shape=(self.n_samples, self.n_clusters), dtype=np.float)
        self.weighted_probs = np.zeros(
            shape=(self.n_samples, self.n_clusters), dtype=np.float)
        self.sum_prob = []
        for iter_num in range(self.max_iters):
            self._E_step()
            self._M_step()
            if len(self.sum_prob) > 1 and np.abs(self.sum_prob[-1] - self.sum_prob[-2]) < self.torlerance:
                break
        print(self.sum_prob[0:10])

    def _E_step(self):
        probs = self._get_prob(self.X)
        self.sum_prob.append(probs.sum())
        self.weighted_probs = self.weights * probs
        self.assignments = self.weighted_probs.argmax(axis=1)
        sums = np.sum(self.weighted_probs, axis=1)
        self.weighted_probs = self.weighted_probs / sums[:, np.newaxis]

    def _get_prob(self, data):
        probs = np.zeros(
            shape=(data.shape[0], self.n_clusters), dtype=np.float)
        for c in range(self.n_clusters):
            probs[:, c] = multivariate_normal.pdf(
                data, self.mean[c], self.covs[c])
        return probs

    def _M_step(self):
        self.weights = self.weighted_probs.sum(axis=0)
        # (m, k) => (1,k)
        for c in range(self.n_clusters):
            #(1,n)
            resp = self.weighted_probs[:, c][:,np.newaxis]
            self.mean[c] = (resp*self.X).sum(axis=0)/resp.sum()
            #(1,n)
            self.covs[c] = (self.X-self.mean[c]).T.dot((self.X-self.mean[c])*resp)/resp.sum()
        self.weights /= self.weights.sum()

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        probs = self.weights * self._get_prob(X)
        return probs.argmax(axis=1)
