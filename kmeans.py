import numpy as np
from base import BaseEstimator
import random
from metrics import euclidean_distance, manhaton_distance

class KMeans(BaseEstimator):
    y_required = False
    centroids = []

    def __init__(self, n_clusters=2, max_iters=500, torlerance=1e-4, init='random', distance_func=euclidean_distance):
        self.n_clusters=n_clusters
        self.max_iters=max_iters
        self.torlerance = torlerance
        self.init = init
        self.distance_func = distance_func

    def fit(self, X):
        self._setup_input(X)
        centroids = []
        if self.init == 'random':
            centroids = [self.X[x] for x in random.sample(range(self.n_samples), self.n_clusters)]
        elif self.init == '++':
            point = random.choice(self.X)
            dists = self.distance_func(self.X, point)
            dist_sum = dists.sum()
            for _ in range(self.n_clusters):
                total = 0.0
                if len(centroids) != 0:
                    dists = self.distance_func(self.X,centroids)
                    dists = np.min(dists, axis=0)
                    dist_sum = dists.sum()
                rand = np.random.random() * dist_sum
                for i in range(self.n_samples):
                    total += dists[i]
                    if rand <= total:
                        centroids.append(self.X[i])
                        break
        else:
            raise NotImplementedError()
        
        dists = np.zeros(shape=(self.n_samples, self.n_clusters))
        loss = []
        # last_centroids=np.zeros(shape=(self.n_clusters, self.n_features))
        for iter_num in range(self.max_iters):
            for i in range(self.n_clusters):
                dists[:, i] = self.distance_func(self.X, centroids[i])
            self.assign = np.argmin(dists, axis=1)
            loss.append(0.0)
            for i in range(self.n_clusters):
                clusters = self.X[self.assign == i]
                # last_centroids[i] = centroids[i]
                centroids[i] = np.mean(clusters, axis=0)
                if np.isnan(centroids[i][0]):
                    print(centroids[i])
                loss[iter_num] += np.sum(self.distance_func(clusters, centroids[i]))
            loss[iter_num] /= self.n_samples
            if len(loss) > 1 and np.abs(loss[-1] - loss[-2]) < self.torlerance:
                break
            print(loss[iter_num])
        
        # print(loss[-20:])
        self.centroids=np.array(centroids)
    
    def _predict(self, X=None):
        return self.assign