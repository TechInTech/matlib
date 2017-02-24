import numpy as np
from collections import Counter
from base import BaseEstimator
from metrics import euclidean_distance

class KNN(BaseEstimator):
    y_required=False
    def __init__(self, K=10, distance_func=euclidean_distance):
        self.K = 10
        self.distance_func = distance_func
    
    def aggregate(self, nn, indices=None):
        raise NotImplementedError()

    def _predict(self, X=None):
        if X == None:
            if self.n_samples < self.K+1:
                raise ValueError("sample number < k")
            predictions = [self._predict_real(x) for x in self.X]
        else:
            predictions = [self._predict_real(x) for x in X]
        return np.array(predictions)
    
    def _predict_real(self, x):
        dists = np.array([self.distance_func(x, xi) for xi in self.X])
        indices=np.argsort(dists)[:self.K]
        nn = self.X[indices]
        return self.aggregate(nn, indices)

class KNNClassifier(KNN):

    def aggregate(self, nn, indices=None):
        ys = self.y[indices]
        unique = np.unique(ys)
        counts = np.array([[u, np.sum(ys == u)] for u in unique], dtype=[('val', 'count')])
        np.sort(counts, axis=0, order='val')
        return counts[0][0]

class KNNRegressor(KNN):

    def aggregate(self, nn, indices=None):
        return np.mean(nn, axis=0)