import numpy as np
from base import BaseEstimator
from tree import Tree, AdaBoostTree
from scipy.special import expit


class AdaBoost(BaseEstimator):

    def __init__(self, n_estimators=200, max_tree_depth=None, min_samples_split=4, min_gain=0.01, max_features=None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_features = max_features
        self.max_tree_depth = max_tree_depth
        pass

    def fit(self, X, y, ):
        self._setup_input(X, y)
        self.weights = np.repeat(1.0 / self.n_samples, self.n_samples)
        self.alphas = np.zeros(self.n_estimators)
        if list(np.unique(self.y)) != [-1,1]:
            if list(np.unique(self.y)) == [0,1]:
                self.y[self.y == 0] = -1
            else:
                raise ValueError("y wrong")
        trees = []
        for iter_num in range(self.n_estimators):
            tree = AdaBoostTree(regression=False)
            tree.fit(self.X, self.y, self.weights, self.max_tree_depth,
                     self.min_samples_split, self.min_gain, self.max_features)
            pred = tree.predict(self.X)
            err = np.mean(pred == self.y)
            alpha = 0.5 * (np.log(1 - err) - np.log(err))
            self.alphas[iter_num] = alpha
            weights = self.weights * np.exp(-alpha * self.y * pred)
            print(np.unique(weights), err, alpha)
            self.weights /= weights.sum()
            trees.append(tree)
        self.trees = trees

    def _predict(self, X):
        pred = []
        for tree in self.trees:
            pred.append(tree.predict(X))
        pred = np.array(pred)
        return np.sign(pred.T.dot(self.alphas))
