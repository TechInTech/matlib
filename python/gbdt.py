import numpy as np
from base import BaseEstimator
from tree import Tree, LeastSquareLoss, GBDTree


class GBDT(BaseEstimator):

    def __init__(self,lr=0.01, n_estimators=100, 
                 max_tree_depth=5, min_samples_split=10, min_gain=0.01, loss=LeastSquareLoss(), max_features=None):
        self.n_estimators = n_estimators
        self.max_tree_depth = max_tree_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.loss = loss
        self.lr = lr
        self.trees = None
        self.max_features=max_features
        pass

    def fit(self, X, y):
        self._setup_input(X, y)
        trees = []
        y_pred = np.zeros(self.n_samples, dtype=np.dtype('float32'))
        for _ in range(self.n_estimators):
            tree = GBDTree(regression=True, loss=self.loss)
            tree.fit(self.X, y, y_pred, max_tree_depth=self.max_tree_depth,
                        min_samples_split=self.min_samples_split, min_gain=self.min_gain, max_features=self.max_features)
            prediction = tree.predict(self.X)
            y_pred += self.lr * prediction
            trees.append(tree)
        self.trees = trees

    def _predict(self, X):
        preds = []
        for tree in self.trees:
            preds.append(self.lr*tree.predict(X))
        if X.ndim == 1:
            return np.sum(preds)
        else:
            return np.sum(preds, axis=0)
