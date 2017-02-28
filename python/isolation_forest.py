import numpy as np
from base import BaseEstimator
from tree import Tree
import random


class iTree(BaseEstimator):
    y_required = False
    euler = 0.57721566490153286060651209

    def __init__(self):
        self.ix = None
        self.vx = None
        self.left_leaf = None
        self.right_leaf = None
        self.leaf_size = 1

    @property
    def is_terminal(self):
        return bool(self.left_leaf is None or self.right_leaf is None)

    def fit(self, X, max_depth=None):
        # print(max_depth)
        self._setup_input(X)
        try:
            assert(self.n_samples > 1)
            if max_depth is not None:
                # if max_depth > np.log2(self.n_samples):
                #     max_depth = np.ceil(np.log2(self.n_samples))
                max_depth -= 1
                assert(max_depth >= 0)

            has_diff = False
            for j in range(self.n_features):
                u = np.unique(self.X[:, j])
                if len(u) > 1:
                    has_diff = True
                    break
            assert(has_diff)
            self._pick_split()

            left, right = self._split_data(self.X, self.ix, self.vx)
            self.left_leaf = iTree()
            self.left_leaf.fit(left, max_depth)
            self.right_leaf = iTree()
            self.right_leaf.fit(right, max_depth)
        except AssertionError:
            self.leaf_size = self.n_samples

    def _pick_split(self):
        while True:
            self.ix = random.randint(0, self.n_features - 1)
            values = self.X[:, self.ix]
            if len(np.unique(values)) == 1:
                continue
            xmax = values.max()
            xmin = values.min()
            self.vx = np.random.random() * (xmax-xmin) + xmin
            return
            
    def _split_data(self, data, ix, vx):
        left_mask = data[:, ix] <= vx
        right_mask = data[:, ix] > vx
        return data[left_mask], data[right_mask]

    def _predict_x(self, x):
        if self.is_terminal:
            if self.leaf_size <= 1:
                p = 2*(1+self.euler)
            else:
                p = 2*(np.log(self.leaf_size-1)+self.euler) - 2*(self.leaf_size-1)/self.leaf_size
            # print(self.leaf_size, p)
            return p
        else:
            if x[self.ix] <= self.vx:
                return 1 + self.left_leaf._predict_x(x)
            else:
                return 1 + self.right_leaf._predict_x(x)

    def _predict(self, X):
        if X.ndim == 1:
            pred = self._predict_x(X)
        else:
            pred = [self._predict_x(x) for x in X]
        return pred

class IsolationForest(BaseEstimator):
    y_required = False
    euler = 0.57721566490153286060651209

    def __init__(self, n_estimators=50, n_choosen=256):
        self.n_estimators=n_estimators
        self.n_choosen=n_choosen

    def fit(self, X):
        self._setup_input(X)
        assert(self.n_choosen <= self.n_samples)

        trees = []
        for i in range(self.n_estimators):
            tree = iTree()
            x = self.X[random.sample(range(self.n_samples), self.n_choosen)]
            tree.fit(x, max_depth=np.ceil(np.log2(self.n_choosen)))
            trees.append(tree)
        self.trees=trees

    def _predict_x(self, x):
        pred = np.array([tree.predict(x) for tree in self.trees])
        pred = np.mean(np.log(pred))+self.euler
        cn = 2*(np.log(self.n_choosen-1)+self.euler) - (2*(self.n_choosen-1)/self.n_choosen)
        return np.exp2(-pred/cn)

    def _predict(self, X):
        if X.ndim == 1:
            return self._predict_x(X)
        else:
            return np.array([self._predict_x(x) for x in X])