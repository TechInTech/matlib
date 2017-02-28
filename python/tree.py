import numpy as np
import random
from base import BaseEstimator
from scipy import stats
from scipy.special import expit
from metrics import entropy_criterion, gini_criterion


class Loss(object):

    def __init__(self, regularization=1.0):
        self.regularization = regularization

    def grad(self, target, prediction):
        raise NotImplementedError()

    def hess(self, target, prediction):
        raise NotImplementedError()

    def approximate(self, actual, prediction):
        return self.grad(actual, prediction).sum() / (self.hess(actual, prediction).sum() + self.regularization)

    def transform(self, pred):
        return pred

    def gain(self, actual, prediction):
        """Calculate gain for split search."""
        nominator = self.grad(actual, prediction).sum() ** 2
        denominator = (self.hess(actual, prediction).sum() +
                       self.regularization)
        return 0.5 * (nominator / denominator)


class LeastSquareLoss(Loss):

    def grad(self, target, prediction):
        return target - prediction

    def hess(self, target, prediction):
        return np.ones_like(target)


class LogisticLoss(Loss):

    def grad(self, target, prediction):
        return target * expit(-target * prediction)

    def hess(self, target, prediction):
        e = expit(prediction)
        return e * (1 - e)

    def transform(self, pred):
        return expit(pred)


class Tree(BaseEstimator):

    def __init__(self, regression=True, criterion='entropy'):
        self.regression = regression
        self.criterion = criterion
        self.left_tree = None
        self.right_tree = None
        self.leaf_value = None
        self.yu = None

    @property
    def is_leafnode(self):
        return bool(self.left_tree is None and self.right_tree is None)

    def fit(self, X, y, max_tree_depth=None, min_samples_split=10, min_gain=0.01, max_features=None):
        self._setup_input(X, y)

        try:
            assert(max_tree_depth is None or max_tree_depth > 0)
            assert(self.n_samples > min_samples_split)
            if max_tree_depth is not None:
                max_tree_depth -= 1
            if max_features is not None:
                assert(max_features <= self.n_features)

            split_x, split_v, gain = self._find_best_split(
                self.X, self.y, max_features)
            if self.regression:
                assert(gain != 0)
            else:
                assert(gain > min_gain)

            left_X, left_y, right_X, right_y = self._split_data(
                self.X, self.y, split_x, split_v)
            self.left_tree = Tree()
            self.left_tree.fit(left_X, left_y, max_tree_depth,
                               min_samples_split, min_gain, max_features)
            self.right_tree = Tree()
            self.right_tree.fit(right_X, right_y, max_tree_depth,
                                min_samples_split, min_gain, max_features)
            self.split_x = split_x
            self.split_v = split_v

        except AssertionError:
            self._cal_leafvalue()

    def _find_best_split(self, X, y, max_features=None):
        if max_features is None:
            max_features = X.shape[1]
        choosen_features = random.sample(
            list(range(0, X.shape[1])), max_features)
        if self.criterion == 'entropy':
            initial_entropy = entropy_criterion(y)
        else:
            initial_entropy = gini_criterion(y)

        split_feature = None
        split_value = None
        max_gain = None
        if initial_entropy == 0.0:
            return split_feature, split_value, max_gain

        for i in choosen_features:
            xi = X[:, i].copy()
            np.sort(xi)
            splits = []
            for j in range(self.n_samples - 1):
                if np.isclose(xi[j], xi[j + 1]):
                    pass
                splits.append((xi[j] + xi[j + 1]) * 0.5)
            if len(splits) == 0:
                splits.append((xi[-1] + xi[-2]) / 2)
            for split in splits:
                split_entropy = self._split_entropy(X, y, i, split)
                if (max_gain is None) or max_gain < initial_entropy - split_entropy:
                    max_gain = initial_entropy - split_entropy
                    split_feature = i
                    split_value = split
        return split_feature, split_value, max_gain

    def _split_entropy(self, X, y, ix, split):
        y_left, y_right = self._split_data(
            X, y, ix, split, return_X=False)
        if self.criterion == 'entropy':
            split_entropy = (len(y_left) * entropy_criterion(y_left) +
                             len(y_right) * entropy_criterion(y_right)) / y.shape[0]
        elif self.criterion == 'gini':
            split_entropy = (len(y_left) * gini_criterion(y_left) +
                             len(y_right) * gini_criterion(y_right)) / y.shape[0]
        else:
            raise NotImplementedError()
        return split_entropy

    def _split_data(self, data, target, split_x, split_v, return_X=True):
        left_mask = [data[:, split_x] <= split_v]
        right_mask = [data[:, split_x] > split_v]
        left_target = target[left_mask]
        right_target = target[right_mask]
        if return_X:
            left_data = data[left_mask]
            right_data = data[right_mask]
            return left_data, left_target, right_data, right_target
        else:
            return left_target, right_target

    def _cal_leafvalue(self):
        if self.regression:
            self.leaf_value = np.mean(self.y)
        else:
            self.leaf_value = stats.itemfreq(
                self.y)[:, 1] / float(self.n_samples)

    def _predict_x(self, X):
        if not self.is_leafnode:
            if X[self.split_x] <= self.split_v:
                return self.left_tree.predict(X)
            else:
                return self.right_tree.predict(X)
        else:
            return self.leaf_value

    def _predict(self, X):
        if X.ndim == 1:
            return self._predict_x(X)
        else:
            ret = []
            for x in X:
                ret.append(self._predict_x(x))
            return np.array(ret)


class GBDTree(Tree):

    def __init__(self, regression=True, loss=LeastSquareLoss()):
        self.regression = regression
        self.left_tree = None
        self.right_tree = None
        self.leaf_value = None
        self.loss = loss

    def fit(self, X, y, y_pred, max_tree_depth=None, min_samples_split=10, min_gain=0.01, max_features=None):
        self._setup_input(X, y)
        if not isinstance(y_pred, np.ndarray):
            self.y_pred = np.array(y_pred)
        else:
            self.y_pred = y_pred

        try:
            assert(max_tree_depth is None or max_tree_depth > 0)
            assert(self.n_samples >= min_samples_split)
            if max_tree_depth is not None:
                max_tree_depth -= 1
            if max_features is not None:
                assert(max_features <= self.n_features)

            split_x, split_v, gain = self._find_best_split(
                self.X, self.y, self.y_pred, max_features)
            if self.regression:
                assert(gain != 0)
            else:
                assert(gain > min_gain)

            left_mask = [self.X[:, split_x] <= split_v]
            right_mask = [self.X[:, split_x] > split_v]
            left_X, left_y, left_y_pred = self.X[left_mask], self.y[
                left_mask], self.y_pred[left_mask]
            right_X, right_y, right_y_pred = self.X[
                right_mask], self.y[right_mask], self.y_pred[right_mask]

            self.left_tree = GBDTree(
                regression=self.regression, loss=self.loss)
            self.left_tree.fit(left_X, left_y, left_y_pred, max_tree_depth,
                               min_samples_split, min_gain, max_features)
            self.right_tree = GBDTree(
                regression=self.regression, loss=self.loss)
            self.right_tree.fit(right_X, right_y, right_y_pred,
                                max_tree_depth, min_samples_split, min_gain, max_features)
            self.split_x = split_x
            self.split_v = split_v

        except AssertionError:
            self._cal_leafvalue()

    def _find_best_split(self, X, y, y_pred, max_features=None):
        if max_features is None:
            max_features = X.shape[1]
        choosen_features = random.sample(
            list(range(0, X.shape[1])), max_features)

        initial_entropy = self.loss.gain(self.y, self.y_pred)
        split_feature = None
        split_value = None
        max_gain = None

        for i in choosen_features:
            xi = X[:, i].copy()
            np.sort(xi)
            splits = []
            for j in range(self.n_samples - 1):
                if np.isclose(xi[j], xi[j + 1]):
                    pass
                splits.append((xi[j] + xi[j + 1]) * 0.5)
            if len(splits) == 0:
                splits.append((xi[-1] + xi[-2]) / 2)
            for split in splits:
                left_mask = [X[:, i] <= split]
                right_mask = [X[:, i] > split]
                y_left, y_right = y[left_mask], y[right_mask]
                y_pred_left, y_pred_right = y_pred[
                    left_mask], y_pred[right_mask]
                gain = self.loss.gain(
                    y_left, y_pred_left) + self.loss.gain(y_right, y_pred_right) - initial_entropy
                if (max_gain is None) or max_gain < gain:
                    max_gain = gain
                    split_feature = i
                    split_value = split
        return split_feature, split_value, max_gain

    def _cal_leafvalue(self):
        self.leaf_value = self.loss.approximate(self.y, self.y_pred)


class AdaBoostTree(Tree):

    def __init__(self, regression=True):
        self.regression = regression
        self.left_tree = None
        self.right_tree = None
        self.leaf_value = None
        self.yu = None

    def fit(self, X, y, weights, max_tree_depth=None, min_samples_split=10, min_gain=0.01, max_features=None):
        self._setup_input(X, y)
        if not isinstance(weights, np.ndarray):
            self.weights = np.array(weights)
        else:
            self.weights = weights

        try:
            assert(max_tree_depth is None or max_tree_depth > 0)
            assert(self.n_samples >= min_samples_split)
            if max_tree_depth is not None:
                max_tree_depth -= 1
            if max_features is not None:
                assert(max_features <= self.n_features)

            split_x, split_v, gain = self._find_best_split(
                self.X, self.y, self.weights, max_features)
            if self.regression:
                assert(gain != 0)
            else:
                assert(gain > min_gain)

            left_mask = [self.X[:, split_x] <= split_v]
            right_mask = [self.X[:, split_x] > split_v]
            left_X, left_y, left_weights = self.X[left_mask], self.y[
                left_mask], self.weights[left_mask]
            right_X, right_y, right_weights = self.X[
                right_mask], self.y[right_mask], self.weights[right_mask]

            self.left_tree = AdaBoostTree(
                regression=self.regression)
            self.left_tree.fit(left_X, left_y, left_weights, max_tree_depth,
                               min_samples_split, min_gain, max_features)
            self.right_tree = AdaBoostTree(
                regression=self.regression)
            self.right_tree.fit(right_X, right_y, right_weights,
                                max_tree_depth, min_samples_split, min_gain, max_features)
            self.split_x = split_x
            self.split_v = split_v

        except AssertionError:
            self._cal_leafvalue()

    def _find_best_split(self, X, y, weights, max_features=None):
        if max_features is None:
            max_features = X.shape[1]
        choosen_features = random.sample(
            list(range(0, X.shape[1])), max_features)

        initial_entropy = gini_criterion(self.y, self.weights)
        split_feature = None
        split_value = None
        max_gain = None

        for i in choosen_features:
            xi = X[:, i].copy()
            np.sort(xi)
            splits = []
            for j in range(self.n_samples - 1):
                if np.isclose(xi[j], xi[j + 1]):
                    pass
                splits.append((xi[j] + xi[j + 1]) * 0.5)
            if len(splits) == 0:
                splits.append((xi[-1] + xi[-2]) / 2)
            for split in splits:
                left_mask = [X[:, i] <= split]
                right_mask = [X[:, i] > split]
                y_left, y_right = y[left_mask], y[right_mask]
                weights_left, weights_right = weights[
                    left_mask], weights[right_mask]
                gain = gini_criterion(
                    y_left, weights_left) + gini_criterion(y_right, weights_right) - initial_entropy
                if (max_gain is None) or max_gain < gain:
                    max_gain = gain
                    split_feature = i
                    split_value = split
        return split_feature, split_value, max_gain

    def _split_entropy(self, X, y, ix, split):
        return None

    def _cal_leafvalue(self):
        if self.regression:
            self.leaf_value = np.mean(self.y)
        else:
            unique = np.unique(self.y)
            freq = {}
            for u in unique:
                freq[u] = np.sum(self.y == u) / float(self.n_samples)
            self.leaf_value = max(freq, key=freq.get)
