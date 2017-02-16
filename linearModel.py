#!--coding: utf-8
import numpy as np
from base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder

class LinearRegression(BaseEstimator):

    def __init__(self, lr=0.001, alpha=0.1, penalty=None, tolerance=1e-4, max_iters=1000, descent_grad=None):
        if penalty is not None and penalty != 'l1' and penalty != 'l2':
            raise ValueError('penalty is either none or l1 or l2')
        self.penalty = penalty
        self.alpha = alpha
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.lr = lr
        self.descent_grad = descent_grad
        self.lastGrad = 0

    @staticmethod
    def _add_intercept(X):
        intercept = np.ones([X.shape[0], 1])
        return np.concatenate([intercept, X], axis=1)

    def fit(self, X, y):
        self._setup_input(X, y)
        X = self._add_intercept(self.X)
        y = self.y.reshape(-1, 1)
        w = np.random.normal(size=(self.n_features + 1, 1), scale=0.5)

        loss = self._loss(X, y, w)
        for i in range(self.max_iters):
            w_grad = self._loss_grad(X, y, w)
            w -= w_grad
            loss_new = self._loss(X, y, w)
            if np.abs(loss - loss_new) <= self.tolerance:
                break
            loss = loss_new
            if i % 100 == 0:
                print(loss)
        self.w = w

    def _loss(self, X, y, w):
        loss = np.sum((np.dot(X, w) - y)**2) / (2 * self.n_samples)
        if self.penalty == "l2":
            loss += self.alpha * np.sum(w**2)
        elif self.penalty == "l1":
            loss += self.alpha * np.sum(np.abs(w))
        return loss

    def _loss_grad_real(self, X, y, w):
        w_grad = np.dot(X.T, np.dot(X, w) - y) / self.n_samples
        if self.penalty == "l2":
            w_grad += 2 * self.alpha * w
        elif self.penalty == "l1":
            w_grad += 2 * self.alpha * np.sign(w)
        return w_grad

    def _loss_grad(self, X, y, w):
        w_grad = self._loss_grad_real(X, y, w)
        if self.descent_grad == 'momentum':
            w_grad = 0.9 * self.lastGrad + self.lr * \
                self._loss_grad_real(X, y, w)
        elif self.descent_grad == 'nag':
            if not isinstance(self.lastGrad, np.ndarray):
                w_grad = 0.9 * self.lastGrad + self.lr * \
                    self._loss_grad_real(X, y, w)
            else:
                grad_add = self.lr * \
                    self._loss_grad_real(X - (0.9 * self.lastGrad).T, y, w)
                w_grad = 0.9 * self.lastGrad + grad_add
        elif self.descent_grad == 'Adadelta':
            if not isinstance(self.lastGrad, np.ndarray):
                w_grad = self.lr * self._loss_grad_real(X, y, w)
            else:
                eg = 0.9 * (self.lastGrad**2) + 0.1 * \
                    (self._loss_grad_real(X, y, w)**2)
                self.eg = eg
                w_grad = self.lr / \
                    np.sqrt(eg + 1e-8) * self._loss_grad_real(X, y, w)
        else:
            w_grad = self.lr * self._loss_grad_real(X, y, w)
        self.lastGrad = w_grad

        if np.isnan(self.lastGrad[0]):
            print('wrong')

        return w_grad

    def predict(self, X):
        return self._predict(X)

    def _predict(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X = self._add_intercept(X)
        return np.dot(X, self.w)


class LogisticRegression(BaseEstimator):

    def __init__(self, lr=0.001, alpha=0.1, gamma=1, penalty=None, tolerance=1e-4, max_iters=1000, epsilon=1e-5):
        if penalty is not None and penalty != 'l1' and penalty != 'l2':
            raise ValueError('penalty is either none or l1 or l2')
        self.penalty = penalty
        self.alpha = alpha
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    @staticmethod
    def _add_intercept(X):
        intercept = np.ones([X.shape[0], 1])
        return np.concatenate([intercept, X], axis=1)

    def fit(self, X, y):
        self._setup_input(X, y)
        X = self._add_intercept(self.X)
        self.y[self.y == -1] = 0
        if list(np.unique(self.y)) != [0, 1]:
            raise ValueError('y is not suitable for classification')
        y = self.y.reshape(-1, 1)
        w = np.random.normal(size=(self.n_features + 1, 1), scale=0.5)

        loss = self._loss(X, y, w)
        for i in range(self.max_iters):
            w_grad = self._loss_grad(X, y, w)
            w -= self.lr * w_grad
            loss_new = self._loss(X, y, w)
            if np.abs(loss - loss_new) <= self.tolerance:
                break
            loss = loss_new
            if i % 100 == 0:
                print(loss)
        self.w = w

    def _loss(self, X, y, w):
        predictions = np.clip(self.sigmoid(np.dot(X, w)),
                              self.epsilon, 1 - self.epsilon)
        loss = -np.mean(np.log(predictions) * y + (1 - y)
                        * np.log(1 - predictions))
        if self.penalty == "l2":
            loss += 0.5 * self.alpha * np.sum(w**2)
        elif self.penalty == "l1":
            loss += self.alpha * np.sum(np.abs(w))
        return loss

    def _loss_grad(self, X, y, w):
        predictions = np.clip(self.sigmoid(np.dot(X, w)),
                              self.epsilon, 1 - self.epsilon)
        w_grad = np.dot(X.T, predictions - y) / self.n_samples
        if self.penalty == "l2":
            w_grad += self.alpha * w
        elif self.penalty == "l1":
            w_grad += self.alpha * np.sign(w)
        return w_grad

    def predict(self, X):
        return self._predict(X)

    def _predict(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X = self._add_intercept(X)
        y = self.sigmoid(np.dot(X, self.w))
        return y

    def sigmoid(self, x):
        return 1.0 / (np.exp(-self.gamma * x) + 1.0)

class SoftMaxRegression(BaseEstimator):

    def __init__(self, lr=0.001, n_classes=2, tolerance=1e-4, max_iters=1000, alpha=0.01, penalty=None):
        self.lr=0.001
        self.n_classes=n_classes
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.alpha=0.01
        self.penalty=penalty

    def fit(self, X, y):
        self._setup_input(X, y)
        self.X = self._add_intercept(self.X)
        if y.size != self.n_samples * self.n_classes:
            if y.size == self.n_samples:
                self.y = OneHotEncoder(n_values=self.n_classes, sparse=False, dtype=np.int).fit_transform(self.y)
            else:
                raise ValueError('unexpected y')
        self.y = self.y.reshape(-1, self.n_classes)
        w = np.random.normal(size=(self.n_features + 1, self.n_classes), scale=0.5)
        
        loss = self._loss(self.X, self.y, w)
        
        for i in range(self.max_iters):
            w -= self.lr*self._loss_grad(self.X, self.y, w)
            loss_new=self._loss(self.X, self.y, w)
            if np.abs(loss_new-loss) < self.tolerance:
                break
            if i%100 == 0:
                print("loss %f"%(loss))
            loss=loss_new
        self.w=w
    
    def _loss(self, X, y, w):
        prediction = np.dot(X, w)
        predictions = np.exp(prediction)
        tmp = np.sum(predictions, axis=1)
        hypothesis = predictions / np.sum(predictions, axis=1).reshape(-1,1)
        # h (sample , classes) y (sample , classes)
        loss = -np.sum(np.sum(np.log(hypothesis)*self.y))/self.n_samples
        if self.penalty == 'l2':
            loss += 0.5*self.alpha*np.sum(np.linalg.norm(w))
        elif self.penalty == 'l1':
            loss += self.alpha*np.sum(np.abs(w))
        return loss
    
    def _loss_grad(self, X, y, w):
        prediction = np.dot(X, w)
        predictions = np.exp(prediction)
        # (sample , classes)
        hypothesis = predictions / np.sum(predictions, axis=1).reshape(-1,1)
        # (sample, feature).T .* (sample , classes) = (feature, classes)
        w_grad = -np.dot(self.X.T, self.y-hypothesis)/self.n_samples
        if self.penalty == 'l2':
            w_grad += self.alpha*w
        elif self.penalty == 'l1':
            w_grad += self.alpha*np.sign(w)
        return w_grad
    
    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X=np.array(X)
        X=self._add_intercept(X)
        prediction = np.dot(X, self.w)
        predictions = np.exp(prediction)
        return predictions / np.sum(predictions, axis=1).reshape(-1,1)