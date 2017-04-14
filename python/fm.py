import numpy as np
from base import BaseEstimator
from scipy.special import expit as sigmoid

class FM(BaseEstimator):

    def __init__(self, K=2, max_iters=200, lr=0.01, regression=True, reg_w0=0.01, reg_w=0.1, reg_v=0.5):
        self.K = K
        self.max_iters=max_iters
        self.lr=lr
        self.regression=regression
        self.reg_w = reg_w
        self.reg_v = reg_v
        self.reg_w0=reg_w0

    def fit(self, X, y):
        self._setup_input(X, y)
        # self.X = self._add_intercept(self.X)
        assert(self.K <= self.n_features)
        w0 = 0.0
        w = np.zeros(shape=(self.n_features))
        v = np.random.normal(size=(self.n_features, self.K), scale=0.5)
        self.losses = []
        loss = None
        for iter_num in range(self.max_iters):
            pred = w0 + np.dot(self.X, w) + np.sum(np.dot(self.X, v)**2-np.dot(self.X**2, v**2), axis=1)*0.5
            if self.regression:
                loss=0.5*np.mean((pred-self.y)**2)
                diff = pred-self.y
                w0 -= self.lr*np.mean(diff) + 2*self.reg_w0*w0
                w_grad = self.lr*np.dot(self.X.T, diff)/float(self.n_samples) + 2*self.reg_w*w
                for ix, x in enumerate(self.X):
                    for i in range(self.n_features):
                        v_grad = x[i] * x.dot(v) - v[i] * x[i] ** 2
                        v[i] -= self.lr * v_grad + 2 * self.reg_v * v[i]
            else:
                # predicted=np.clip(pred, 1e-15,1-1e-15)
                # loss=-np.mean(self.y*np.log(predicted)-(1-self.y)*np.log(1-predicted))
                psign = np.sign(pred)
                loss = -np.mean(sigmoid(psign*self.y))
                diff = (sigmoid(psign*self.y)-1)*self.y
                w0 -= self.lr*np.mean(diff) + 2*self.reg_w0*w0
                w_grad = self.lr*np.dot(self.X.T, diff)/float(self.n_samples) + 2*self.reg_w*w
                for ix, x in enumerate(self.X):
                    for i in range(self.n_features):
                        v_grad = x[i] * x.dot(v) - v[i] * x[i] ** 2
                        v[i] -= self.lr * v_grad + 2 * self.reg_v * v[i]
            # print("loss %f as iter %d" % (loss, iter_num))
            self.losses.append(loss)
        self.w = w
        self.w0 = w
        self.v = v

    def _predict(self, X=None):
        if X is None:
            X = self.X
        linear_output = np.dot(X, self.w)
        factors_output = np.sum(np.dot(X, self.v) ** 2 - np.dot(X ** 2, self.v ** 2), axis=1) / 2.
        pred = self.w0 + linear_output + factors_output
        if self.regression:
            return np.sign(pred)
        else:
            return pred