import numpy as np
from base import BaseEstimator
import matplotlib.pyplot as plt
import os

class MLP(BaseEstimator):

    def __init__(self, lr=0.001, max_iters=1000, torlerance=1e-4):
        self.lr = lr
        self.max_iters=max_iters
        self.torlerance=torlerance

    def fit(self, X, y):
        self._setup_input(X, y)
        if list(np.unique(self.y)) == [0, 1]:
            self.y[self.y == 0] = -1
        if list(np.unique(self.y)) != [-1,1]:
            raise ValueError('y is either -1 or 1')
        self.y = self.y.reshape(-1,1)

        w = np.random.normal(size=(self.n_features,1), scale=0.5)
        b = np.random.random()
        loss = None
        for i in range(self.max_iters):
            prediction = np.dot(self.X,w) + b
            prediction[prediction > 0] = 1
            prediction[prediction <= 0] = -1
            indices = np.array(np.where(self.y != prediction)[0])
            if len(indices) == 0:
                break
            loss_new = -np.sum(self.y[indices]*(np.dot(self.X[indices], w)+b))/self.n_samples
            if i > 0 and i%100 == 0:
                print("loss {%f}"%loss_new)
            if loss != None and np.abs(loss-loss_new) <= self.torlerance:
                break
            loss=loss_new

            ix = np.random.choice(indices)
            w_grad = -np.reshape(self.lr*self.y[ix]*self.X[ix], newshape=(self.n_features, 1))
            w = w-w_grad
            b = b+self.lr*self.y[ix]
        
        self.w=w
        self.b=b

    def predict(self, X):
        prediction = np.dot(X, self.w)+self.b
        prediction[prediction > 0] = 1
        prediction[prediction <= 0] = 0
        return prediction