import numpy as np
from base import BaseEstimator

def softmax(x, axis=-1):

    if x.ndim >= 2:
        x_exp = np.exp(x - np.amax(x, axis=1, keepdims=True))
        return x_exp / x_exp.sum(axis=1, keepdims=True)
    else:
        raise ValueError('need 2 dims')

class RNN(BaseEstimator):

    def __init__(self, units, bptt_truncate, hidden_units=100, n_unfold=8, lr=0.005):
        self.bptt_truncate = bptt_truncate
        self.units = units
        self.hidden_units = hidden_units
        self.n_unfold = n_unfold
        self.lr = lr

    def fit(self, X, y):
        self._setup_input(X, y)
        assert self.n_features == self.units
        self.U = np.random.uniform(-np.sqrt(self.hidden_units), np.sqrt(
            self.hidden_units), size=(self.hidden_units, self.units))
        self.W = np.random.uniform(-np.sqrt(self.hidden_units), np.sqrt(self.hidden_units), size=(
            self.hidden_units, self.hidden_units))
        self.V = np.random.uniform(-np.sqrt(self.hidden_units), np.sqrt(self.hidden_units), size=(self.units, self.hidden_units))

        for xi, yi in enumerate(zip(self.X, self.y)):
            dLdU, dLdW, dLdV = self.back_propagation(xi, yi)
            self.U -= self.lr*dLdU
            self.W -= self.lr*dLdW
            self.V -= self.lr*dLdV
            
    def forward_propagation(self, x):
        T = len(x)
        s = np.zeros((T+1, self.hidden_units))
        s[-1] = 0.0
        o = np.zeros((T, self.units))

        for t in range(T):
            s[t] = np.tanh(self.U.dot(x[t]) + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return o, s
    
    def _predict(self, x):
        o, _ = self.forward_propagation(x)
        return np.argmax(o, axis=1)
    
    def back_propagation(self, x, y):
        T = len(y)
        o, s = self.forward_propagation(x)
        dLdU = np.zeros(self.U.shape)
        dLdW = np.zeros(self.W.shape)
        dLdV = np.zeros(self.V.shape)
        delta_o = o
        delta_o[:, y] -= 1.0
        for t in np.arange(T-1):
            dLdV += np.outer(delta_o[t], s[t].T)
            delta_t = self.V.T.dot(delta_o[t]) * (1-s[t]**2)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:, x[bptt_step]] += delta_t
                delta_t=self.W.T.dot(delta_t) * (1-s[bptt_step-1]**2)
        return [dLdU, dLdW, dLdV]


