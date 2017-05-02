import numpy as np
from base import BaseEstimator

class HMM(BaseEstimator):
    y_required = False

    def __init__(self, visibles, hiddens, start_probs = None, trans_probs = None, emit_probs=None):
        self.trans_probs = trans_probs
        self.emit_probs = emit_probs
        self.start_probs = start_probs

        # [0, 1, 2, ..., M]
        self.visibles = visibles
        assert len(self.visibles) == len(np.unique(self.visibles))
        self.n_visible = len(np.unique(self.visibles))

        # [0, 1, 2, ..., N]
        self.hiddens = hiddens
        assert len(self.visibles) == len(np.unique(self.visibles))
        self.n_hidden = len(np.unique(self.hiddens))
        
        assert self.trans_probs.shape == (self.n_hidden, self.n_hidden)
        assert self.emit_probs.shape == (self.n_hidden, self.n_visible)
        assert self.start_probs.shape[0] == self.n_hidden

    def evaluate(self, X):
        self._setup_input(X)
        assert len(np.unique(self.X)) <= self.n_visible

        # beta = self._backward()
        # return (beta[0] * self.emit_probs[:, self.X[0]] * self.start_probs).sum()

        alpha = self._forward()
        return alpha[-1].sum()

    def _forward(self):
        alpha = np.zeros(shape=(self.n_features, self.n_hidden))
        for idx in xrange(self.n_features):
            if idx == 0:
                alpha[idx] = self.start_probs * self.emit_probs[:, self.X[idx]]
            else:
                alpha[idx] = np.dot(alpha[idx-1], self.trans_probs) * self.emit_probs[:, self.X[idx]]
        return alpha
    
    def _backward(self):
        beta = np.zeros(shape=(self.n_features, self.n_hidden))
        for idx in xrange(self.n_features-1, -1, -1):
            if idx == self.n_features-1:
                beta[idx] = 1.0
            else:
                beta[idx] = np.dot(self.trans_probs, self.emit_probs[:, self.X[idx+1]] * beta[idx+1])
        return beta

    def veterbi(self, X):
        self._setup_input(X)
        assert len(np.unique(self.X)) <= self.n_visible

        alpha = np.zeros(shape=(self.n_features, self.n_hidden))
        visibles = np.zeros(self.n_features)
        for idx in xrange(self.n_features):
            if idx == 0:
                alpha[idx] = self.start_probs * self.emit_probs[:, self.X[idx]]
            else:
                alpha[idx] = np.dot(alpha[idx-1], self.trans_probs) * self.emit_probs[:, self.X[idx]]
            visibles[idx] = np.argmax(alpha[idx])
        return visibles
    
    def fit(self, X, smoothing=0.0):
        self._setup_input(X)
        assert len(np.unique(self.X)) <= self.n_visible

        alpha = self._forward()
        beta = self._backward()

        # gamma = np.zeros(shape=(self.n_features, self.n_hidden))
        gamma = alpha*beta
        for idx in xrange(self.n_features):
            if gamma[idx].sum() != 0.0:
                gamma[idx] /= gamma[idx].sum()
        
        xi = []
        for idx in xrange(self.n_features-1):
            # self.emit_probs[:, self.X[idx+1]] (N)
            # np.outer(alpha[idx], beta[idx+1]) * self.trans_probs (N, N)
            x = np.outer(alpha[idx], beta[idx+1]) * self.trans_probs * self.emit_probs[:, self.X[idx+1]]
            x /= x.sum()
            xi.append(x)
        
        for hidden in self.hiddens:
            self.start_probs[hidden] = (smoothing+gamma[0][hidden]) / (1+self.n_hidden*smoothing)
            gamma_sum = gamma[0:self.n_features-1, hidden].sum()

            if gamma_sum > 0:
                denominator = gamma_sum + self.n_hidden*smoothing
                self.trans_probs[hidden] = (smoothing  + xi[:][hidden][:].sum(axis=0))/denominator
                self.trans_probs[hidden] /= self.trans_probs[hidden].sum()
            else:
                self.trans_probs[hidden] = 0.0
            
            gamma_sum = gamma[:, hidden].sum()
            emit_gamma_sum = np.zeros(shape=(self.n_visible))
            for idx in xrange(self.n_features):
                emit_gamma_sum[self.X[idx]] += gamma[idx][hidden]
            if gamma_sum > 0:
                denominator = gamma_sum + self.n_visible * smoothing
                self.emit_probs[hidden] = (smoothing + emit_gamma_sum) / denominator
            else:
                self.emit_probs[hidden] = 0.0
        