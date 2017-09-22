from base import BaseEstimator
import scipy.spatial.distance as dist
import logging

class Linear(object):
    def __call__(self, x, y):
        return np.dot(x, y.T)

    def __repr__(self):
        return 'Linear kernel'


class Poly(object):
    def __init__(self, degree=2):
        self.degree = degree

    def __call__(self, x, y):
        return np.dot(x, y.T) ** self.degree

    def __repr__(self):
        return 'Poly kernel'


class RBF(object):
    def __init__(self, gamma=0.1):
        self.gamma = gamma

    def __call__(self, x, y):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        return np.exp(-self.gamma * dist.cdist(x, y) ** 2).flatten()

    def __repr__(self):
        return 'RBF kernel'

class SVM(BaseEstimator):

    def __init__(self, C=1.0, kernal=None, tol=1e-3, max_iter=100):
        self.C = C
        if kernal is not None:
            self.kernal = kernal
        else:
            self.kernal = Linear()
        self.tol = tol
        self.max_iter=max_iter
        self.K = None
        self.b = 0
    
    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.K = np.zeros(self.n_samples, self.n_samples)
        for i in range(self.n_samples):
            self.K[:, i] = self.kernal(self.X, self.X[i,:])
        self.alpha = np.zeros(self.n_samples)
        self.sv_idx = np.arange(0, self.n_samples)
        return self._train()
    
    def _train(self):
        iters = 0
        while iters < self.max_iter:
            iters += 1
            alpha_prev = np.copy(self.alpha)

            for j in range(self.n_samples):
                i = self.random_pick(j)
                eta = 2.0*self.K[i,j]- self.K[i, i]-self.K[j,j]
                if eta >= 0.0:
                    continue
                L, H = self._find_bounds(i, j)
                e_i, e_j = self._error(i), self._error(j)
                alpha_io, alpha_jo = self.alpha[i], self.alpha[j]

                self.alpha[j] -= self.y(e_i-e_j) / eta
                self.alpha[j] = self.clip(self.alpha[j], H, L)

                self.alpha[i] += self.y[i]*self.y[j]*(alpha_jo - self.alpha[j])
                b1 = self.b - e_i - self.y[i] * (self.alpha[i] - alpha_io) * self.K[i, i] - \
                     self.y[j] * (self.alpha[j] - alpha_jo) * self.K[i, j]
                b2 = self.b - e_j - self.y[i] * (self.alpha[i] - alpha_io) * self.K[i, j] - \
                    self.y[j] * (self.alpha[j] - alpha_jo) * self.K[j, j]

                if 0 < self.alpha[i] < self.C:
                    self.b = b1
                elif 0 < self.alpha[j] < self.C:
                    self.b = b2
                else:
                    self.b = 0.5 * (b1 + b2)
                
                            # Check convergence
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                break
        logging.info('Convergence has reached after %s.' % iters)

        # Save support vectors index
        self.sv_idx = np.where(self.alpha > 0)[0]

    def clip(self, alpha, H, L):
        if alpha > H:
            alpha = H
        if alpha < L:
            alpha = L
        return alpha

    def _predict(self, X=None):
        n = X.shape[0]
        result = np.zeros(n)
        for i in range(n):
            result[i] = np.sign(self._predict_row(X[i, :]))
        return result

    def _predict_row(self, X):
        k_v = self.kernal(self.X[self.sv_idx], self.X)
        return np.dot((self.alpha[self.sv_idx] * self.y[self.sv_idx]).T, k_v.T)+self.b

    def _error(self, i):
        return self._predict_row(self.X[i]) - self.y[i]

    def _find_bounds(self, i, j):
        if self.y[i] != self.y[j]:
            L = max(0.0, self.alpha[j]-self.alpha[i])
            H = min(self.C, self.C + self.alpha[j]-self.alpha[i])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        return L, H

    def random_pick(self, j):
        i = j
        while i == j:
            i = np.random_int(0, self.n_samples-1)
        return i