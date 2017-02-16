import numpy as np
from base import BaseEstimator
from metrics import l2_distance


class PCA(BaseEstimator):
    y_required = False

    def __init__(self, variance_ratio=0.95):
        self.variance_ratio = variance_ratio
        self.components = None

    def fit(self, X):
        X = X - np.mean(X, axis=0)
        _, v, w = np.linalg.svd(X)
        v_square = v**2
        cumv = np.cumsum(v_square) / v_square.sum()
        num = np.argwhere(cumv >= self.variance_ratio)[0][0]
        self.components = w[0:num]
        return self

    def transform(self, X):
        X = X - np.mean(X, axis=0)
        return np.dot(X, self.components.T)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LDA(BaseEstimator):

    def __init__(self, n_components=None, n_classes=2):
        # self.n_classes = n_classes
        self.components = None
        self.n_components = n_components
        self.n_classes = n_classes

    def fit(self, X, y):
        self._setup_input(X, y)
        assert(self.n_components < self.n_features)

        uniques = np.unique(self.y)
        self.n_classes = len(uniques)

        means = np.empty(shape=(self.n_classes, self.n_features),
                         dtype=np.dtype('float'))
        # Within-class scatter matrix Sw = sum(Si), Si =
        # sum(row-mean)(row-mean).T for each class
        Si = np.zeros(shape=(self.n_classes, self.n_features,
                             self.n_features), dtype=np.dtype('float'))
        # Between-class scatter matrix Sb
        Sb = np.zeros(shape=(self.n_features, self.n_features),
                      dtype=np.dtype('float'))

        xmean = self.X.mean(axis=0).reshape(1, self.n_features)

        for i, uni in enumerate(uniques):
            means[i] = self.X[self.y == uni].mean(axis=0)
            for row in self.X[self.y == uni]:
                diff = (row - means[i]).reshape(self.n_features, 1)
                Si[i] += diff.dot(diff.T)
            diff = np.array(means[i] - xmean).reshape(self.n_features, 1)
            Sb += np.sum(self.y == uni) * diff.dot(diff.T)
        Sw = np.sum(Si, axis=0)
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

        # Sb/Sw*v = lamda*v
        for i in range(len(eigvals)):
            eigv = eigvecs[:, i].reshape(self.n_features, 1)
            np.testing.assert_array_almost_equal(np.linalg.inv(Sw).dot(
                Sb).dot(eigv), eigvals[i] * eigv, decimal=6, verbose=True)
        indices = np.argsort(np.abs(eigvals))
        idx = indices[-self.n_components:]
        self.components = eigvecs.real[:, idx]
        ratio = np.abs(eigvals.real[indices]).sum() / np.abs(eigvals).sum()
        print('Explained variance ratio: %s' % ratio)
        # print(np.abs(eigvals.real[indices])/ np.abs(eigvals).sum())
        return self

    def transform(self, X):
        return np.dot(X, self.components)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class TSNE(BaseEstimator):
    y_required = False

    def __init__(self, lr=0.01, perplexity=30., n_components=2, max_iters=1000, perplexity_tries=50, tol=1e-5):
        self.lr = 0.01
        self.n_components = n_components
        self.lastGrad = None
        self.perplexity = perplexity
        self.max_iters = max_iters
        self.perplexity_tries = perplexity_tries
        self.tol = tol
        self.initial_momentum = 0.5
        self.final_momentum = 0.8
        self.min_gain = 0.01

    def fit(self, X):
        self._setup_input(X)
        y = np.random.normal(
            size=(self.n_samples, self.n_components))
        # (m,m,n)
        velocity = np.zeros_like(y)
        gains = np.zeros_like(y)

        P = self._simlilarity(X)

        for iter_num in range(self.max_iters):
            D = np.array(l2_distance(y))
            Q = self._q_distribution(D)
            Q_n = Q/np.sum(Q)

            # Early exaggeration & momentum
            pmul = 4.0 if iter_num < 100 else 1.0
            momentum = 0.5 if iter_num < 20 else 0.8

            grads = np.zeros_like(y)
            for i in range(self.n_samples):
                grads[i] = 4 * np.dot((pmul * P[i] - Q_n[i]) * Q[i], y[i] - y)
            
            gains = (gains + 0.2) * ((grads > 0) != (velocity > 0)) + (gains * 0.8) * ((grads > 0) == (velocity > 0))
            gains = gains.clip(min=self.min_gain)

            velocity = momentum*velocity - self.lr*(gains*grads)
            y += velocity
            y = y - np.mean(y, 0)

            if iter_num % 100 == 0:
                error = np.sum(P * np.log(P / Q_n))
                print("loss %f"%error)
        
        return y
        # Qij = self._simlilarity(y)
        # y_grad = 4*np.sum((Pij-Qij).dot(y_diff))
        # y = y-self.lr*y_grad

    def _simlilarity(self, X):
        m, _ = X.shape
        simlilarity = np.zeros(shape=(m, m))
        target_entropy = np.log(self.perplexity)
        distances = l2_distance(X)
        for i, x in enumerate(X):
            # (m)
            simlilarity[i] = self._binary_search(distances[i], target_entropy)
        # (m, m)
        np.fill_diagonal(simlilarity, 1.e-12)
        simlilarity = simlilarity.clip(min=1.e-100)
        simlilarity = (simlilarity + simlilarity.T) / m * 0.5
        return np.array(simlilarity)

    def _binary_search(self, dist, target_entropy):
        precision_min = 0
        precision_max = 1.e15
        precision = 1.e5

        for _ in range(self.perplexity_tries):
            denom = np.sum(np.exp(-dist[dist > 0.] / precision))
            beta = np.exp(-dist / precision) / denom

            g_beta = beta[beta > 0.]
            entropy = np.sum(-g_beta * np.log2(g_beta))

            error = entropy - target_entropy
            if error > 0.0:
                precision_max = precision
                precision = (precision + precision_min) * 0.5
            else:
                precision_min = precision
                precision = (precision + precision_max) * 0.5

            if np.abs(error) < self.tol:
                break
        return beta

    def _q_distribution(self, D):
        """Computes Student t-distribution."""
        Q = 1.0 / (1.0 + D)
        np.fill_diagonal(Q, 0.0)
        Q = Q.clip(min=1e-100)
        return Q