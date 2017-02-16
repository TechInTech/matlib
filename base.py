import numpy as np

class BaseEstimator(object):
    X = None
    y = None
    y_required = True
    fit_required = True
    
    def _setup_input(self, X, y=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.size == 0:
            raise ValueError('X can not be empty')
        
        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])
        
        self.X = X
        if self.y_required:
            if y is None:
                raise ValueError('Missed required argument y')

            if not isinstance(y, np.ndarray):
                y = np.array(y)
                
            if y.size == 0:
                raise ValueError('Number of targets must be > 0')

        self.y = y
    
    @staticmethod
    def _add_intercept(X):
        intercept = np.ones([X.shape[0], 1])
        return np.concatenate([intercept, X], axis=1)

    def fit(self, X, y=None):
        self._setup_input(X, y)

    def predict(self, X=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.X is not None or not self.fit_required:
            return self._predict(X)
        else:
            raise ValueError('You must call `fit` before `predict`')

    def _predict(self, X=None):
        raise NotImplementedError()