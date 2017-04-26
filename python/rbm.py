from base import BaseEstimator
from scipy.special import expit as sigmoid
import numpy as np
import logging
from metrics import batch_iterator
from activation import SigmoidActivationFunction, ReLUActivationFunction
from scipy.stats import truncnorm

class RBM(BaseEstimator):
    y_required=False

    def __init__(self, n_hidden=128, lr=0.05, batch_size=10, max_epochs=100, active_func='sigmoid'):
        self.n_hidden=n_hidden
        self.lr=lr
        self.batch_size=batch_size
        self.max_epochs=max_epochs
        self.n_visible = None
        self.W = None
        self.bias_v = None
        self.bias_h = None
        self.errors = []
        self.active_func = active_func
        self.active_func_class = None

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.n_visible = self.X.shape[1]
        self._init_weights()
        self._train()

    def _init_weights(self):
        if self.active_func == 'sigmoid':
            self.W = np.random.randn(self.n_visible, self.n_hidden)/ np.sqrt(self.n_visible)
            self.bias_v = np.random.randn(self.n_visible)/ np.sqrt(self.n_visible)
            self.bias_h = np.random.randn(self.n_hidden)/np.sqrt(self.n_visible)
            self.active_func_class = SigmoidActivationFunction
        elif self.active_func == 'relu':
            self.W = truncnorm.rvs(-0.2, 0.2, size=[self.n_visible, self.n_hidden]) / np.sqrt(
                self.n_visible)
            self.bias_v = np.full(self.n_visible, 0.1) / np.sqrt(self.n_visible)
            self.bias_h = np.full(self.n_hidden, 0.1) / np.sqrt(self.n_visible)
            self.active_func_class = ReLUActivationFunction
        else:
            raise ValueError('unknown active func')
        self.errors = []

    def _train(self):
        for i in range(self.max_epochs):
            error = 0
            for batch in batch_iterator(self.X, batch_size=self.batch_size):
                positive_hidden = self.active_func_class.function(np.dot(batch, self.W) + self.bias_h)
                hidden_states = self._sample(positive_hidden)
                positive_associations = np.dot(batch.T, positive_hidden)

                negative_visible = self.active_func_class.function(np.dot(hidden_states, self.W.T)+self.bias_v)
                # negative_visible = self._sample(negative_visible)
                negative_hidden = self.active_func_class.function(np.dot(negative_visible, self.W)+self.bias_h)
                negative_associations = np.dot(negative_visible.T, negative_hidden)

                lr = self.lr / float(batch.shape[0])
                self.W += lr*((positive_associations-negative_associations)/float(self.batch_size))
                self.bias_h += lr*(negative_hidden.sum(axis=0) - negative_associations.sum(axis=0))
                self.bias_v += lr*(np.asarray(batch.sum(axis=0)).squeeze() - negative_visible.sum(axis=0))
                
                error += np.sum((batch-negative_visible)**2)
                
            self.errors.append(error)
            logging.info('Iteration %s, error %s' % (i, error))
        logging.debug('Weights: %s' % self.W)
        logging.debug('Hidden bias: %s' % self.bias_h)
        logging.debug('Visible bias: %s' % self.bias_v)

    def _sample(self, X):
        return X > np.random.random_sample(size=X.shape)

    def _predict(self, X):
        return self.active_func_class.function(np.dot(X, self.W) + self.bias_h)
        