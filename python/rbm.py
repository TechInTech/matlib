from base import BaseEstimator
from scipy.special import expit as sigmoid
import numpy as np
import logging
from metrics import batch_iterator

class RBM(BaseEstimator):
    y_required=False

    def __init__(self, n_hidden=128, lr=0.05, batch_size=10, max_epochs=100):
        self.n_hidden=n_hidden
        self.lr=lr
        self.batch_size=batch_size
        self.max_epochs=max_epochs
        self.n_visible = 0

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.n_visible = self.X.shape[1]
        self._init_weights()
        self._train()

    def _init_weights(self):
        self.W = np.random.normal(scale=1e-1, size=(self.n_visible, self.n_hidden))
        self.bias_v = np.random.normal(scale=1e-4, size=(self.n_visible))
        self.bias_h = np.random.normal(scale=1e-4, size=(self.n_hidden))

        self.errors = []

    def _train(self):
        for i in range(self.max_epochs):
            error = 0
            for batch in batch_iterator(self.X, batch_size=self.batch_size):
                positive_hidden = sigmoid(np.dot(batch, self.W) + self.bias_h)
                hidden_states = positive_hidden > np.random.random_sample(size=positive_hidden.shape)
                positive_associations = np.dot(batch.T, positive_hidden)

                negative_visible = sigmoid(np.dot(hidden_states, self.W.T)+self.bias_v)
                negative_hidden = sigmoid(np.dot(negative_visible, self.W)+self.bias_h)
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
    
    def _predict(self, X=None):
        if X is None:
            return sigmoid(np.dot(self.X, self.W) + self.bias_h)
        else:
            return sigmoid(np.dot(X, self.W) + self.bias_h)
        