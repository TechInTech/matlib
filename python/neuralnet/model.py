# coding: utf-8
import numpy as np
from .layer import Layer
from . import constraints, activations, optimizers, regularizers, losses, initializers
from autograd import elementwise_grad
from ..base import BaseEstimator

class PhaseMixin(object):
    _train = False

    @property
    def is_training(self):
        return self._train

    @is_training.setter
    def is_training(self, is_train=True):
        self._train = is_train

    @property
    def is_testing(self):
        return not self._train

    @is_testing.setter
    def is_testing(self, is_test=True):
        self._train = not is_test

class Model(PhaseMixin):

    y_required = False

    def __init__(self, layers, optimizer, loss, max_epochs=100, batch_size=64):
        self.loss = losses.get(loss)
        self.optimizer = optimizers.get(optimizer)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.layers = layers
        self.loss_grad = elementwise_grad(self.loss)

        for layer in self.layers:
            assert isinstance(layer, Layer)
        
        self._initialized = False
    
    def _setup_layers(self, x_shape):
        x_shape = list(x_shape)
        x_shape[0] = self.batch_size
        
        for layer in self.layers:
            layer.build(x_shape)
            x_shape = layer.compute_output_shape(x_shape)
        
        self.n_layers = len(self.layers)
        assert self.n_layers > 0
        self._initialized = True
        self.optimizer

    def fit(self, X, y):
        if not self._initialized:
            self._setup_layers(X.shape)

        if not isinstance(X, np.ndarray):
            X = np.array(X)
        self.X = X
        if not isinstance(y, np.ndarray):
            y = np.array(y)
            if y.ndim == 1:
                y = y[:, np.newaxis]
        self.y = y

        