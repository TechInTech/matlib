from __future__ import absolute_import
import six
import numpy as np

class Optimizer(object):

    def __init__(self, **kwargs):
        allowed_kwargs = {'clipnorm', 'clipvalue'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('unexpected argument')
        self.__dict__.update(kwargs)
        self.updates = []
        self.weights = []
    
    def get_updates(self, params, constraints, loss):
        raise NotImplementedError
    
    def get_gradients(self, loss, params):
        pass