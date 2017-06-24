from __future__ import absolute_import
import six
import numpy as np
from ..utils import deserialize_object, serialize_object
from autograd import elementwise_grad

def clip_norm(g, c, n):
    if c > 0:
        if n >= c:
            g = g * c / n
    return g

class Optimizer(object):

    def __init__(self, **kwargs):
        allowed_kwargs = {'clipnorm', 'clipvalue'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('unexpected argument ' + k)
        self.clipnorm = 0.0
        self.clipvalue = 0.0
        self.__dict__.update(kwargs)
        self.updates = []
        self.weights = []
    
    def get_updates(self, params, constraints, loss):
        raise NotImplementedError
    
    def get_gradients(self, loss, params):
        loss_grad = elementwise_grad(loss)
        grads = loss_grad(params['y_true'], params['y_pred'])
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = np.sqrt(np.square(grads).sum())
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [np.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads
    

    def get_config(self):
        config = {}
        if hasattr(self, 'clipnorm'):
            config['clipnorm'] = self.clipnorm
        if hasattr(self, 'clipvalue'):
            config['clipvalue'] = self.clipvalue
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class SGD(Optimizer):

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, iterations=100, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.iterations = iterations
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.initial_decay = decay
        self.nesterov = nesterov
    
    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0.:
            lr = lr * (1. / (1. + self.decay*self.iterations))
        
        last_grad = params['last_grad']
        v = self.momentum * last_grad - lr * g  # velocity

        if self.nesterov:
            new_p = p + self.momentum * v - lr * g
        else:
            new_p = p + v

        # apply constraints
        if p in constraints:
            c = constraints[p]
            new_p = c(new_p)

        return self.updates

    def get_config(self):
        config = {'lr': self.lr,
                  'momentum': self.momentum,
                  'decay': self.decay,
                  'nesterov': self.nesterov}
        base_config = super(SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# aliases

sgd = SGD

def serialize(optimizer):
    return serialize_object(optimizer)

def deserialize(config, custom_objects=None):
    all_classes = {
        'sgd': SGD,
        # 'rmsprop': RMSprop,
        # 'adagrad': Adagrad,
        # 'adadelta': Adadelta,
        # 'adam': Adam,
        # 'adamax': Adamax,
        # 'nadam': Nadam,
        # 'tfoptimizer': TFOptimizer,
    }
    # Make deserialization case-insensitive for built-in optimizers.
    if config['class_name'].lower() in all_classes:
        config['class_name'] = config['class_name'].lower()
    return deserialize_object(config,
                                    module_objects=all_classes,
                                    custom_objects=custom_objects,
                                    printable_module_name='optimizer')


def get(identifier):
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    if isinstance(identifier, Optimizer):
        return identifier
    else:
        raise ValueError('Could not interpret optimizer identifier:',
                         identifier)