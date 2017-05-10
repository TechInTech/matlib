from __future__ import absolute_import
import six
import numpy as np 
from ..utils import serialize_object, deserialize_object

class Regularizer(object):
    """Regularizer base class.
    """

    def __call__(self, x):
        return 0.

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class L1L2(Regularizer):

    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2=l2

    def __call__(self, x):
        regularization = 0.0
        if self.l1 > 0.:
            regularization += np.sum(self.l1*np.abs(x))
        if self.l2 > 0.:
            regularization += np.sum(self.l2*np.square(x))
        return regularization
    
    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2)}


# Aliases.


def l1(l=0.01):
    return L1L2(l1=l)


def l2(l=0.01):
    return L1L2(l2=l)


def l1_l2(l1=0.01, l2=0.01):
    return L1L2(l1=l1, l2=l2)


def serialize(regularizer):
    return serialize_object(regularizer)


def deserialize(config, custom_objects=None):
    return deserialize_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='regularizer')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret regularizer identifier:',
                         identifier)