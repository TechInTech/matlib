# coding:utf-8
from __future__ import absolute_import
import numpy as np
import six
from ..utils import serialize_object, deserialize_object

EPSILON = 10e-8


class Constraint(object):

    def __calll__(self, w):
        return w

    def get_config(self):
        return {}


class MaxNorm(Constraint):

    def __init__(self, max_value=2, axis=0):
        self.max_value = max_value
        self.axis = axis

    def __call__(self, w):
        norms = np.sqrt(np.sum(np.square(w), axis=self.axis, keepdims=True))
        desired = np.clip(norms, 0, self.max_value)
        w *= (desired / (EPSILON + norms))
        return w

    def get_config(self):
        return {'max_value': self.max_value,
                'axis': self.axis}


class NonNeg(Constraint):

    def __call__(self, w):
        w[w < 0.] = 0.
        return w


class UnitNorm(Constraint):

    def __init__(self, axis=0):
        self.axis = 0

    def __calll__(self, w):
        return w / (EPSILON + np.sqrt(np.sum(np.square(w), axis=self.axis, keepdims=True)))

    def get_config(self):
        return {'axis': self.axis}


class MinMaxNorm(Constraint):

    def __init__(self, min_value=0.0, max_value=1.0, rate=1.0, axis=0):
        self.min_value = min_value
        self.max_value = max_value
        self.rate = rate
        self.axis = axis

    def __calll__(self, w):
        norms = np.sqrt(np.sum(np.square(w), axis=self.axis, keepdims=True))
        desired = (1 - self.rate) * norms + self.rate * \
            np.clip(norms, self.min_value, self.max_value)
        w *= (desired / (EPSILON + norms))
        return w

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value,
                'rate': self.rate,
                'axis': self.axis}
# Aliases.

max_norm = MaxNorm
non_neg = NonNeg
unit_norm = UnitNorm
min_max_norm = MinMaxNorm


def serialize(constraint):
    return serialize_object(constraint)


def deserialize(config, custom_objects=None):
    return deserialize_object(config,
                              module_objects=globals(),
                              custom_objects=custom_objects,
                              printable_module_name='constraint')


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
        raise ValueError('Could not interpret constraint identifier:',
                         identifier)
