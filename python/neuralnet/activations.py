from __future__ import absolute_import
import numpy as np
import six
import warnings
from ..utils import serialize_object, deserialize_object


class Layer(object):
    pass


def softmax(x, axis=-1):

    if x.ndim >= 2:
        x_exp = np.exp(x - np.amax(x, axis=1, keepdims=True))
        return x_exp / x_exp.sum(axis=1, keepdims=True)
    else:
        raise ValueError('need 2 dims')


def linear(x):
    return x


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    """soft relu"""
    return np.log(1 + np.exp(x))


def softsign(x):
    return x / (1 + np.abs(x))


def tanh(x):
    return np.tanh(x)


def hard_sigmoid(x):
    x = (0.2 * x) + 0.5
    return np.clip(x, 0.0, 1.0)


def serialize(activation):
    return activation.__name__


def deserialize(name, custom_objects=None):
    return deserialize_object(name,
                              module_objects=globals(),
                              custom_objects=custom_objects,
                              printable_module_name='activation function')


def get(identifier):
    if identifier is None:
        return linear
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        if isinstance(identifier, Layer):
            warnings.warn((
                'Do not pass a layer instance (such as {identifier}) as the '
                'activation argument of another layer. Instead, advanced '
                'activation layers should be used just like any other '
                'layer in a model.'
            ).format(identifier=identifier.__class__.__name__))
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'activation function identifier:', identifier)
