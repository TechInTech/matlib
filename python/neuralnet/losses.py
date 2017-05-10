from __future__ import absolute_import
import numpy as np
import six
import warnings
from ..utils import serialize_object, deserialize_object

EPSILON = 1e-15


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2, axis=-1)


def mean_absolute_error(y_true, y_pred):
    return np.abs(y_true - y_pred, axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    diff = np.abs((y_true - y_pred) / np.clip(y_true, EPSILON, None))
    return 100. * np.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    first = np.log(np.clip(y_pred, EPSILON, None) + 1.)
    second = np.log(np.clip(y_true, EPSILON, None) + 1.)
    return np.mean((first - second)**2, axis=-1)


def squared_hinge(y_true, y_pred):
    return np.mean(np.maximum(1. - y_pred * y_pred, 0.)**2, axis=-1)


def hinge(y_true, y_pred):
    return np.mean(np.maximum(1. - y_pred * y_pred, 0.), axis=-1)


def logcosh(y_true, y_pred):
    return np.mean(np.log(np.cosh(y_pred - y_true)), axis=-1)


def categorical_crossentropy(y_true, y_pred):
    y_pred = y_pred / np.sum(y_pred, axis=y_pred.ndim - 1, keepdims=True)
    y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
    return -np.mean(y_true * np.log(y_pred), axis=y_pred.ndim - 1)


def binary_crossentropy(y_true, y_pred):
    y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=-1)


def kullback_leibler_divergence(y_true, y_pred):
    y_true = np.clip(y_true, EPSILON, 1)
    y_pred = np.clip(y_pred, EPSILON, 1)
    return np.sum(y_true * np.log(y_true / y_pred), axis=-1)


def poisson(y_true, y_pred):
    return np.mean(y_pred - y_true * np.log(y_pred + EPSILON), axis=-1)


def cosine_proximity(y_true, y_pred):
    def l2_normalize(x):
        #  x / sqrt(max(sum(x**2), epsilon))
        return x / np.sqrt(np.clip(np.sum(x**2, axis=-1, keepdims=True), EPSILON, None))
    y_true = l2_normalize(y_true)
    y_pred = l2_normalize(y_pred)
    return -np.mean(y_true * y_pred, axis=-1)


# Aliases.

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence
cosine = cosine_proximity


def serialize(loss):
    return loss.__name__


def deserialize(name, custom_objects=None):
    return deserialize_object(name,
                              module_objects=globals(),
                              custom_objects=custom_objects,
                              printable_module_name='loss function')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'loss function identifier:', identifier)
