import numpy as np
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


def l2_distance(X):
    sum_X = np.sum(X * X, axis=1)
    return (-2 * np.dot(X, X.T) + sum_X).T + sum_X

# Aliases.

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence
cosine = cosine_proximity


def euclidean_distance(a, b):
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    if b.ndim == 1:
        b = b.reshape(1, -1)
        return np.sqrt(np.sum((a - b) ** 2, axis=1))
    else:
        dists = []
        for xb in b:
            dists.append(np.sqrt(np.sum((a - xb) ** 2, axis=1)))
        return dists


def manhaton_distance(a, b):
    return np.sum(np.abs(a - b), axis=1)


def entropy_criterion(y):
    yu = np.unique(y)
    py = np.zeros(shape=(len(yu)))
    for i in range(len(yu)):
        py[i] = np.sum([y == yu[i]]) / y.shape[0]
    entropy = np.sum(-py * np.log(py))
    return entropy


def gini_criterion(y, weights=None):
    yu = np.unique(y)
    py = None
    if weights is None:
        py = [np.sum([y == yi]) / y.shape[0] for yi in yu]
    else:
        py = [np.sum(weights[y == yi]) for yi in yu]
        py /= np.sum(py)
    return 1 - (py * py).sum()


def batch_iterator(X, y=None, batch_size=64):
    """Splits X into equal sized chunks."""
    if y is not None:
        assert(y.shape[0] == X.shape[0])
    n_samples = X.shape[0]
    n_batches = n_samples // batch_size
    batch_end = 0

    for b in range(n_batches):
        batch_begin = b * batch_size
        batch_end = batch_begin + batch_size

        if y is None:
            X_batch = X[batch_begin:batch_end]
            yield X_batch
        else:
            yield X[batch_begin:batch_end], y[batch_begin:batch_end]

    if n_batches * batch_size < n_samples:
        if y is None:
            yield X[batch_end:]
        else:
            yield X[batch_end:], y[batch_end:]
