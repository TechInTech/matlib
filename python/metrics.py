import numpy as np

def mean_squared_error(x, y):
    return np.mean((x-y)**2)


def l2_distance(X):
    sum_X = np.sum(X*X, axis=1)
    return (-2 * np.dot(X, X.T) + sum_X).T + sum_X

def euclidean_distance(a, b):
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    if b.ndim == 1:
        b = b.reshape(1,-1)
        return np.sqrt(np.sum((a - b) ** 2, axis=1))
    else:
        dists = []
        for xb in b:
            dists.append(np.sqrt(np.sum((a - xb) ** 2, axis=1)))
        return dists

def manhaton_distance(a, b):
    return np.sum(np.abs(a-b), axis=1)

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
    return 1-(py*py).sum()


def batch_iterator(X, batch_size=64):
    """Splits X into equal sized chunks."""
    n_samples = X.shape[0]
    n_batches = n_samples // batch_size
    batch_end = 0

    for b in range(n_batches):
        batch_begin = b * batch_size
        batch_end = batch_begin + batch_size

        X_batch = X[batch_begin:batch_end]

        yield X_batch

    if n_batches * batch_size < n_samples:
        yield X[batch_end:]