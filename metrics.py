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