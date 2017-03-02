import numpy as np
from base import BaseEstimator

class RandomForest(BaseEstimator):

    def __init__(self, n_estimators=50, n_choosen=10):
        self.n_estimators=n_estimators
        self.n_choosen=n_choosen
    
    def fit(self, X, y):
        self._setup_input(X, y)