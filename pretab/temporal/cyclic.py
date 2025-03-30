import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

class CyclicalTimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, period: int):
        """
        Encodes a cyclical time variable (e.g., hour of day, day of week).

        Parameters:
        - period: The full cycle length (e.g., 24 for hours, 7 for weekdays).
        """
        self.period = period

    def fit(self, X, y=None):
        X = check_array(X, ensure_2d=True)
        if not np.all((X >= 0) & (X <= self.period)):
            raise ValueError("Input should be within the range [0, period].")
        return self

    def transform(self, X):
        X = check_array(X, ensure_2d=True)
        angle = 2 * np.pi * X / self.period
        sin = np.sin(angle)
        cos = np.cos(angle)
        return np.hstack([sin, cos])

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
