import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_lags=1):
        """
        Creates lag features for time series inputs.

        Parameters:
        - n_lags: Number of lag steps to include.
        """
        self.n_lags = n_lags

    def fit(self, X, y=None):
        X = check_array(X, ensure_2d=True)
        if X.shape[0] <= self.n_lags:
            raise ValueError("n_lags must be smaller than the number of samples.")
        return self

    def transform(self, X):
        X = check_array(X, ensure_2d=True)
        n_samples, n_features = X.shape
        if n_samples <= self.n_lags:
            raise ValueError("n_lags must be smaller than the number of samples.")

        lagged = [X[self.n_lags - i: -i or None] for i in range(1, self.n_lags + 1)]
        return np.hstack(lagged)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
