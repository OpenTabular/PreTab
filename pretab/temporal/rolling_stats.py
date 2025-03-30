import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

class RollingStatsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=5, stats=("mean", "std")):
        """
        Computes rolling statistics over a fixed window.

        Parameters:
        - window_size: Number of past observations to use.
        - stats: Tuple of stats to compute: any of "mean", "std", "min", "max".
        """
        self.window_size = window_size
        self.stats = stats

    def fit(self, X, y=None):
        X = check_array(X, ensure_2d=True)
        if X.shape[0] < self.window_size:
            raise ValueError("window_size must be less than number of samples.")
        return self

    def transform(self, X):
        X = check_array(X, ensure_2d=True)
        n_samples = X.shape[0]
        if n_samples < self.window_size:
            raise ValueError("Insufficient samples for the given window size.")

        results = []
        for stat in self.stats:
            rolled = np.lib.stride_tricks.sliding_window_view(X, self.window_size, axis=0)
            if stat == "mean":
                stat_val = rolled.mean(axis=2)
            elif stat == "std":
                stat_val = rolled.std(axis=2)
            elif stat == "min":
                stat_val = rolled.min(axis=2)
            elif stat == "max":
                stat_val = rolled.max(axis=2)
            else:
                raise ValueError(f"Unsupported stat: {stat}")
            results.append(stat_val)

        return np.hstack(results)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
