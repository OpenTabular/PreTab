import numpy as np
from sklearn.utils.validation import check_is_fitted

from ...core.base import BasePreTabTransformer


class LagFeatureTransformer(BasePreTabTransformer):
    """Create lagged features for time-series inputs.

    For each input column, previous time steps are appended as additional
    features, which is useful for autoregressive modeling.

    Parameters
    ----------
    n_lags : int, default=1
        Number of lag steps to include.

    Notes
    -----
    Because the first ``n_lags`` observations have no complete history, the
    transformed output has ``n_samples - n_lags`` rows. Each input feature is
    expanded into ``n_lags`` lagged columns.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import LagFeatureTransformer
    >>> X = np.arange(6).reshape(-1, 1)
    >>> transformer = LagFeatureTransformer(n_lags=2)
    >>> transformer.fit_transform(X).shape
    (4, 2)
    """

    _allow_nan = False
    _feature_suffix_value = "lag"

    def __init__(self, n_lags=1):
        self.n_lags = n_lags

    def fit(self, X, y=None):
        X = self._validate(X, reset=True)
        if X.shape[0] <= self.n_lags:
            raise ValueError("n_lags must be smaller than the number of samples.")
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        X = self._validate(X, reset=False)
        n_samples = X.shape[0]
        if n_samples <= self.n_lags:
            raise ValueError("n_lags must be smaller than the number of samples.")

        lagged = [X[self.n_lags - i: -i or None] for i in range(1, self.n_lags + 1)]
        return np.hstack(lagged)

    def _output_sizes(self) -> list[int]:
        return [self.n_lags] * self.n_features_in_
