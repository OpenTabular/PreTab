import numpy as np
from sklearn.utils.validation import check_is_fitted

from ...core.base import BasePreTabTransformer
from ...core.exceptions import InsufficientSamplesError, invalid_param_error


class RollingStatsTransformer(BasePreTabTransformer):
    """Compute rolling-window statistics over time-series inputs.

    A sliding window of fixed size is moved across each feature and the requested
    summary statistics are computed within each window.

    Parameters
    ----------
    window_size : int, default=5
        Number of consecutive observations in each rolling window.
    stats : tuple of str, default=("mean", "std")
        Statistics to compute. Any of ``"mean"``, ``"std"``, ``"min"``, ``"max"``.

    Notes
    -----
    Using a sliding window of size ``window_size`` yields
    ``n_samples - window_size + 1`` output rows. Each requested statistic adds one
    column per input feature.

    This is a **standalone time-series utility**. It intentionally changes the
    row count and assumes the rows are ordered in time, so it does not satisfy
    the row-count-preserving contract that :class:`~sklearn.compose.ColumnTransformer`
    (and therefore :class:`~pretab.preprocessor.Preprocessor`) require. Apply it
    directly to an ordered array rather than routing it through the preprocessing
    pipeline.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import RollingStatsTransformer
    >>> X = np.arange(10).reshape(-1, 1).astype(float)
    >>> transformer = RollingStatsTransformer(window_size=3, stats=("mean", "std"))
    >>> transformer.fit_transform(X).shape
    (8, 2)
    """

    _allow_nan = False
    _feature_suffix_value = "roll"

    def __init__(self, window_size=5, stats=("mean", "std")):
        self.window_size = window_size
        self.stats = stats

    def fit(self, X, y=None):
        X = self._validate(X, reset=True)
        if X.shape[0] < self.window_size:
            raise InsufficientSamplesError("window_size must be less than number of samples.")
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        X = self._validate(X, reset=False)
        n_samples = X.shape[0]
        if n_samples < self.window_size:
            raise InsufficientSamplesError("Insufficient samples for the given window size.")

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
                raise invalid_param_error(
                    type(self).__name__,
                    "stats",
                    stat,
                    "each stat must be one of 'mean', 'std', 'min', 'max'",
                    valid={"mean", "std", "min", "max"},
                )
            results.append(stat_val)

        return np.hstack(results)

    def _output_sizes(self) -> list[int]:
        return [len(self.stats)] * self.n_features_in_
