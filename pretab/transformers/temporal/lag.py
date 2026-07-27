import numpy as np
from sklearn.utils.validation import check_is_fitted

from ...core.base import BasePreTabTransformer
from ...core.exceptions import InsufficientSamplesError


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

    This is a **standalone time-series utility**. It intentionally changes the
    row count and assumes the rows are ordered in time, so it does not satisfy
    the row-count-preserving contract that :class:`~sklearn.compose.ColumnTransformer`
    (and therefore :class:`~pretab.preprocessor.Preprocessor`) require. Apply it
    directly to an ordered array rather than routing it through the preprocessing
    pipeline.

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
            raise InsufficientSamplesError("n_lags must be smaller than the number of samples.")
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        X = self._validate(X, reset=False)
        n_samples = X.shape[0]
        if n_samples <= self.n_lags:
            raise InsufficientSamplesError("n_lags must be smaller than the number of samples.")

        lagged = [X[self.n_lags - i: -i or None] for i in range(1, self.n_lags + 1)]
        return np.hstack(lagged)

    def _output_sizes(self) -> list[int]:
        return [self.n_lags] * self.n_features_in_

    def get_feature_names_out(self, input_features=None):
        """Return output names in the lag-major order ``transform`` produces.

        ``transform`` hstacks one ``(n_rows, n_features)`` block per lag, so the
        columns run ``lag1_f0, lag1_f1, ..., lag2_f0, ...``. The inherited
        feature-major default would label them the other way round, mislabelling
        every column but the first and last for multi-feature input.
        """
        check_is_fitted(self, "n_features_in_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(
            [
                f"{feature}_lag{lag}"
                for lag in range(self.n_lags)
                for feature in input_features
            ],
            dtype=object,
        )
