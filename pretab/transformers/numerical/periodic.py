import numpy as np
from sklearn.utils.validation import check_is_fitted

from ...core.base import BasePreTabTransformer
from ...exceptions import PretabDataError


class PeriodicEncodingTransformer(BasePreTabTransformer):
    r"""Encode a cyclical time variable using sine and cosine components.

    Maps a periodic integer feature (such as hour of day or day of week) onto two
    continuous features so that the cyclic boundary is continuous.

    Parameters
    ----------
    period : int
        The full cycle length (e.g., 24 for hours, 7 for weekdays).

    Notes
    -----
    For a value :math:`x` with period :math:`p`, the encoding is

    .. math::

        \left(\sin\!\left(\frac{2\pi x}{p}\right),\;
               \cos\!\left(\frac{2\pi x}{p}\right)\right).

    Each input feature therefore expands into two output columns.

    This is a **standalone time-series utility**. Although it preserves the row
    count, it takes a required per-feature ``period`` and constrains inputs to
    ``[0, period]``, so it is not wired into :class:`~pretab.preprocessor.Preprocessor`
    (which applies one method uniformly across columns). Apply it directly to the
    relevant cyclical column instead.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import PeriodicEncodingTransformer
    >>> X = np.array([[0], [6], [12], [18]])
    >>> transformer = PeriodicEncodingTransformer(period=24)
    >>> transformer.fit_transform(X).shape
    (4, 2)
    """

    _allow_nan = False
    _feature_suffix_value = "cyclic"

    def __init__(self, period: int):
        self.period = period

    def fit(self, X, y=None):
        X = self._validate(X, reset=True)
        if not np.all((X >= 0) & (X <= self.period)):
            raise PretabDataError("Input should be within the range [0, period].")
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        X = self._validate(X, reset=False)
        angle = 2 * np.pi * X / self.period
        sin = np.sin(angle)
        cos = np.cos(angle)
        return np.hstack([sin, cos])

    def _output_sizes(self) -> list[int]:
        return [2] * self.n_features_in_
