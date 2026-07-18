import numpy as np
from sklearn.utils.validation import check_is_fitted

from ...core.base import BasePreTabTransformer
from ...core.exceptions import PretabDataError


class CyclicalTimeTransformer(BasePreTabTransformer):
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

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import CyclicalTimeTransformer
    >>> X = np.array([[0], [6], [12], [18]])
    >>> transformer = CyclicalTimeTransformer(period=24)
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
