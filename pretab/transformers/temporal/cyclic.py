import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

class CyclicalTimeTransformer(BaseEstimator, TransformerMixin):
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

    def __init__(self, period: int):
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
