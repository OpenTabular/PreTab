import numpy as np
from sklearn.utils.validation import check_is_fitted

from ...core.base import BasePreTabTransformer
from ...exceptions import InvalidParamError, PretabDataError


class PeriodicEncodingTransformer(BasePreTabTransformer):
    r"""Encode a cyclical variable using sine and cosine harmonics.

    Maps a periodic feature (such as hour of day or day of week) onto smooth
    continuous features so that the cyclic boundary is continuous. Higher
    ``harmonics`` add finer-grained sinusoids, and ``include_original`` keeps the
    raw value alongside the trigonometric encoding.

    Parameters
    ----------
    period : int
        The full cycle length (e.g., 24 for hours, 7 for weekdays).
    harmonics : int, default=1
        The number of sine/cosine harmonic pairs to emit. Harmonic ``h`` uses the
        angle :math:`2\pi h x / p`, so ``harmonics`` pairs contribute
        ``2 * harmonics`` columns per input feature.
    include_original : bool, default=False
        If ``True``, prepend the (validated) raw value as an extra column per
        input feature.

    Notes
    -----
    For a value :math:`x` with period :math:`p`, each harmonic :math:`h` maps to

    .. math::

        \left(\sin\!\left(\frac{2\pi h x}{p}\right),\;
               \cos\!\left(\frac{2\pi h x}{p}\right)\right).

    Each input feature therefore expands into ``2 * harmonics`` columns, plus one
    extra column when ``include_original`` is set. Columns are laid out
    per-feature: the optional original value first, then ``(sin, cos)`` pairs in
    ascending harmonic order.

    This is a **standalone time-series utility**. Although it preserves the row
    count, it takes a required per-feature ``period`` and constrains inputs to
    ``[0, period]``, so it is not wired into :class:`~pretab.preprocessor.Preprocessor`
    (which applies one method uniformly across columns). Apply it directly to the
    relevant cyclical column instead.

    :meth:`fit` rejects a value outside ``[0, period]``, but :meth:`transform` does not
    repeat that check: the trigonometric encoding wraps any input around the cycle
    automatically (``period + x`` maps to the same output as ``x``), which is the
    mathematically correct result for a genuinely cyclic quantity. If you need
    transform-time inputs strictly confined to ``[0, period]`` as well, validate them
    before calling :meth:`transform`.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import PeriodicEncodingTransformer
    >>> X = np.array([[0], [6], [12], [18]])
    >>> transformer = PeriodicEncodingTransformer(period=24)
    >>> transformer.fit_transform(X).shape
    (4, 2)
    >>> PeriodicEncodingTransformer(period=24, harmonics=3).fit_transform(X).shape
    (4, 6)
    """

    _allow_nan = False
    _feature_suffix_value = "cyclic"
    _representation_family = "periodic"
    _representation_component_kind = "frequency"

    def __init__(self, period: int, harmonics: int = 1, include_original: bool = False):
        self.period = period
        self.harmonics = harmonics
        self.include_original = include_original

    def _representation_periodic(self):
        """Report periodicity with the configured period length."""
        return True, float(self.period)

    def fit(self, X, y=None):
        X = self._validate(X, reset=True)
        if not isinstance(self.harmonics, (int, np.integer)) or self.harmonics < 1:
            raise InvalidParamError(f"harmonics must be a positive integer; got {self.harmonics!r}.")
        if not np.all((X >= 0) & (X <= self.period)):
            raise PretabDataError("Input should be within the range [0, period].")
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        X = self._validate(X, reset=False)
        blocks = []
        for j in range(X.shape[1]):
            column = X[:, j : j + 1]
            feats = [column] if self.include_original else []
            for harmonic in range(1, self.harmonics + 1):
                angle = 2 * np.pi * harmonic * column / self.period
                feats.append(np.sin(angle))
                feats.append(np.cos(angle))
            blocks.append(np.hstack(feats))
        return np.hstack(blocks)

    def _output_sizes(self) -> list[int]:
        per_feature = 2 * self.harmonics + (1 if self.include_original else 0)
        return [per_feature] * self.n_features_in_
