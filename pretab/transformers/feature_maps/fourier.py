import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from ...core.base import BasePreTabTransformer
from ...exceptions import InvalidParamError

_FREQUENCY_STRATEGIES = ("harmonic", "log_spaced", "random")


class FourierFeatureTransformer(BasePreTabTransformer):
    r"""Deterministic Fourier (sine/cosine) feature expansion for numerical data.

    Expands each feature into a bank of sine/cosine pairs at data-derived
    frequencies, giving a smooth periodic basis without requiring a known period
    (unlike :class:`PeriodicEncodingTransformer`). The fundamental frequency is
    set from each feature's observed range at ``fit`` time, and
    ``frequency_strategy`` controls how the ``n_frequencies`` frequencies are
    spread above it.

    Parameters
    ----------
    n_frequencies : int, default=5
        Number of frequencies (sine/cosine pairs) per input feature. Each feature
        expands into ``2 * n_frequencies`` columns, plus one extra column when
        ``include_original`` is set.
    frequency_strategy : {"harmonic", "log_spaced", "random"}, default="harmonic"
        How the frequencies are spaced above the fundamental ``2*pi / range``:
        ``"harmonic"`` uses integer multiples ``k * fundamental``; ``"log_spaced"``
        uses octaves ``2**(k-1) * fundamental``; ``"random"`` draws frequencies
        from a half-normal scaled by the fundamental (seeded by ``random_state``).
    include_original : bool, default=False
        If ``True``, prepend the raw feature value as an extra column per feature.
    random_state : int, RandomState instance or None, default=None
        Seeds the ``"random"`` frequency draw. Unused by the deterministic
        strategies.

    Attributes
    ----------
    offsets_ : list of float
        Per-feature origin (the observed minimum) subtracted before projection.
    frequencies_ : list of ndarray
        Per-feature angular frequencies used to build the sine/cosine bank.
    n_features_in_ : int
        Number of input features seen during ``fit``.
    total_output_dim_ : int
        Total number of output columns produced across all input features.

    Notes
    -----
    For a feature :math:`x` with fitted origin :math:`x_0` and frequency
    :math:`\omega_k`, each frequency contributes

    .. math::

        \left(\sin\!\left(\omega_k (x - x_0)\right),\;
               \cos\!\left(\omega_k (x - x_0)\right)\right).

    Columns are laid out per-feature: the optional raw value first, then the
    sine block followed by the cosine block in ascending frequency order.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import FourierFeatureTransformer
    >>> X = np.linspace(0, 10, 50).reshape(-1, 1)
    >>> FourierFeatureTransformer(n_frequencies=4).fit_transform(X).shape
    (50, 8)
    """

    _allow_nan = False
    _feature_suffix_value = "fourier"

    def __init__(
        self,
        n_frequencies: int = 5,
        frequency_strategy: str = "harmonic",
        include_original: bool = False,
        random_state: int | None = None,
    ):
        self.n_frequencies = n_frequencies
        self.frequency_strategy = frequency_strategy
        self.include_original = include_original
        self.random_state = random_state

    def _build_frequencies(self, span, rng):
        """Return the ``n_frequencies`` angular frequencies for a feature span."""
        fundamental = 2.0 * np.pi / span
        k = np.arange(1, self.n_frequencies + 1)
        if self.frequency_strategy == "harmonic":
            return fundamental * k
        if self.frequency_strategy == "log_spaced":
            return fundamental * (2.0 ** (k - 1))
        # random: half-normal spread around the fundamental.
        return np.abs(rng.normal(loc=0.0, scale=fundamental, size=self.n_frequencies))

    def fit(self, X, y=None):
        X = self._validate(X, reset=True)
        if not isinstance(self.n_frequencies, (int, np.integer)) or self.n_frequencies < 1:
            raise InvalidParamError(f"n_frequencies must be a positive integer; got {self.n_frequencies!r}.")
        if self.frequency_strategy not in _FREQUENCY_STRATEGIES:
            raise InvalidParamError(
                f"frequency_strategy must be one of {_FREQUENCY_STRATEGIES}; got {self.frequency_strategy!r}."
            )

        rng = check_random_state(self.random_state)
        self.offsets_ = []
        self.frequencies_ = []
        for j in range(X.shape[1]):
            column = X[:, j]
            low = float(np.min(column))
            span = float(np.max(column) - low)
            if not np.isfinite(span) or span <= 0.0:
                span = 1.0
            self.offsets_.append(low)
            self.frequencies_.append(self._build_frequencies(span, rng))
        return self

    def transform(self, X):
        check_is_fitted(self, "frequencies_")
        X = self._validate(X, reset=False)
        blocks = []
        for j in range(X.shape[1]):
            column = X[:, j : j + 1]
            angles = (column - self.offsets_[j]) * self.frequencies_[j]
            feats = [X[:, j : j + 1]] if self.include_original else []
            feats.append(np.sin(angles))
            feats.append(np.cos(angles))
            blocks.append(np.hstack(feats))
        return np.hstack(blocks)

    def _output_sizes(self) -> list[int]:
        per_feature = 2 * self.n_frequencies + (1 if self.include_original else 0)
        return [per_feature] * self.n_features_in_
