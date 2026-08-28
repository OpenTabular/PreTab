import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.utils.validation import check_is_fitted

from ..core.base import BasePreTabTransformer
from ..exceptions import InvalidParamError


class RandomFourierFeaturesTransformer(BasePreTabTransformer):
    r"""Random Fourier features approximating an RBF kernel map (multivariate).

    Thin wrapper around :class:`sklearn.kernel_approximation.RBFSampler` that
    jointly maps all input features into a randomized low-dimensional feature
    space whose inner products approximate a Gaussian (RBF) kernel. This is a
    **standalone, multivariate** transformer: it models the feature block as a
    whole and is therefore not selectable per column through
    :class:`~pretab.preprocessor.Preprocessor`.

    Parameters
    ----------
    n_components : int, default=100
        Number of Monte-Carlo random features (output columns).
    gamma : float, default=1.0
        Bandwidth of the approximated RBF kernel ``exp(-gamma * ||x - y||^2)``.
    random_state : int, RandomState instance or None, default=None
        Seeds the random projection for reproducibility.

    Attributes
    ----------
    sampler_ : RBFSampler
        The fitted underlying scikit-learn sampler.
    n_features_in_ : int
        Number of input features seen during ``fit``.
    total_output_dim_ : int
        Total number of output columns (equals ``n_components``).

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import RandomFourierFeaturesTransformer
    >>> X = np.random.default_rng(0).uniform(size=(40, 3))
    >>> RandomFourierFeaturesTransformer(n_components=20, random_state=0).fit_transform(X).shape
    (40, 20)
    """

    _allow_nan = False
    _feature_suffix_value = "rff"
    _representation_family = "random_fourier"
    _representation_scope = "multivariate"

    def __init__(self, n_components: int = 100, gamma: float = 1.0, random_state: int | None = None):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None):
        X = self._validate(X, reset=True)
        if not isinstance(self.n_components, (int, np.integer)) or self.n_components < 1:
            raise InvalidParamError(f"n_components must be a positive integer; got {self.n_components!r}.")
        self.sampler_ = RBFSampler(
            n_components=self.n_components,
            gamma=self.gamma,
            random_state=self.random_state,
        ).fit(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "sampler_")
        X = self._validate(X, reset=False)
        return np.asarray(self.sampler_.transform(X))

    def _output_sizes(self) -> list[int]:
        return [self.n_components]
