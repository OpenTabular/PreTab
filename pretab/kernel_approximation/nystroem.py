import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.utils.validation import check_is_fitted

from ..core.base import BasePreTabTransformer
from ..exceptions import InvalidParamError

_NYSTROEM_KERNELS = ("rbf", "poly", "polynomial", "sigmoid", "laplacian", "cosine", "linear", "chi2", "additive_chi2")


class NystroemFeaturesTransformer(BasePreTabTransformer):
    r"""Nystroem kernel-map approximation over the full feature block (multivariate).

    Thin wrapper around :class:`sklearn.kernel_approximation.Nystroem` that builds
    a low-rank approximation of an arbitrary kernel by sampling ``n_components``
    landmark rows from the training data. This is a **standalone, multivariate**
    transformer and is not selectable per column through
    :class:`~pretab.preprocessor.Preprocessor`.

    Parameters
    ----------
    n_components : int, default=100
        Number of landmark points sampled to build the approximation (output
        columns). Clamped to the number of samples by the underlying estimator.
    kernel : str, default="rbf"
        Kernel passed to the underlying :class:`~sklearn.kernel_approximation.Nystroem`
        (e.g. ``"rbf"``, ``"poly"``, ``"sigmoid"``, ``"laplacian"``, ``"cosine"``).
    gamma : float or None, default=None
        Kernel coefficient for the RBF / poly / sigmoid kernels. ``None`` defers
        to the scikit-learn default (``1 / n_features``).
    degree : float, default=3
        Degree of the polynomial kernel (ignored by other kernels).
    coef0 : float, default=1
        Independent term for the poly / sigmoid kernels.
    random_state : int, RandomState instance or None, default=None
        Seeds the landmark sampling for reproducibility.

    Attributes
    ----------
    nystroem_ : Nystroem
        The fitted underlying scikit-learn estimator.
    n_features_in_ : int
        Number of input features seen during ``fit``.
    total_output_dim_ : int
        Total number of output columns (the effective number of landmarks).

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import NystroemFeaturesTransformer
    >>> X = np.random.default_rng(0).uniform(size=(60, 3))
    >>> NystroemFeaturesTransformer(n_components=20, random_state=0).fit_transform(X).shape
    (60, 20)
    """

    _allow_nan = False
    _feature_suffix_value = "nystroem"
    _representation_family = "nystroem"
    _representation_scope = "multivariate"

    def __init__(
        self,
        n_components: int = 100,
        kernel: str = "rbf",
        gamma: float | None = None,
        degree: float = 3,
        coef0: float = 1,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.random_state = random_state

    def fit(self, X, y=None):
        X = self._validate(X, reset=True)
        if not isinstance(self.n_components, (int, np.integer)) or self.n_components < 1:
            raise InvalidParamError(f"n_components must be a positive integer; got {self.n_components!r}.")
        if self.kernel not in _NYSTROEM_KERNELS:
            raise InvalidParamError(f"kernel must be one of {_NYSTROEM_KERNELS}; got {self.kernel!r}.")
        self.nystroem_ = Nystroem(
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            n_components=self.n_components,
            random_state=self.random_state,
        ).fit(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "nystroem_")
        X = self._validate(X, reset=False)
        return np.asarray(self.nystroem_.transform(X))

    def _output_sizes(self) -> list[int]:
        return [np.asarray(self.nystroem_.components_).shape[0]]
