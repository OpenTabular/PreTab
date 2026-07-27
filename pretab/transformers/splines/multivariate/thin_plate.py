import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from ....exceptions import InsufficientSamplesError, InvalidParamError
from ..mixins import SplineBasisMixin

_LANDMARK_STRATEGIES = ("kmeans", "subsample")
_RANK_STRATEGIES = ("eigen", "nystroem")


class ThinPlateSplineTransformer(SplineBasisMixin, TransformerMixin, BaseEstimator):
    r"""Multivariate low-rank thin-plate regression spline basis.

    Builds a smooth thin-plate spline (TPS) feature map that jointly models all
    input features. A set of ``n_components + d + 1`` landmark points is chosen
    from the data (``d`` is the number of input features), the null-space
    (linear-polynomial) part is projected out of the landmark TPS kernel, and the
    leading eigenvectors of the projected kernel form a rank-``n_components``
    basis. This landmark construction follows the low-rank thin-plate regression
    spline of Wood (2003) and keeps the cost governed by ``n_components`` rather
    than the sample size.

    Parameters
    ----------
    n_components : int, default=10
        Number of (non-bias) basis functions to emit -- the rank of the
        approximation and the output width. Must be at least 1. Fitting requires
        at least ``n_components + d + 1`` samples.
    landmark_strategy : {"kmeans", "subsample"}, default="kmeans"
        How the landmark points are chosen from the data. ``"kmeans"`` uses
        k-means cluster centers (space-filling); ``"subsample"`` draws a random
        subset of the observed rows.
    rank_strategy : {"eigen", "nystroem"}, default="eigen"
        How the reduced basis is extracted from the projected landmark kernel.
        ``"eigen"`` keeps the leading (raw) eigenvectors; ``"nystroem"`` whitens
        them by the inverse square-root of the eigenvalues to decorrelate the
        features. Both emit exactly ``n_components`` columns.
    include_bias : bool, default=False
        If True, prepend a constant intercept column to the output. The bias term
        is left unpenalized (a zero leading row/column is added to the penalty).
    random_state : int, RandomState instance or None, default=None
        Seeds the landmark selection (k-means initialization or subsampling).

    Attributes
    ----------
    landmarks_ : ndarray of shape (n_landmarks, n_features)
        The landmark points used to build the TPS kernel.
    components_ : ndarray of shape (n_landmarks, n_components)
        The linear map from a data-to-landmark kernel row to the reduced basis.
    eigvals_ : ndarray of shape (n_components,)
        The retained eigenvalues of the projected landmark kernel.
    penalty_ : ndarray
        Diagonal smoothing penalty of ``eigvals_`` (with an unpenalized leading
        row/column when ``include_bias=True``).
    d_ : int
        Number of input features (also ``n_features_in_``).
    n_basis_ : list of int
        Single-element list with the output width (``n_components`` plus the
        optional bias).
    n_features_in_ : int
        Number of input features seen during ``fit``.
    total_output_dim_ : int
        Total number of output columns (fitted); equals ``n_components``
        (``+1`` when ``include_bias``).

    Notes
    -----
    - Unlike the knot-based spline families, the thin-plate basis is kernel-based:
      the knot-oriented options (``degree``, ``target_aware``,
      ``placement_strategy``, ``task``) do not apply.
    - The radial kernel depends on the input dimension: :math:`r^3` for ``d=1``,
      :math:`r^2\log r` for ``d=2``, and the biharmonic kernel :math:`r` for
      ``d>=3``.
    - The construction follows the thin-plate spline theory of Wahba [1]_ and the
      low-rank thin-plate regression spline of Wood [2]_.

    References
    ----------
    .. [1] Wahba, G. (1990). "Spline Models for Observational Data". SIAM.
    .. [2] Wood, S.N. (2003). "Thin plate regression splines". Journal of the
       Royal Statistical Society: Series B.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import ThinPlateSplineTransformer
    >>> X = np.random.default_rng(0).uniform(size=(60, 2))
    >>> transformer = ThinPlateSplineTransformer(n_components=6, random_state=0)
    >>> transformer.fit_transform(X).shape
    (60, 6)
    >>> transformer.total_output_dim_
    6
    """

    _feature_suffix_value = "tps"
    _representation_family = "thinplate"
    _representation_scope = "multivariate"

    def __init__(
        self,
        n_components=10,
        landmark_strategy="kmeans",
        rank_strategy="eigen",
        include_bias=False,
        random_state=None,
    ):
        self.n_components = n_components
        self.landmark_strategy = landmark_strategy
        self.rank_strategy = rank_strategy
        self.include_bias = include_bias
        self.random_state = random_state

    @staticmethod
    def _tps_kernel(r, d):
        """Return the thin-plate radial kernel for input dimension ``d``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            if d == 1:
                return r**3
            if d == 2:
                return np.where(r > 0, r**2 * np.log(np.where(r > 0, r, 1.0)), 0.0)
            # d >= 3: biharmonic (linear) radial kernel.
            return r

    def _select_landmarks(self, X, n_landmarks, rng):
        """Choose ``n_landmarks`` landmark points from ``X``."""
        n = X.shape[0]
        if n_landmarks >= n:
            return X
        if self.landmark_strategy == "kmeans":
            return KMeans(n_clusters=n_landmarks, random_state=rng, n_init=10).fit(X).cluster_centers_
        idx = rng.choice(n, size=n_landmarks, replace=False)
        return X[idx]

    def fit(self, X, y=None):
        X = self._validate_allow_nan(X, reset=True)

        if not isinstance(self.n_components, (int, np.integer)) or self.n_components < 1:
            raise InvalidParamError(f"n_components must be a positive integer; got {self.n_components!r}.")
        if self.landmark_strategy not in _LANDMARK_STRATEGIES:
            raise InvalidParamError(
                f"landmark_strategy must be one of {_LANDMARK_STRATEGIES}; got {self.landmark_strategy!r}."
            )
        if self.rank_strategy not in _RANK_STRATEGIES:
            raise InvalidParamError(f"rank_strategy must be one of {_RANK_STRATEGIES}; got {self.rank_strategy!r}.")

        n, d = X.shape
        n_landmarks = self.n_components + d + 1
        if n < n_landmarks:
            raise InsufficientSamplesError(
                f"ThinPlateSplineTransformer with n_components={self.n_components} on {d} feature(s) "
                f"needs at least {n_landmarks} samples; got {n}."
            )

        rng = check_random_state(self.random_state)
        C = np.asarray(self._select_landmarks(X, n_landmarks, rng), dtype=float)
        length = C.shape[0]
        self.landmarks_ = C

        # Project out the linear-polynomial null space on the landmarks.
        T = np.hstack([np.ones((length, 1)), C])
        P = np.eye(length) - T @ np.linalg.pinv(T.T @ T) @ T.T

        K = self._tps_kernel(cdist(C, C), d)
        K_proj = P @ K @ P
        K_proj = 0.5 * (K_proj + K_proj.T)  # symmetrize against round-off

        eigvals, eigvecs = eigh(K_proj)
        order = np.argsort(np.abs(eigvals))[::-1][: self.n_components]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        self.eigvals_ = eigvals

        if self.rank_strategy == "nystroem":
            scale = 1.0 / np.sqrt(np.clip(np.abs(eigvals), 1e-12, None))
        else:  # eigen
            scale = np.full(self.n_components, np.sqrt(length))
        # ``components_`` maps a raw data->landmark kernel row into the basis.
        self.components_ = P @ (eigvecs * scale)

        penalty = np.diag(eigvals)
        if self.include_bias:
            penalty = np.pad(penalty, ((1, 0), (1, 0)))
        self.penalty_ = penalty
        self.d_ = d
        self.n_basis_ = [self.n_components + (1 if self.include_bias else 0)]
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        X = self._validate_allow_nan(X, reset=False)
        K_new = self._tps_kernel(cdist(X, self.landmarks_), self.d_)
        out = K_new @ self.components_
        if self.include_bias:
            out = np.hstack([np.ones((out.shape[0], 1)), out])
        return out

    def get_penalty_matrix(self, feature_index=0):
        """Return the smoothing penalty matrix for the fitted basis.

        Parameters
        ----------
        feature_index : int, default=0
            Accepted for signature parity with the other spline transformers;
            ignored because the thin-plate basis is a single joint expansion.

        Returns
        -------
        penalty_ : ndarray
            Diagonal penalty of the retained eigenvalues (with an unpenalized
            leading row/column when ``include_bias=True``).
        """
        check_is_fitted(self, "penalty_")
        return self.penalty_
