import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...core.exceptions import InvalidParamError, PretabDataError
from .mixins import SplineBasisMixin


class ThinPlateSplineTransformer(SplineBasisMixin, TransformerMixin, BaseEstimator):
    r"""
    Thin Plate Spline Transformer for smooth univariate basis expansion.

    This transformer constructs a smooth, nonparametric basis using eigen-decomposed
    thin plate spline (TPS) kernels. It supports only univariate input and is useful
    for modeling smooth nonlinear functions in regression tasks. The basis functions
    are the leading eigenvectors of the projected TPS kernel matrix.

    Let :math:`m := \mathtt{output\_dim}` be the number of non-bias output columns.
    The transformer keeps the top :math:`m` eigenvectors, so the output width equals
    ``output_dim`` directly (this family is knot-free; there is no knot inversion).
    ``include_bias=True`` adds one further intercept column.

    Parameters
    ----------
    output_dim : int, default=10
        Number of non-bias output columns (:math:`m`) extracted from the
        eigen-decomposition of the TPS kernel. Must be at least 1.

    include_bias : bool, default=False
        If True, prepend a constant intercept column to the output. The bias term
        is left unpenalized (a zero row/column is added to the penalty matrix).

    Attributes
    ----------
    x_ : ndarray of shape (n_samples, 1)
        Training input used to compute the TPS kernel and projection matrix.

    Z_ : ndarray of shape (n_samples, 2)
        Matrix containing intercept and linear term (used for null space projection).

    eigvals_ : ndarray of shape (output_dim,)
        Top eigenvalues from the projected kernel matrix.

    basis_ : ndarray of shape (n_samples, output_dim)
        Orthogonal basis functions corresponding to the top eigenvectors.

    penalty_ : ndarray of shape (output_dim, output_dim)
        Diagonal penalty matrix containing eigenvalues (used for smoothing regularization).

    n_basis_ : list of int
        Number of output columns for the single feature, including the optional bias.

    n_features_in_ : int
        Number of input features seen during ``fit`` (always 1).

    total_output_dim_ : int
        Total number of output columns (fitted); equals
        ``output_dim (+1 if include_bias)``.

    Notes
    -----
    - Input must be univariate. Multivariate input will raise a ValueError.
    - Basis functions are derived from a kernel matrix projected onto the orthogonal complement of the null space
      of the linear terms (intercept and slope).
    - The transformer uses an eigendecomposition of the projected TPS kernel to define the basis.
    - The transformer is kernel-based rather than knot-based, so the knot-oriented options shared by the other
      splines (``degree``, ``strategy``, ``selector``, ``task``) do not apply here.

    References
    ----------
    .. [1] Wahba, G. (1990). "Spline Models for Observational Data". SIAM.
    .. [2] Wood, S.N. (2003). "Thin plate regression splines". Journal of the
       Royal Statistical Society: Series B.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import ThinPlateSplineTransformer
    >>> X = np.linspace(0, 1, 30).reshape(-1, 1)
    >>> transformer = ThinPlateSplineTransformer(output_dim=6)
    >>> Xt = transformer.fit_transform(X)
    >>> Xt.shape
    (30, 6)
    >>> transformer.total_output_dim_
    6
    """

    _feature_suffix_value = "tps"

    def __init__(self, output_dim=10, include_bias=False):
        self.output_dim = output_dim
        self.include_bias = include_bias

    def _tps_kernel(self, r):
        with np.errstate(divide="ignore", invalid="ignore"):
            log_r = np.where(r == 0, 0, np.log(r))
            K = r**2 * log_r
            K[r == 0] = 0
        return K

    def fit(self, X, y=None):
        X = self._validate_allow_nan(X, reset=True)

        if X.shape[1] > 1:
            raise PretabDataError("ThinPlateSplineTransformer supports only univariate input.")

        if self.output_dim < 1:
            raise InvalidParamError(f"output_dim must be >= 1, got {self.output_dim}")

        x = X.reshape(-1, 1)
        self.x_ = x
        n = x.shape[0]

        Z = np.hstack([np.ones_like(x), x])
        self.Z_ = Z

        r = cdist(x, x, metric="euclidean")
        K = self._tps_kernel(r)

        ZTZ_inv = np.linalg.pinv(Z.T @ Z)
        P = np.eye(n) - Z @ ZTZ_inv @ Z.T
        KP = P @ K @ P

        eigvals, eigvecs = eigh(KP)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        self.eigvals_ = eigvals[: self.output_dim]
        self.basis_ = eigvecs[:, : self.output_dim] * np.sqrt(n)
        penalty = np.diag(self.eigvals_)
        if self.include_bias:
            penalty = np.pad(penalty, ((1, 0), (1, 0)))
        self.penalty_ = penalty
        self.n_basis_ = [self.basis_.shape[1] + (1 if self.include_bias else 0)]

        return self

    def transform(self, X):
        check_is_fitted(self, "basis_")
        X = self._validate_allow_nan(X, reset=False)
        if X.shape[1] > 1:
            raise PretabDataError("ThinPlateSplineTransformer supports only univariate input.")

        x_new = X.reshape(-1, 1)
        r_new = cdist(x_new, self.x_, metric="euclidean")
        K_new = self._tps_kernel(r_new)

        Z = self.Z_
        ZTZ_inv = np.linalg.pinv(Z.T @ Z)
        P_new = np.eye(Z.shape[0]) - Z @ ZTZ_inv @ Z.T
        K_new_proj = K_new @ P_new

        out = K_new_proj @ self.basis_
        if self.include_bias:
            out = np.hstack([np.ones((out.shape[0], 1)), out])
        return out

    def get_penalty_matrix(self, feature_index=0):
        """Return the smoothing penalty matrix for the fitted basis.

        Parameters
        ----------
        feature_index : int, default=0
            Accepted for signature parity with the other spline transformers;
            ignored because the thin-plate transformer is univariate.

        Returns
        -------
        penalty_ : ndarray of shape (output_dim, output_dim)
            Diagonal penalty matrix of eigenvalues used for regularization (with
            an unpenalized leading row/column when ``include_bias=True``).
        """
        check_is_fitted(self, "penalty_")
        return self.penalty_
