import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .mixins import SplineBasisMixin


class ThinPlateSplineTransformer(SplineBasisMixin, TransformerMixin, BaseEstimator):
    """
    Thin Plate Spline Transformer for smooth univariate basis expansion.

    This transformer constructs a smooth, nonparametric basis using eigen-decomposed thin plate spline (TPS) kernels.
    It supports only univariate input and is useful for modeling smooth nonlinear functions in regression tasks.
    The basis functions are derived from the principal components of the projected TPS kernel matrix.

    Parameters
    ----------
    n_basis : int, default=10
        Number of basis functions to extract from the eigen-decomposition of the TPS kernel.

    Attributes
    ----------
    x_ : ndarray of shape (n_samples, 1)
        Training input used to compute the TPS kernel and projection matrix.

    Z_ : ndarray of shape (n_samples, 2)
        Matrix containing intercept and linear term (used for null space projection).

    eigvals_ : ndarray of shape (n_basis,)
        Top eigenvalues from the projected kernel matrix.

    basis_ : ndarray of shape (n_samples, n_basis)
        Orthogonal basis functions corresponding to the top eigenvectors.

    penalty_ : ndarray of shape (n_basis, n_basis)
        Diagonal penalty matrix containing eigenvalues (used for smoothing regularization).

    n_features_in_ : int
        Number of input features seen during `fit`.

    Notes
    -----
    - Input must be univariate. Multivariate input will raise a ValueError.
    - Basis functions are derived from a kernel matrix projected onto the orthogonal complement of the null space
      of the linear terms (intercept and slope).
    - The transformer uses an eigendecomposition of the projected TPS kernel to define the basis.

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
    >>> transformer = ThinPlateSplineTransformer(n_basis=6)
    >>> transformer.fit_transform(X).shape
    (30, 6)
    """

    _feature_suffix_value = "tps"

    def __init__(self, n_basis=10):
        self.n_basis = n_basis

    def _tps_kernel(self, r):
        with np.errstate(divide="ignore", invalid="ignore"):
            log_r = np.where(r == 0, 0, np.log(r))
            K = r**2 * log_r
            K[r == 0] = 0
        return K

    def fit(self, X, y=None):
        X = self._validate_allow_nan(X, reset=True)

        if X.shape[1] > 1:
            raise ValueError("ThinPlateSplineTransformer supports only univariate input.")

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

        self.eigvals_ = eigvals[: self.n_basis]
        self.basis_ = eigvecs[:, : self.n_basis] * np.sqrt(n)
        self.penalty_ = np.diag(self.eigvals_)
        self.n_basis_ = [self.basis_.shape[1]]

        return self

    def transform(self, X):
        check_is_fitted(self, "basis_")
        X = self._validate_allow_nan(X, reset=False)
        if X.shape[1] > 1:
            raise ValueError("ThinPlateSplineTransformer supports only univariate input.")

        x_new = X.reshape(-1, 1)
        r_new = cdist(x_new, self.x_, metric="euclidean")
        K_new = self._tps_kernel(r_new)

        Z = self.Z_
        ZTZ_inv = np.linalg.pinv(Z.T @ Z)
        P_new = np.eye(Z.shape[0]) - Z @ ZTZ_inv @ Z.T
        K_new_proj = K_new @ P_new

        return K_new_proj @ self.basis_

    def get_penalty_matrix(self):
        """Return the smoothing penalty matrix for the fitted basis.

        Returns
        -------
        penalty_ : ndarray of shape (n_basis, n_basis)
            Diagonal penalty matrix of eigenvalues used for regularization.
        """
        check_is_fitted(self, "penalty_")
        return self.penalty_
