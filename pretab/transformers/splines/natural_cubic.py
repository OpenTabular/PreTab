import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


class NaturalCubicSplineTransformer(BaseEstimator, TransformerMixin):
    """
    Natural Cubic Spline Transformer for continuous features.

    This transformer expands each input feature using a natural cubic spline basis. Natural cubic splines are
    piecewise cubic polynomials that are linear beyond the boundary knots, ensuring smooth extrapolation.

    The resulting transformation includes:
    - A linear component (and optionally a bias term),
    - Several non-linear basis functions constrained to produce a natural spline.

    Parameters
    ----------
    n_knots : int, default=5
        Number of knots to place uniformly across the range of each feature.
        The spline basis functions are derived from these knots.

    include_bias : bool, default=False
        If True, includes a constant bias (intercept) column in the output.

    Attributes
    ----------
    knots_ : list of ndarray
        List of knot vectors used for each feature.

    designs_ : list of ndarray
        Cached spline basis design matrices (used for penalty computation or inspection).

    n_features_in_ : int
        Number of input features seen during `fit`.

    Methods
    -------
    get_penalty_matrix(feature_index=0)
        Returns the penalty matrix for the second derivative (curvature) of the spline basis for a specific feature.
        Useful for regularization or smoothing in generalized additive models.

    Notes
    -----
    The basis is constructed to satisfy the natural spline constraint: the second derivative of the spline is zero
    at the boundary knots. This reduces the tendency to overfit at the boundaries and improves extrapolation.

    Each feature is transformed independently and their expanded outputs are concatenated.
    """

    def __init__(self, n_knots=5, include_bias=False):
        self.n_knots = n_knots
        self.include_bias = include_bias

    def _basis(self, x, knots):
        x = np.asarray(x).reshape(-1, 1)
        K = knots
        n_samples = x.shape[0]
        n_knots = len(K)

        basis = [np.ones((n_samples, 1))] if self.include_bias else []
        basis.append(x)

        def omega(z, k):
            return np.maximum(0, z - k) ** 3

        def d(k):
            return omega(x, k) - omega(x, K[-1])

        denom = K[-1] - K[0]
        D = np.array(
            [
                d(k) - ((K[-1] - k) / denom) * d(K[0]) - ((k - K[0]) / denom) * d(K[-1])
                for k in K[1:-1]
            ]
        )
        basis.extend(list(D))
        return np.hstack(basis)

    def fit(self, X, y=None):
        original_dim = np.shape(X)[1] if np.ndim(X) == 2 else 1
        X = check_array(
            X, dtype=np.float64, ensure_2d=True, ensure_all_finite="allow-nan"
        )
        if X.shape[1] < original_dim:
            warnings.warn(
                "Some input features were dropped during check_array validation.",
                UserWarning,
            )

        self.knots_ = []
        self.designs_ = []

        for i in range(X.shape[1]):
            xi = X[:, i]
            xi_min, xi_max = np.min(xi), np.max(xi)
            knots = np.linspace(xi_min, xi_max, self.n_knots)
            self.knots_.append(knots)
            self.designs_.append(self._basis(xi, knots))

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        original_dim = np.shape(X)[1] if np.ndim(X) == 2 else 1
        X = check_array(
            X, dtype=np.float64, ensure_2d=True, ensure_all_finite="allow-nan"
        )
        if X.shape[1] < original_dim:
            warnings.warn(
                "Some input features were dropped during check_array validation.",
                UserWarning,
            )

        transformed = []
        for i in range(X.shape[1]):
            xi = X[:, i]
            basis = self._basis(xi, self.knots_[i])
            transformed.append(basis)

        return np.hstack(transformed)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_penalty_matrix(self, feature_index=0):
        knots = self.knots_[feature_index]
        B = self._basis(np.linspace(knots[0], knots[-1], 200), knots)
        B_dd = np.gradient(np.gradient(B, axis=0), axis=0)

        n_basis = B.shape[1]
        P = np.zeros((n_basis, n_basis))
        offset = 2 if self.include_bias else 1

        for i in range(offset, n_basis):
            for j in range(offset, n_basis):
                integrand = B_dd[:, i] * B_dd[:, j]
                P[i, j] = np.trapezoid(integrand, np.linspace(knots[0], knots[-1], 200))

        return P
