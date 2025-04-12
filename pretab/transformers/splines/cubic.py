import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
import warnings


class CubicSplineTransformer(BaseEstimator, TransformerMixin):
    """
    Cubic Spline Transformer for one-dimensional or multi-dimensional input features.

    This transformer applies cubic spline basis expansions to continuous features using uniformly spaced knots.
    The output includes standard polynomial features up to cubic degree and cubic spline basis functions derived
    from shifted knot positions. Optionally, a bias term can be included.

    Parameters
    ----------
    n_knots : int, default=10
        Number of internal knots to place uniformly between the minimum and maximum of each feature.

    degree : int, default=3
        Degree of the polynomial spline. Currently fixed to 3 (cubic), included for compatibility.

    include_bias : bool, default=False
        Whether to include a bias (intercept) term in the output feature set.

    Attributes
    ----------
    knots_ : list of ndarray
        List of arrays containing the knots used for each feature.

    designs_ : list of ndarray
        List of design matrices (spline basis evaluations) for each input feature during fitting.

    n_features_in_ : int
        Number of input features seen during `fit`.

    Methods
    -------
    get_penalty_matrix(feature_index=0)
        Returns the penalty matrix for regularization of the spline basis functions of a specific feature.
        Penalizes the second derivative (i.e., curvature) of the spline for smoothness.

    Notes
    -----
    The basis includes:
    - Polynomial terms: x, x^2, x^3
    - Truncated power basis functions: (x - knot)^3_+

    The implementation is based on B-spline basis functions but follows a truncated power basis formulation.
    Each transformed feature is expanded to a higher-dimensional representation depending on `n_knots` and
    whether bias is included.

    This transformer supports multidimensional input and stacks all expanded features horizontally.
    """

    def __init__(self, n_knots=10, degree=3, include_bias=False):
        self.n_knots = n_knots
        self.degree = degree
        self.include_bias = include_bias

    def _bspline_basis(self, x, knots):
        x = np.asarray(x).reshape(-1, 1)
        n_samples = x.shape[0]

        X = [np.ones((n_samples, 1))] if self.include_bias else []
        X.append(x)
        X.append(x**2)
        X.append(x**3)

        for knot in knots:
            X.append(np.maximum(0, (x - knot)) ** 3)

        return np.hstack(X)

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
            self.designs_.append(self._bspline_basis(xi, knots))

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
            design = self._bspline_basis(xi, self.knots_[i])
            transformed.append(design)

        return np.hstack(transformed)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_penalty_matrix(self, feature_index=0):
        n_basis = self.designs_[feature_index].shape[1]
        P = np.zeros((n_basis, n_basis))
        offset = 4 if self.include_bias else 3
        for i in range(offset, n_basis):
            for j in range(offset, n_basis):
                ki = self.knots_[feature_index][i - offset]
                kj = self.knots_[feature_index][j - offset]
                P[i, j] = self._spline_penalty_entry(ki, kj, self.knots_[feature_index])
        return P

    def _spline_penalty_entry(self, ki, kj, knots):
        kmax = max(ki, kj)
        upper = knots[-1]
        x_vals = np.linspace(kmax, upper, 100)
        integrand = 36 * (x_vals - ki) * (x_vals - kj)
        return np.trapezoid(integrand, x_vals)
