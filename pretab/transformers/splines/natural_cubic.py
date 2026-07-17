from typing import ClassVar

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...core.params import UNSET
from .mixins import SplineBasisMixin


class NaturalCubicSplineTransformer(SplineBasisMixin, TransformerMixin, BaseEstimator):
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

    Notes
    -----
    The basis is constructed to satisfy the natural spline constraint: the second derivative of the spline is zero
    at the boundary knots. This reduces the tendency to overfit at the boundaries and improves extrapolation.

    Each feature is transformed independently and their expanded outputs are concatenated.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import NaturalCubicSplineTransformer
    >>> X = np.linspace(0, 1, 20).reshape(-1, 1)
    >>> transformer = NaturalCubicSplineTransformer(n_knots=5)
    >>> transformer.fit_transform(X).shape
    (20, 4)
    """

    _feature_suffix_value = "ncs"
    _param_aliases: ClassVar[dict[str, str]] = {"n_knots": "n_basis"}

    def __init__(self, n_basis=UNSET, include_bias=False, n_knots=UNSET):
        self.n_basis = n_basis
        self.include_bias = include_bias
        self.n_knots = n_knots

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
        D = np.array([d(k) - ((K[-1] - k) / denom) * d(K[0]) - ((k - K[0]) / denom) * d(K[-1]) for k in K[1:-1]])
        basis.extend(list(D))
        return np.hstack(basis)

    def fit(self, X, y=None):
        X = self._validate_allow_nan(X, reset=True)
        n_basis = self._resolve_param("n_basis", default=5)

        self.knots_ = []
        self.designs_ = []

        for i in range(X.shape[1]):
            xi = X[:, i]
            xi_min, xi_max = np.min(xi), np.max(xi)
            knots = np.linspace(xi_min, xi_max, n_basis)
            self.knots_.append(knots)
            self.designs_.append(self._basis(xi, knots))

        self.n_basis_ = [design.shape[1] for design in self.designs_]
        return self

    def transform(self, X):
        check_is_fitted(self, "n_basis_")
        X = self._validate_allow_nan(X, reset=False)

        transformed = []
        for i in range(X.shape[1]):
            xi = X[:, i]
            basis = self._basis(xi, self.knots_[i])
            transformed.append(basis)

        return np.hstack(transformed)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_penalty_matrix(self, feature_index=0):
        """Return the curvature penalty matrix for a fitted feature.

        Parameters
        ----------
        feature_index : int, default=0
            Index of the feature whose penalty matrix is returned.

        Returns
        -------
        P : ndarray of shape (n_basis, n_basis)
            Penalty matrix approximating the integrated squared second derivative
            of the natural cubic spline basis.
        """
        check_is_fitted(self, "knots_")
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
