import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CubicSplineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_knots=10, degree=3, include_bias=False):
        self.n_knots = n_knots
        self.degree = degree
        self.include_bias = include_bias

    def _bspline_basis(self, x, knots):
        """
        Construct cubic spline basis using truncated power basis
        as done in mgcv (thin plate regression splines approximate this form).
        """
        x = np.asarray(x).reshape(-1, 1)
        n_samples = x.shape[0]

        # Truncated power basis (cubic): x, x^2, x^3, (x - knot)^3_+
        X = [np.ones((n_samples, 1))] if self.include_bias else []
        X.append(x)
        X.append(x ** 2)
        X.append(x ** 3)

        for knot in knots:
            X.append(np.maximum(0, (x - knot)) ** 3)

        return np.hstack(X)

    def fit(self, X, y=None):
        x = np.asarray(X).ravel()
        x_min, x_max = np.min(x), np.max(x)

        # Choose knots uniformly within the range
        self.knots_ = np.linspace(x_min, x_max, self.n_knots)
        self.X_design_ = self._bspline_basis(x, self.knots_)
        return self

    def transform(self, X):
        x = np.asarray(X).ravel()
        return self._bspline_basis(x, self.knots_)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_penalty_matrix(self):
        """
        Return the roughness penalty matrix (integral of squared second derivative).
        This penalizes the spline curvature.
        """
        # Build penalty matrix for truncated power basis
        # The structure: [x, x^2, x^3, (x - k1)^3_+, ..., (x - km)^3_+]
        n_basis = self.X_design_.shape[1]
        P = np.zeros((n_basis, n_basis))

        # Only penalize the (x - knot)^3_+ terms (not intercept or linear part)
        offset = 4 if self.include_bias else 3
        for i in range(offset, n_basis):
            for j in range(offset, n_basis):
                # Approximate inner product of second derivatives over domain
                ki, kj = self.knots_[i - offset], self.knots_[j - offset]
                P[i, j] = self._spline_penalty_entry(ki, kj)

        return P

    def _spline_penalty_entry(self, ki, kj):
        """
        Compute the integral of the product of second derivatives of
        basis functions (x - ki)^3_+ and (x - kj)^3_+.
        """
        # The second derivative of (x - k)^3_+ is 6(x - k)_+ for x > k
        # The integral over [max(ki, kj), ∞) of 36(x - ki)(x - kj) dx
        kmax = max(ki, kj)
        upper = self.knots_[-1]
        x_vals = np.linspace(kmax, upper, 100)
        integrand = 36 * (x_vals - ki) * (x_vals - kj)
        return np.trapz(integrand, x_vals)
