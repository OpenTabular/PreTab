import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class NaturalCubicSplineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_knots=5, include_bias=False):
        self.n_knots = n_knots
        self.include_bias = include_bias

    def _basis(self, x, knots):
        """
        Construct the natural cubic spline basis using truncated power basis,
        constrained so that the second derivatives at the endpoints are zero.
        """
        x = np.asarray(x).reshape(-1, 1)
        K = knots
        n_samples = x.shape[0]
        n_knots = len(K)

        # First basis: intercept, x
        basis = [np.ones((n_samples, 1))] if self.include_bias else []
        basis.append(x)

        # Define helper function for the omega term
        def omega(z, k):
            return np.maximum(0, z - k) ** 3

        # Compute the last n_knots - 2 basis functions
        def d(k):
            return omega(x, k) - omega(x, K[-1])
        
        denom = K[-1] - K[0]
        D = np.array([d(k) - ((K[-1] - k) / denom) * d(K[0]) - ((k - K[0]) / denom) * d(K[-1]) for k in K[1:-1]])
        basis.extend(list(D))

        return np.hstack(basis)

    def fit(self, X, y=None):
        x = np.asarray(X).ravel()
        x_min, x_max = np.min(x), np.max(x)
        self.knots_ = np.linspace(x_min, x_max, self.n_knots)
        self.X_design_ = self._basis(x, self.knots_)
        return self

    def transform(self, X):
        x = np.asarray(X).ravel()
        return self._basis(x, self.knots_)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_penalty_matrix(self):
        """
        Return the curvature penalty matrix (integral of squared second derivative).
        This penalizes the smoothness of the spline.
        """
        n_basis = self.X_design_.shape[1]
        P = np.zeros((n_basis, n_basis))

        # Penalize only the natural spline components (not intercept or linear term)
        offset = 2 if self.include_bias else 1
        for i in range(offset, n_basis):
            for j in range(offset, n_basis):
                P[i, j] = self._penalty_entry(i - offset, j - offset)

        return P

    def _penalty_entry(self, i, j):
        """
        Approximate the integral of the product of the second derivatives of the
        i-th and j-th natural spline basis functions.
        """
        # Use numerical approximation over the domain
        x_vals = np.linspace(self.knots_[0], self.knots_[-1], 200)
        B = self._basis(x_vals, self.knots_)
        B_dd = np.gradient(np.gradient(B, axis=0), axis=0)
        integrand = B_dd[:, i] * B_dd[:, j]
        return np.trapz(integrand, x_vals)
