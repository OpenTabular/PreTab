import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ISplineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_knots=5, degree=3, include_bias=False):
        self.n_knots = n_knots
        self.degree = degree
        self.include_bias = include_bias

    def _make_knots(self, x):
        x = np.asarray(x)
        x_min, x_max = np.min(x), np.max(x)
        # Add boundary knots for spline degree
        inner_knots = np.linspace(x_min, x_max, self.n_knots)
        t = np.concatenate((
            np.repeat(inner_knots[0], self.degree + 1),
            inner_knots,
            np.repeat(inner_knots[-1], self.degree + 1)
        ))
        return t

    def _m_spline_basis(self, x, t, k):
        """
        Evaluate M-spline basis (degree k) at positions x with knot vector t.
        """
        x = np.atleast_1d(x)
        n = len(t) - k - 1
        B = np.zeros((len(x), n))

        # Degree 0 (piecewise constant)
        for i in range(n):
            B[:, i] = ((t[i] <= x) & (x < t[i + 1])).astype(float) / (t[i + 1] - t[i] + 1e-12)

        # Recursive Cox-de Boor formula
        for d in range(1, k + 1):
            for i in range(n):
                denom1 = t[i + d] - t[i]
                denom2 = t[i + d + 1] - t[i + 1]
                term1 = 0 if denom1 == 0 else (x - t[i]) * B[:, i] / denom1
                term2 = 0 if denom2 == 0 else (t[i + d + 1] - x) * B[:, i + 1] / denom2
                B[:, i] = d * (term1 + term2)
        return B

    def _i_spline_basis(self, x, t, k):
        """
        Compute I-spline basis (integrated M-spline).
        """
        B = self._m_spline_basis(x, t, k)
        dx = np.diff(x).mean()
        I = np.cumsum(B, axis=0) * dx
        I = I / np.max(I, axis=0, keepdims=True)  # Normalize to [0, 1]
        return I

    def fit(self, X, y=None):
        x = np.asarray(X).ravel()
        self.knots_ = self._make_knots(x)
        self.X_design_ = self._i_spline_basis(x, self.knots_, self.degree)
        return self

    def transform(self, X):
        x = np.asarray(X).ravel()
        return self._i_spline_basis(x, self.knots_, self.degree)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_penalty_matrix(self):
        """
        Compute roughness penalty: L2 penalty on second derivative of M-spline.
        I-spline basis are integrals of M-spline, so we penalize curvature of M-spline.
        """
        B = self._m_spline_basis(
            np.linspace(self.knots_[self.degree], self.knots_[-self.degree - 1], 200),
            self.knots_,
            self.degree
        )
        B_dd = np.gradient(np.gradient(B, axis=0), axis=0)
        P = B_dd.T @ B_dd
        return P / B.shape[0]
