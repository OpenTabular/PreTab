import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class BSplineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_knots=6, degree=3, include_bias=False):
        self.n_knots = n_knots
        self.degree = degree
        self.include_bias = include_bias

    def _make_knots(self, x):
        x_min, x_max = np.min(x), np.max(x)
        inner_knots = np.linspace(x_min, x_max, self.n_knots)
        t = np.concatenate((
            np.repeat(inner_knots[0], self.degree),
            inner_knots,
            np.repeat(inner_knots[-1], self.degree)
        ))
        return t

    def _bspline_basis(self, x, t, k):
        n = len(t) - k - 1
        B = np.zeros((len(x), n))

        # Degree 0
        for i in range(n):
            B[:, i] = ((t[i] <= x) & (x < t[i+1])).astype(float)

        # Recursion
        for d in range(1, k+1):
            for i in range(n):
                denom1 = t[i+d] - t[i]
                denom2 = t[i+d+1] - t[i+1]
                term1 = 0 if denom1 == 0 else (x - t[i]) * B[:, i] / denom1
                term2 = 0 if denom2 == 0 else (t[i+d+1] - x) * B[:, i+1] / denom2
                B[:, i] = term1 + term2
        return B

    def fit(self, X, y=None):
        x = np.asarray(X).ravel()
        self.knots_ = self._make_knots(x)
        self.X_design_ = self._bspline_basis(x, self.knots_, self.degree)
        return self

    def transform(self, X):
        x = np.asarray(X).ravel()
        return self._bspline_basis(x, self.knots_, self.degree)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_penalty_matrix(self):
        B = self._bspline_basis(
            np.linspace(self.knots_[self.degree], self.knots_[-self.degree - 1], 200),
            self.knots_,
            self.degree
        )
        B_dd = np.gradient(np.gradient(B, axis=0), axis=0)
        return B_dd.T @ B_dd / B.shape[0]
