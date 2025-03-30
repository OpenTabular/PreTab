import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import product

class TensorProductSplineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_knots=5, degree=3):
        self.n_knots = n_knots
        self.degree = degree

    def _make_knots(self, x):
        x_min, x_max = np.min(x), np.max(x)
        inner = np.linspace(x_min, x_max, self.n_knots)
        t = np.concatenate((
            np.repeat(inner[0], self.degree),
            inner,
            np.repeat(inner[-1], self.degree)
        ))
        return t

    def _bspline_basis(self, x, t, k):
        n = len(t) - k - 1
        B = np.zeros((len(x), n))
        for i in range(n):
            B[:, i] = ((t[i] <= x) & (x < t[i+1])).astype(float)
        for d in range(1, k+1):
            for i in range(n):
                denom1 = t[i+d] - t[i]
                denom2 = t[i+d+1] - t[i+1]
                term1 = 0 if denom1 == 0 else (x - t[i]) * B[:, i] / denom1
                term2 = 0 if denom2 == 0 else (t[i+d+1] - x) * B[:, i+1] / denom2
                B[:, i] = term1 + term2
        return B

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.knots_ = [self._make_knots(X[:, i]) for i in range(X.shape[1])]
        self.bases_ = [self._bspline_basis(X[:, i], self.knots_[i], self.degree) for i in range(X.shape[1])]
        # Compute tensor product basis
        grid = list(product(*[range(b.shape[1]) for b in self.bases_]))
        self.grid_ = grid
        self.X_design_ = np.prod([self.bases_[d][:, [g[d]]] for d in range(X.shape[1]) for g in grid], axis=0).reshape(len(X), -1)
        return self

    def transform(self, X):
        X = np.asarray(X)
        bases = [self._bspline_basis(X[:, i], self.knots_[i], self.degree) for i in range(X.shape[1])]
        return np.prod([bases[d][:, [g[d]]] for d in range(X.shape[1]) for g in self.grid_], axis=0).reshape(len(X), -1)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_penalty_matrix(self):
        B = self.X_design_
        B_dd = np.gradient(np.gradient(B, axis=0), axis=0)
        return B_dd.T @ B_dd / B.shape[0]
