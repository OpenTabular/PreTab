import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def bspline_basis(x, knots, degree, i):
    """
    Cox–de Boor recursion for B-spline basis function B_{i,degree}(x)
    """
    if degree == 0:
        return np.where((x >= knots[i]) & (x < knots[i + 1]), 1.0, 0.0)
    else:
        denom1 = knots[i + degree] - knots[i]
        denom2 = knots[i + degree + 1] - knots[i + 1]

        term1 = 0.0 if denom1 == 0 else (x - knots[i]) / denom1 * bspline_basis(x, knots, degree - 1, i)
        term2 = 0.0 if denom2 == 0 else (knots[i + degree + 1] - x) / denom2 * bspline_basis(x, knots, degree - 1, i + 1)

        return term1 + term2


class PSplineTransformer(BaseEstimator, TransformerMixin):
    """
    Penalized B-spline (P-spline) transformer for 1D inputs with cubic basis and difference penalty.
    """

    def __init__(self, n_knots=20, degree=3, diff_order=2):
        self.n_knots = n_knots
        self.degree = degree
        self.diff_order = diff_order

    def fit(self, X, y=None):
        x = np.asarray(X).reshape(-1)
        xmin, xmax = x.min(), x.max()

        # Extend knots for B-spline of given degree
        inner_knots = np.linspace(xmin, xmax, self.n_knots)
        self.knots_ = np.concatenate((
            np.repeat(inner_knots[0], self.degree),
            inner_knots,
            np.repeat(inner_knots[-1], self.degree)
        ))
        self.n_basis_ = len(self.knots_) - self.degree - 1
        self.xmin_, self.xmax_ = xmin, xmax

        # Build penalty matrix (difference matrix on coefficients)
        D = np.eye(self.n_basis_)
        for _ in range(self.diff_order):
            D = np.diff(D, n=1, axis=0)
        self.penalty_ = D.T @ D

        return self

    def transform(self, X):
        x = np.asarray(X).reshape(-1)
        X_basis = np.zeros((len(x), self.n_basis_))
        for i in range(self.n_basis_):
            X_basis[:, i] = bspline_basis(x, self.knots_, self.degree, i)
        return X_basis

    def get_penalty_matrix(self):
        return self.penalty_
