import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


def bspline_basis(x, knots, degree, i):
    if degree == 0:
        return np.where((x >= knots[i]) & (x < knots[i + 1]), 1.0, 0.0)
    else:
        denom1 = knots[i + degree] - knots[i]
        denom2 = knots[i + degree + 1] - knots[i + 1]

        term1 = (
            0.0
            if denom1 == 0
            else (x - knots[i]) / denom1 * bspline_basis(x, knots, degree - 1, i)
        )
        term2 = (
            0.0
            if denom2 == 0
            else (knots[i + degree + 1] - x)
            / denom2
            * bspline_basis(x, knots, degree - 1, i + 1)
        )

        return term1 + term2


class PSplineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_knots=20, degree=3, diff_order=2):
        self.n_knots = n_knots
        self.degree = degree
        self.diff_order = diff_order

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
        self.penalty_ = []
        self.n_basis_ = []

        for i in range(X.shape[1]):
            x = X[:, i]
            xmin, xmax = x.min(), x.max()
            inner_knots = np.linspace(xmin, xmax, self.n_knots)
            knots = np.concatenate(
                (
                    np.repeat(inner_knots[0], self.degree),
                    inner_knots,
                    np.repeat(inner_knots[-1], self.degree),
                )
            )
            n_basis = len(knots) - self.degree - 1
            D = np.eye(n_basis)
            for _ in range(self.diff_order):
                D = np.diff(D, n=1, axis=0)
            self.knots_.append(knots)
            self.n_basis_.append(n_basis)
            self.penalty_.append(D.T @ D)

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

        all_basis = []
        for i in range(X.shape[1]):
            x = X[:, i]
            basis = np.zeros((len(x), self.n_basis_[i]))
            for j in range(self.n_basis_[i]):
                basis[:, j] = bspline_basis(x, self.knots_[i], self.degree, j)
            all_basis.append(basis)

        return np.hstack(all_basis)

    def get_penalty_matrix(self, feature_index=0):
        return self.penalty_[feature_index]
