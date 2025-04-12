import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


class ISplineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_knots=5, degree=3, include_bias=False):
        self.n_knots = n_knots
        self.degree = degree
        self.include_bias = include_bias

    def _make_knots(self, x):
        x_min, x_max = np.min(x), np.max(x)
        inner_knots = np.linspace(x_min, x_max, self.n_knots)
        t = np.concatenate(
            (
                np.repeat(inner_knots[0], self.degree + 1),
                inner_knots,
                np.repeat(inner_knots[-1], self.degree + 1),
            )
        )
        return t

    def _m_spline_basis(self, x, t, k):
        x = np.atleast_1d(x)
        n = len(t) - k - 1
        B = np.zeros((len(x), n))

        for i in range(n):
            B[:, i] = ((t[i] <= x) & (x < t[i + 1])).astype(float) / (
                t[i + 1] - t[i] + 1e-12
            )

        for d in range(1, k + 1):
            for i in range(n):
                denom1 = t[i + d] - t[i]
                denom2 = t[i + d + 1] - t[i + 1]
                term1 = 0 if denom1 == 0 else (x - t[i]) * B[:, i] / denom1
                term2 = 0 if denom2 == 0 else (t[i + d + 1] - x) * B[:, i + 1] / denom2
                B[:, i] = d * (term1 + term2)
        return B

    def _i_spline_basis(self, x, t, k):
        B = self._m_spline_basis(x, t, k)
        dx = np.diff(x).mean()
        I = np.cumsum(B, axis=0) * dx
        I = I / np.maximum(np.max(I, axis=0, keepdims=True), 1e-12)
        return I

    def fit(self, X, y=None):
        original_dim = np.shape(X)[1] if np.ndim(X) == 2 else 1
        X = check_array(X, dtype=np.float64, ensure_2d=True, ensure_all_finite=True)
        if X.shape[1] < original_dim:
            warnings.warn(
                "Some input features were dropped during check_array validation.",
                UserWarning,
            )

        self.knots_ = []
        self.designs_ = []

        for i in range(X.shape[1]):
            xi = X[:, i]
            knots = self._make_knots(xi)
            self.knots_.append(knots)
            self.designs_.append(self._i_spline_basis(xi, knots, self.degree))

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        original_dim = np.shape(X)[1] if np.ndim(X) == 2 else 1
        X = check_array(X, dtype=np.float64, ensure_2d=True, ensure_all_finite=True)
        if X.shape[1] < original_dim:
            warnings.warn(
                "Some input features were dropped during check_array validation.",
                UserWarning,
            )

        transformed = []
        for i in range(X.shape[1]):
            xi = X[:, i]
            I = self._i_spline_basis(xi, self.knots_[i], self.degree)
            transformed.append(I)

        return np.hstack(transformed)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_penalty_matrix(self, feature_index=0):
        knots = self.knots_[feature_index]
        x_vals = np.linspace(knots[self.degree], knots[-self.degree - 1], 200)
        B = self._m_spline_basis(x_vals, knots, self.degree)
        B_dd = np.gradient(np.gradient(B, axis=0), axis=0)
        P = B_dd.T @ B_dd
        return P / B.shape[0]
