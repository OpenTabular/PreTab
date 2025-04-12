import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from itertools import product


def bspline_basis(x, knots, degree, i):
    if degree == 0:
        return ((knots[i] <= x) & (x < knots[i + 1])).astype(float)
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


class TensorProductSplineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_knots=5, degree=3, diff_order=2):
        self.n_knots = n_knots
        self.degree = degree
        self.diff_order = diff_order

    def _make_knots(self, x):
        xmin, xmax = np.min(x), np.max(x)
        inner = np.linspace(xmin, xmax, self.n_knots)
        return np.concatenate(
            (np.repeat(inner[0], self.degree), inner, np.repeat(inner[-1], self.degree))
        )

    def _basis_matrix(self, x, knots):
        n_basis = len(knots) - self.degree - 1
        B = np.zeros((len(x), n_basis))
        for i in range(n_basis):
            B[:, i] = bspline_basis(x, knots, self.degree, i)
        return B

    def _difference_penalty(self, n_basis):
        D = np.eye(n_basis)
        for _ in range(self.diff_order):
            D = np.diff(D, n=1, axis=0)
        return D.T @ D

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

        self.dim_ = X.shape[1]
        self.knots_ = []
        self.bases_ = []
        self.penalties_ = []

        for d in range(self.dim_):
            knots = self._make_knots(X[:, d])
            basis = self._basis_matrix(X[:, d], knots)
            penalty = self._difference_penalty(basis.shape[1])
            self.knots_.append(knots)
            self.bases_.append(basis)
            self.penalties_.append(penalty)

        n_samples = X.shape[0]
        design = self.bases_[0]
        for b in self.bases_[1:]:
            design = np.einsum("ni,nj->nij", design, b).reshape(n_samples, -1)
        self.X_design_ = design

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

        bases = []
        for d in range(self.dim_):
            basis = self._basis_matrix(X[:, d], self.knots_[d])
            bases.append(basis)

        n_samples = X.shape[0]
        design = bases[0]
        for b in bases[1:]:
            design = np.einsum("ni,nj->nij", design, b).reshape(n_samples, -1)
        return design

    def get_penalty_matrices(self):
        kron_penalties = []
        for i, Si in enumerate(self.penalties_):
            mats = [np.eye(b.shape[1]) for j, b in enumerate(self.bases_) if j != i]
            P = Si
            for M in mats:
                P = np.kron(P, M)
            kron_penalties.append(P)
        return kron_penalties
