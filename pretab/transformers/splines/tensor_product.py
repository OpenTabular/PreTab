from typing import ClassVar

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...core.params import UNSET
from .mixins import SplineBasisMixin


def bspline_basis(x, knots, degree, i):
    if degree == 0:
        return ((knots[i] <= x) & (x < knots[i + 1])).astype(float)
    else:
        denom1 = knots[i + degree] - knots[i]
        denom2 = knots[i + degree + 1] - knots[i + 1]
        term1 = 0.0 if denom1 == 0 else (x - knots[i]) / denom1 * bspline_basis(x, knots, degree - 1, i)
        term2 = (
            0.0 if denom2 == 0 else (knots[i + degree + 1] - x) / denom2 * bspline_basis(x, knots, degree - 1, i + 1)
        )
        return term1 + term2


class TensorProductSplineTransformer(SplineBasisMixin, TransformerMixin, BaseEstimator):
    """
    Tensor Product Spline Transformer for multivariate smooth basis expansion.

    This transformer generates tensor-product B-spline basis functions for multivariate input,
    allowing for smooth modeling of complex feature interactions. It supports regularization
    via difference penalties in each marginal dimension, suitable for additive models and
    structured regression.

    Parameters
    ----------
    n_knots : int, default=5
        Number of interior knots per input feature (per marginal dimension).

    degree : int, default=3
        Degree of the B-spline basis functions.

    diff_order : int, default=2
        Order of the finite difference penalty used to enforce smoothness along each input dimension.

    Attributes
    ----------
    dim_ : int
        Number of input features (marginal dimensions).

    knots_ : list of ndarray
        List of knot sequences for each marginal dimension.

    bases_ : list of ndarray
        List of B-spline basis matrices (n_samples x n_basis) for each input feature.

    penalties_ : list of ndarray
        List of univariate penalty matrices for each marginal basis.

    X_design_ : ndarray of shape (n_samples, n_total_basis)
        Full tensor-product design matrix, computed during `fit`.

    n_features_in_ : int
        Number of input features seen during `fit`.

    Notes
    -----
    - Uses einsum-based reshaping to build the tensor product basis efficiently.
    - Supports arbitrary number of input dimensions.
    - Commonly used in structured additive models and GAMs where smooth surfaces are desired.

    References
    ----------
    .. [1] Eilers, P.H.C. and Marx, B.D. (2003). "Multivariate calibration with
       temperature interaction using two-dimensional penalized signal regression".
    .. [2] Wood, S.N. (2017). "Generalized Additive Models: An Introduction with R".

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import TensorProductSplineTransformer
    >>> X = np.random.rand(20, 2)
    >>> transformer = TensorProductSplineTransformer(n_knots=4)
    >>> transformer.fit_transform(X).shape
    (20, 36)
    """

    _param_aliases: ClassVar[dict[str, str]] = {"n_knots": "n_basis"}

    def __init__(self, n_basis=UNSET, degree=3, diff_order=2, n_knots=UNSET):
        self.n_basis = n_basis
        self.degree = degree
        self.diff_order = diff_order
        self.n_knots = n_knots

    def _make_knots(self, x, n_basis):
        xmin, xmax = np.min(x), np.max(x)
        inner = np.linspace(xmin, xmax, n_basis)
        return np.concatenate((np.repeat(inner[0], self.degree), inner, np.repeat(inner[-1], self.degree)))

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
        X = self._validate_allow_nan(X, reset=True)
        n_basis = self._resolve_param("n_basis", default=5)

        self.dim_ = X.shape[1]
        self.knots_ = []
        self.bases_ = []
        self.penalties_ = []

        for d in range(self.dim_):
            knots = self._make_knots(X[:, d], n_basis)
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
        check_is_fitted(self, "X_design_")
        X = self._validate_allow_nan(X, reset=False)

        bases = []
        for d in range(self.dim_):
            basis = self._basis_matrix(X[:, d], self.knots_[d])
            bases.append(basis)

        n_samples = X.shape[0]
        design = bases[0]
        for b in bases[1:]:
            design = np.einsum("ni,nj->nij", design, b).reshape(n_samples, -1)
        return design

    def get_feature_names_out(self, input_features=None):
        """Return names for the interaction basis as ``tp_{feat0 i}_{feat1 j}...``."""
        check_is_fitted(self, "bases_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        sizes = [b.shape[1] for b in self.bases_]
        names = []
        for multi_index in np.ndindex(*sizes):
            parts = [f"{input_features[d]}{multi_index[d]}" for d in range(self.dim_)]
            names.append("tp_" + "_".join(parts))
        return np.asarray(names, dtype=object)

    def get_penalty_matrices(self):
        """Return the Kronecker-structured penalty matrices.

        Returns
        -------
        penalties : list of ndarray
            One full penalty matrix per marginal direction, each formed as a
            Kronecker product of a marginal difference penalty with identity
            matrices for the remaining dimensions.
        """
        kron_penalties = []
        for i, Si in enumerate(self.penalties_):
            mats = [np.eye(b.shape[1]) for j, b in enumerate(self.bases_) if j != i]
            P = Si
            for M in mats:
                P = np.kron(P, M)
            kron_penalties.append(P)
        return kron_penalties
