from typing import ClassVar

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...core.params import UNSET
from .mixins import SplineBasisMixin


def bspline_basis(x, knots, degree, i):
    if degree == 0:
        return np.where((x >= knots[i]) & (x < knots[i + 1]), 1.0, 0.0)
    else:
        denom1 = knots[i + degree] - knots[i]
        denom2 = knots[i + degree + 1] - knots[i + 1]

        term1 = 0.0 if denom1 == 0 else (x - knots[i]) / denom1 * bspline_basis(x, knots, degree - 1, i)
        term2 = (
            0.0 if denom2 == 0 else (knots[i + degree + 1] - x) / denom2 * bspline_basis(x, knots, degree - 1, i + 1)
        )

        return term1 + term2


class PSplineTransformer(SplineBasisMixin, TransformerMixin, BaseEstimator):
    """
    P-spline Transformer for smooth spline basis expansion with penalization.

    This transformer expands each input feature into a set of B-spline basis functions
    and stores a corresponding penalty matrix for regularization. It is useful in
    Generalized Additive Models (GAMs) where smoothness is enforced through penalties.

    Parameters
    ----------
    n_basis : int, default=20
        Number of interior knots to place across the range of each feature
        (canonical name; the legacy ``n_knots`` alias still works and emits a
        ``FutureWarning``).

    degree : int, default=3
        Degree of the B-spline basis functions (e.g., 3 for cubic splines).

    diff_order : int, default=2
        The order of the difference penalty used to compute the smoothness penalty matrix.
        For example, 2 corresponds to a second-order difference penalty (encouraging smooth second derivatives).

    include_bias : bool, default=False
        If True, prepend a constant intercept column per feature. The bias term is
        left unpenalized (a zero row/column is added to the penalty matrix).

    strategy : {"uniform", "quantile"}, default="uniform"
        Knot placement rule. ``"uniform"`` reproduces the historical evenly spaced
        knots; ``"quantile"`` places them at evenly spaced data quantiles.

    selector : BaseKnotSelector or None, default=None
        Optional target-aware knot selector (for example ``CARTKnotSelector``).
        When provided it determines the interior knots from the target and
        requires ``y`` during ``fit``.

    task : {"regression", "classification"} or None, default=None
        Task forwarded to a target-aware ``selector``.

    Attributes
    ----------
    knots_ : list of ndarray
        List of extended knot sequences (with added boundary knots) for each feature.

    penalty_ : list of ndarray
        List of penalty matrices (D^T D) for each feature, where D is the differencing matrix.

    n_basis_ : list of int
        Number of B-spline basis functions generated for each feature.

    n_features_in_ : int
        Number of input features seen during `fit`.

    Notes
    -----
    - Boundary knots are added automatically to ensure proper spline behavior near the edges.
    - Internally, this transformer uses recursive B-spline basis construction.
    - This implementation supports multi-dimensional inputs and stacks transformed features horizontally.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import PSplineTransformer
    >>> X = np.linspace(0, 1, 30).reshape(-1, 1)
    >>> transformer = PSplineTransformer(n_knots=6)
    >>> transformer.fit_transform(X).shape
    (30, 8)
    """

    _feature_suffix_value = "ps"
    _param_aliases: ClassVar[dict[str, str]] = {
        "n_knots": "n_basis",
        "knot_strategy": "strategy",
        "knot_selector": "selector",
    }

    def __init__(
        self,
        n_basis=UNSET,
        degree=3,
        diff_order=2,
        include_bias=False,
        strategy=UNSET,
        selector=UNSET,
        task=None,
        n_knots=UNSET,
        knot_strategy=UNSET,
        knot_selector=UNSET,
    ):
        self.n_basis = n_basis
        self.degree = degree
        self.diff_order = diff_order
        self.include_bias = include_bias
        self.strategy = strategy
        self.selector = selector
        self.task = task
        self.n_knots = n_knots
        self.knot_strategy = knot_strategy
        self.knot_selector = knot_selector

    def fit(self, X, y=None):
        X = self._validate_allow_nan(X, reset=True)
        n_basis = self._resolve_param("n_basis", default=20)
        strategy = self._resolve_param("strategy", default="uniform")
        selector = self._resolve_param("selector", default=None)

        self.knots_ = []
        self.penalty_ = []
        self.n_basis_ = []

        for i in range(X.shape[1]):
            x = X[:, i]
            inner_knots = self._place_spanning_knots(x, y, n_basis, strategy, selector, self.task)
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
            penalty = D.T @ D
            if self.include_bias:
                penalty = np.pad(penalty, ((1, 0), (1, 0)))
            self.knots_.append(knots)
            self.n_basis_.append(n_basis + (1 if self.include_bias else 0))
            self.penalty_.append(penalty)

        return self

    def transform(self, X):
        check_is_fitted(self, "n_basis_")
        X = self._validate_allow_nan(X, reset=False)

        all_basis = []
        for i in range(X.shape[1]):
            x = X[:, i]
            nb = len(self.knots_[i]) - self.degree - 1
            basis = np.zeros((len(x), nb))
            for j in range(nb):
                basis[:, j] = bspline_basis(x, self.knots_[i], self.degree, j)
            if self.include_bias:
                basis = np.hstack([np.ones((len(x), 1)), basis])
            all_basis.append(basis)

        return np.hstack(all_basis)

    def get_penalty_matrix(self, feature_index=0):
        """Return the difference penalty matrix for a fitted feature.

        Parameters
        ----------
        feature_index : int, default=0
            Index of the feature whose penalty matrix is returned.

        Returns
        -------
        P : ndarray of shape (n_basis, n_basis)
            Penalty matrix ``D.T @ D`` built from the finite-difference operator,
            suitable for Tikhonov-style smoothness regularization.
        """
        check_is_fitted(self, "penalty_")
        return self.penalty_[feature_index]
