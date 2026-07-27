import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...core.exceptions import InvalidParamError
from ...core.knots import bspline_basis
from ...core.params import UNSET
from .mixins import SplineBasisMixin

# ``bspline_basis`` used to be defined here and in ``tensor_product``; the two
# copies are now a single implementation in ``pretab.core.knots``, re-exported
# under the original name so existing imports keep working.
__all__ = ["PSplineTransformer", "bspline_basis"]


class PSplineTransformer(SplineBasisMixin, TransformerMixin, BaseEstimator):
    r"""
    P-spline Transformer for smooth spline basis expansion with penalization.

    This transformer expands each input feature into a set of B-spline basis
    functions and stores a corresponding difference-penalty matrix for
    regularization. It is useful in Generalized Additive Models (GAMs) where
    smoothness is enforced through penalties.

    Let :math:`p` be the ``degree`` and :math:`m := \mathtt{output\_dim}` the number
    of non-bias output columns per feature. A B-spline basis with :math:`K` interior
    knots (padded by :math:`p + 1` repeated boundary knots on each side) has

    .. math::

        m = \operatorname{len}(\mathtt{knots}) - p - 1 = K + p + 1,

    so the requested ``output_dim`` is inverted to :math:`K = \mathtt{output\_dim} - p - 1`
    interior knots. ``include_bias=True`` adds one further column on top of ``output_dim``.

    Parameters
    ----------
    output_dim : int, default=6
        Number of non-bias output columns per feature (:math:`m`). Must be at least
        ``degree + 1``. The number of interior knots is ``output_dim - degree - 1``.

    degree : int, default=3
        Degree of the B-spline basis functions (e.g., 3 for cubic splines).

    diff_order : int, default=2
        The order of the difference penalty used to compute the smoothness penalty matrix.
        For example, 2 corresponds to a second-order difference penalty (encouraging smooth second derivatives).

    include_bias : bool, default=False
        If True, prepend a constant intercept column per feature. The bias term is
        left unpenalized (a zero row/column is added to the penalty matrix).

    placement_strategy : {"uniform", "quantile"}, default="uniform"
        Interior-knot placement rule. ``"uniform"`` spaces knots evenly across the
        range; ``"quantile"`` places them at evenly spaced data quantiles.

        .. note::
           P-splines are penalized (difference-penalty) splines that assume
           **equally-spaced** knots, so this family is *unsupervised-only*:
           target-aware placement does not apply and only ``"uniform"`` /
           ``"quantile"`` spacing is accepted.

    adaptive : bool, default=False
        Retained for API parity but a no-op for this unsupervised-only family: the
        per-feature width is always fixed to ``output_dim``.

    min_output_dim : int or None, default=None
        Unused for this unsupervised-only family (kept for API parity).

    max_output_dim : int or None, default=None
        Unused for this unsupervised-only family (kept for API parity).

    Attributes
    ----------
    knots_ : list of ndarray
        Full padded knot sequence (interior knots bracketed by ``degree + 1``
        repeated boundary knots on each side) for each feature.

    n_knots_ : list of int
        Number of interior knots for each feature (``output_dim - degree - 1`` on
        the default, non-selector path).

    penalty_ : list of ndarray
        Difference-penalty matrices (``D^T D``) for each feature.

    n_basis_ : list of int
        Number of output columns per feature, including the optional bias (equals
        ``output_dim (+1 if include_bias)`` on the default path).

    n_features_in_ : int
        Number of input features seen during ``fit``.

    total_output_dim_ : int
        Total number of output columns across all features (fitted); equals
        ``n_features * (output_dim (+1 if include_bias))``.

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
    >>> transformer = PSplineTransformer(output_dim=8)
    >>> Xt = transformer.fit_transform(X)
    >>> Xt.shape
    (30, 8)
    >>> transformer.n_knots_
    [4]
    >>> transformer.total_output_dim_
    8
    """

    _feature_suffix_value = "ps"

    def __init__(
        self,
        output_dim=UNSET,
        degree=3,
        diff_order=2,
        include_bias=False,
        placement_strategy: str = "uniform",
        adaptive: bool = False,
        min_output_dim=UNSET,
        max_output_dim=UNSET,
    ):
        self.output_dim = output_dim
        self.degree = degree
        self.diff_order = diff_order
        self.include_bias = include_bias
        self.placement_strategy = placement_strategy
        self.adaptive = adaptive
        self.min_output_dim = min_output_dim
        self.max_output_dim = max_output_dim

    def fit(self, X, y=None):
        X = self._validate_allow_nan(X, reset=True)
        output_dim = self._resolve_param("output_dim", default=6)
        if self.placement_strategy not in ("uniform", "quantile"):
            raise InvalidParamError(
                f"Invalid placement_strategy. Choose 'uniform' or 'quantile'. Got {self.placement_strategy!r}."
            )
        strategy = self.placement_strategy

        if output_dim < self.degree + 1:
            raise InvalidParamError(
                f"output_dim must be >= degree + 1 = {self.degree + 1} for the p-spline basis, got {output_dim}"
            )

        self.knots_ = []
        self.penalty_ = []
        self.n_basis_ = []
        self.n_knots_ = []

        min_interior, max_interior = self._adaptive_interior_bounds(
            output_dim, None, floor=self.degree + 1, offset=self.degree + 1
        )

        for i in range(X.shape[1]):
            x = X[:, i]
            knots = self._place_bspline_knots(
                x, y, output_dim, self.degree, strategy, None, None, min_interior, max_interior
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
            self.n_knots_.append(max(0, len(knots) - 2 * (self.degree + 1)))
            self.penalty_.append(penalty)

        return self

    def transform(self, X):
        check_is_fitted(self, "n_basis_")
        X = self._validate_allow_nan(X, reset=False)

        all_basis = []
        for i in range(X.shape[1]):
            # Clip into the fitted range so values outside it evaluate on the
            # boundary rather than falling off every knot span and producing an
            # all-zero row. Matches BaseSplineTransformer.transform.
            knots = self.knots_[i]
            x = np.clip(X[:, i], knots[0], knots[-1])
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
