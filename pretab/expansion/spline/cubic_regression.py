import itertools

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...core.parameters import UNSET, validate_placement
from ...core.supervised import warn_target_leakage
from ...exceptions import InvalidParamError
from ...placement.adapters import SplinePlacementAdapter
from .mixins import SplineBasisMixin


class CubicRegressionSplineTransformer(SplineBasisMixin, TransformerMixin, BaseEstimator):
    r"""
    Cubic Spline Transformer for one-dimensional or multi-dimensional input features.

    This transformer applies a truncated-power cubic spline basis to each
    continuous feature. The basis stacks the polynomial terms :math:`x, x^2, x^3`
    with one truncated cubic term :math:`(x - \kappa)^3_+` per interior knot
    :math:`\kappa`. Optionally a bias (intercept) column is prepended.

    Let :math:`m := \mathtt{output\_dim}` be the number of non-bias output columns
    produced per feature. A cubic truncated-power basis with :math:`K` interior
    knots has

    .. math::

        m = 3 + K,

    so the requested ``output_dim`` is inverted to :math:`K = \mathtt{output\_dim} - 3`
    interior knots. ``include_bias=True`` adds one further column on top of ``output_dim``.

    Parameters
    ----------
    output_dim : int, default=6
        Number of non-bias output columns per feature (:math:`m`). Must be at
        least 3 (the three polynomial terms; ``output_dim == 3`` places no interior
        knots). The number of interior knots is ``output_dim - 3``.

    degree : int, default=3
        Degree of the polynomial spline. Currently fixed to 3 (cubic), included for compatibility.

    include_bias : bool, default=False
        Whether to include a bias (intercept) term in the output feature set.

    target_aware : bool, default=False
        If True, interior knots are placed by a target-aware selector built from
        ``placement_strategy`` (requires ``y`` during ``fit``). If False, knots use
        the unsupervised ``placement_strategy`` spacing.

    placement_strategy : {"cart", "lightgbm", "uniform", "quantile"}, default="uniform"
        When ``target_aware=True``, the selector: ``"cart"`` or ``"lightgbm"``.
        When ``target_aware=False``, the spacing: ``"uniform"`` spaces knots evenly
        across the range, ``"quantile"`` places them at evenly spaced data quantiles.

    task : {"regression", "classification"} or None, default=None
        Task forwarded to the target-aware selector when ``target_aware=True``.

    adaptive : bool, default=False
        If True (with ``target_aware=True``), the per-feature output dimension may
        vary within ``[min_output_dim, max_output_dim]`` instead of being fixed to
        ``output_dim``. When both bounds are set, ``output_dim`` is ignored entirely.

    min_output_dim : int or None, default=None
        Lower bound on the per-feature output dimension in adaptive mode.

    max_output_dim : int or None, default=None
        Upper bound on the per-feature output dimension in adaptive mode.

    random_state : int or None, default=None
        Random state forwarded to the target-aware selector for reproducibility.

    Attributes
    ----------
    knots_ : list of ndarray
        Interior knots used for each feature (length ``output_dim - 3`` on the
        default, non-selector path).

    n_knots_ : list of int
        Number of interior knots placed for each feature (``len(knots_[i])``).

    n_basis_ : list of int
        Number of output columns per feature, including the optional bias.

    n_features_in_ : int
        Number of input features seen during ``fit``.

    total_output_dim_ : int
        Total number of output columns across all features (fitted); equals
        ``n_features * (output_dim (+1 if include_bias))``.

    Notes
    -----
    The basis includes the polynomial terms ``x, x^2, x^3`` followed by the
    truncated power terms ``(x - knot)^3_+`` for each interior knot. Each feature
    is expanded independently and the results are stacked horizontally.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import CubicRegressionSplineTransformer
    >>> X = np.linspace(0, 1, 20).reshape(-1, 1)
    >>> transformer = CubicRegressionSplineTransformer(output_dim=8)
    >>> Xt = transformer.fit_transform(X)
    >>> Xt.shape
    (20, 8)
    >>> transformer.n_knots_
    [5]
    >>> transformer.total_output_dim_
    8
    """

    _feature_suffix_value = "cs"
    _representation_family = "cubicspline"
    _representation_supervision = "optional"
    _representation_local_support = True

    def __init__(
        self,
        output_dim=UNSET,
        degree=3,
        include_bias=False,
        target_aware: bool = False,
        placement_strategy: str = "uniform",
        task=None,
        adaptive: bool = False,
        min_output_dim=UNSET,
        max_output_dim=UNSET,
        random_state: int | None = None,
    ):
        self.output_dim = output_dim
        self.degree = degree
        self.include_bias = include_bias
        self.target_aware = target_aware
        self.placement_strategy = placement_strategy
        self.task = task
        self.adaptive = adaptive
        self.min_output_dim = min_output_dim
        self.max_output_dim = max_output_dim
        self.random_state = random_state

    def _bspline_basis(self, x, knots):
        x = np.asarray(x).reshape(-1, 1)
        n_samples = x.shape[0]

        X: list[np.ndarray] = [np.ones((n_samples, 1))] if self.include_bias else []
        X.append(x)
        X.append(x**2)
        X.append(x**3)

        for knot in knots:
            X.append(np.maximum(0, (x - knot)) ** 3)

        return np.hstack(X)

    def fit(self, X, y=None):
        warn_target_leakage(self, y)
        validate_placement(self.target_aware, self.placement_strategy)
        X = self._validate_allow_nan(X, reset=True)
        output_dim = self._resolve_param("output_dim", default=6)

        if output_dim < 3:
            raise InvalidParamError(f"output_dim must be >= 3 for the cubic spline basis, got {output_dim}")

        n_interior = output_dim - 3

        if self.target_aware:
            selector = SplinePlacementAdapter(
                placement_strategy=self.placement_strategy,
                degree=self.degree,
                spline_type="bspline",
                random_state=self.random_state,
            )
            strategy = "uniform"
        else:
            selector = None
            strategy = self.placement_strategy

        min_interior, max_interior = self._adaptive_interior_bounds(output_dim, selector, floor=3, offset=3)

        self.knots_ = []
        self.n_basis_ = []
        self.x_min_ = []
        self.x_max_ = []
        for i in range(X.shape[1]):
            xi = X[:, i]
            knots = self._place_interior_knots(
                xi, y, n_interior, strategy, selector, self.task, min_interior, max_interior
            )
            self.knots_.append(knots)
            self.n_basis_.append(self._bspline_basis(xi, knots).shape[1])
            finite_xi, _ = self._finite_column(xi, None)
            self.x_min_.append(float(finite_xi.min()))
            self.x_max_.append(float(finite_xi.max()))

        self.n_knots_ = [len(knots) for knots in self.knots_]
        return self

    def transform(self, X):
        check_is_fitted(self, "n_basis_")
        X = self._validate_allow_nan(X, reset=False)

        transformed = []
        for i in range(X.shape[1]):
            xi = X[:, i]
            design = self._bspline_basis(xi, self.knots_[i])
            transformed.append(design)

        return np.hstack(transformed)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_penalty_matrix(self, feature_index=0):
        """Return the curvature penalty matrix for a fitted feature.

        Penalizes the integrated squared second derivative of every basis
        column over the fitted feature range, including the polynomial
        ``x**2`` / ``x**3`` columns (only ``x`` and the bias have an
        identically-zero second derivative and so are unpenalized).

        Parameters
        ----------
        feature_index : int, default=0
            Index of the feature whose penalty matrix is returned.

        Returns
        -------
        P : ndarray of shape (n_basis, n_basis)
            Penalty matrix that penalizes the second derivative (curvature) of
            the spline basis for smoothness.
        """
        check_is_fitted(self, "n_basis_")
        n_basis = self.n_basis_[feature_index]
        knots = self.knots_[feature_index]
        x_min = self.x_min_[feature_index]
        x_max = self.x_max_[feature_index]

        # Every basis column's second derivative is piecewise linear in x, with
        # kinks only at the knots, so a single-panel Simpson's rule per
        # knot-delimited segment integrates every pairwise product exactly.
        breakpoints = np.unique(np.concatenate(([x_min, x_max], np.clip(knots, x_min, x_max))))
        breakpoints.sort()

        P = np.zeros((n_basis, n_basis))
        for lo, hi in itertools.pairwise(breakpoints):
            if hi <= lo:
                continue
            xs = np.array([lo, 0.5 * (lo + hi), hi])
            weights = (hi - lo) / 6.0 * np.array([1.0, 4.0, 1.0])
            D = np.column_stack([self._second_derivative(xs, i, knots) for i in range(n_basis)])
            P += (D * weights[:, None]).T @ D
        return P

    def _second_derivative(self, x, basis_index, knots):
        """Second derivative of one basis column, evaluated at ``x``."""
        poly_offset = 1 if self.include_bias else 0
        col = basis_index - poly_offset
        if col <= 0:
            return np.zeros_like(x, dtype=float)  # bias / x: identically zero
        if col == 1:
            return np.full_like(x, 2.0, dtype=float)  # x**2
        if col == 2:
            return 6.0 * x  # x**3
        knot = knots[col - 3]
        return 6.0 * np.maximum(x - knot, 0.0)
