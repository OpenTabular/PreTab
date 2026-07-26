import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...core.parameters import UNSET, validate_placement
from ...exceptions import InvalidParamError
from .knot_selectors import build_knot_selector
from .mixins import SplineBasisMixin


class CubicSplineTransformer(SplineBasisMixin, TransformerMixin, BaseEstimator):
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
        ``output_dim``.

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

    designs_ : list of ndarray
        Cached design matrices (spline basis evaluations) for each input feature,
        each of shape ``(n_samples, output_dim (+1 if include_bias))``.

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
    >>> from pretab.transformers import CubicSplineTransformer
    >>> X = np.linspace(0, 1, 20).reshape(-1, 1)
    >>> transformer = CubicSplineTransformer(output_dim=8)
    >>> Xt = transformer.fit_transform(X)
    >>> Xt.shape
    (20, 8)
    >>> transformer.n_knots_
    [5]
    >>> transformer.total_output_dim_
    8
    """

    _feature_suffix_value = "cs"

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

        X = [np.ones((n_samples, 1))] if self.include_bias else []
        X.append(x)
        X.append(x**2)
        X.append(x**3)

        for knot in knots:
            X.append(np.maximum(0, (x - knot)) ** 3)

        return np.hstack(X)

    def fit(self, X, y=None):
        validate_placement(self.target_aware, self.placement_strategy)
        X = self._validate_allow_nan(X, reset=True)
        output_dim = self._resolve_param("output_dim", default=6)

        if output_dim < 3:
            raise InvalidParamError(f"output_dim must be >= 3 for the cubic spline basis, got {output_dim}")

        n_interior = output_dim - 3

        if self.target_aware:
            selector = build_knot_selector(
                self.placement_strategy,
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
        self.designs_ = []
        for i in range(X.shape[1]):
            xi = X[:, i]
            knots = self._place_interior_knots(
                xi, y, n_interior, strategy, selector, self.task, min_interior, max_interior
            )
            self.knots_.append(knots)
            self.designs_.append(self._bspline_basis(xi, knots))

        self.n_basis_ = [design.shape[1] for design in self.designs_]
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
        check_is_fitted(self, "designs_")
        n_basis = self.designs_[feature_index].shape[1]
        P = np.zeros((n_basis, n_basis))
        offset = 4 if self.include_bias else 3
        for i in range(offset, n_basis):
            for j in range(offset, n_basis):
                ki = self.knots_[feature_index][i - offset]
                kj = self.knots_[feature_index][j - offset]
                P[i, j] = self._spline_penalty_entry(ki, kj, self.knots_[feature_index])
        return P

    def _spline_penalty_entry(self, ki, kj, knots):
        kmax = max(ki, kj)
        upper = knots[-1]
        x_vals = np.linspace(kmax, upper, 100)
        integrand = 36 * (x_vals - ki) * (x_vals - kj)
        return np.trapezoid(integrand, x_vals)
