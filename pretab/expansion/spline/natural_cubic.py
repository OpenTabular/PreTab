import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...core.parameters import UNSET, validate_placement
from ...core.supervised import warn_target_leakage
from ...exceptions import InvalidParamError
from ...placement.adapters import SplinePlacementAdapter
from .mixins import SplineBasisMixin


class NaturalCubicSplineTransformer(SplineBasisMixin, TransformerMixin, BaseEstimator):
    r"""
    Natural Cubic Spline Transformer for continuous features.

    This transformer expands each input feature using a natural cubic spline basis.
    Natural cubic splines are piecewise cubic polynomials that are linear beyond the
    boundary knots, ensuring smooth extrapolation. The basis stacks a linear term
    with one constrained non-linear term per interior knot (and, optionally, a bias
    column).

    Let :math:`m := \mathtt{output\_dim}` be the number of non-bias output columns
    per feature and :math:`T` the total number of (spanning) knots including the two
    boundary knots. The natural-spline basis drops the intercept, so

    .. math::

        m = T - 1,

    and the requested ``output_dim`` is inverted to :math:`T = \mathtt{output\_dim} + 1`
    spanning knots (both endpoints retained for the linear-tail constraint).
    ``include_bias=True`` adds one further column on top of ``output_dim``.

    Parameters
    ----------
    output_dim : int, default=6
        Number of non-bias output columns per feature (:math:`m`). Must be at least
        2. The number of spanning knots placed is ``output_dim + 1``.

    degree : int, default=3
        Degree of the spline. Natural cubic splines are cubic by definition, so
        this is fixed at 3 and included only for API parity with the other splines.

    include_bias : bool, default=False
        If True, includes a constant bias (intercept) column in the output.

    target_aware : bool, default=False
        If True, knots are placed by a target-aware selector built from
        ``placement_strategy`` (requires ``y`` during ``fit``). If False, knots use
        the unsupervised ``placement_strategy`` spacing.

    placement_strategy : {"cart", "lightgbm", "uniform", "quantile"}, default="uniform"
        When ``target_aware=True``, the selector: ``"cart"`` or ``"lightgbm"``.
        When ``target_aware=False``, the spacing: ``"uniform"`` reproduces the
        historical evenly spaced knots, ``"quantile"`` places them at evenly spaced
        data quantiles.

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
        Spanning knot vectors used for each feature (length ``output_dim + 1`` on
        the default, non-selector path).

    n_knots_ : list of int
        Number of interior knots for each feature (``len(knots_[i]) - 2``).

    n_basis_ : list of int
        Number of output columns per feature, including the optional bias.

    n_features_in_ : int
        Number of input features seen during ``fit``.

    total_output_dim_ : int
        Total number of output columns across all features (fitted); equals
        ``n_features * (output_dim (+1 if include_bias))``.

    Notes
    -----
    The basis is constructed to satisfy the natural spline constraint: the second
    derivative of the spline is zero at the boundary knots. This reduces overfitting
    at the boundaries and improves extrapolation. Each feature is transformed
    independently and the outputs are concatenated.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import NaturalCubicSplineTransformer
    >>> X = np.linspace(0, 1, 20).reshape(-1, 1)
    >>> transformer = NaturalCubicSplineTransformer(output_dim=4)
    >>> Xt = transformer.fit_transform(X)
    >>> Xt.shape
    (20, 4)
    >>> transformer.n_knots_
    [3]
    >>> transformer.total_output_dim_
    4
    """

    _feature_suffix_value = "ncs"
    _representation_family = "naturalspline"
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

    def _basis(self, x, knots):
        x = np.asarray(x).reshape(-1, 1)
        K = knots
        n_samples = x.shape[0]
        n_knots = len(K)

        basis: list[np.ndarray] = [np.ones((n_samples, 1))] if self.include_bias else []
        basis.append(x)

        def omega(z, k):
            return np.maximum(0, z - k) ** 3

        def d(k):
            return omega(x, k) - omega(x, K[-1])

        denom = K[-1] - K[0]
        D = np.array([d(k) - ((K[-1] - k) / denom) * d(K[0]) - ((k - K[0]) / denom) * d(K[-1]) for k in K[1:-1]])
        basis.extend(list(D))
        return np.hstack(basis)

    def fit(self, X, y=None):
        warn_target_leakage(self, y)
        validate_placement(self.target_aware, self.placement_strategy)
        X = self._validate_allow_nan(X, reset=True)
        output_dim = self._resolve_param("output_dim", default=6)

        if output_dim < 2:
            raise InvalidParamError(f"output_dim must be >= 2 for the natural cubic spline basis, got {output_dim}")

        n_spanning = output_dim + 1

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

        min_interior, max_interior = self._adaptive_interior_bounds(output_dim, selector, floor=2, offset=1)

        self.knots_ = []
        self.n_basis_ = []

        for i in range(X.shape[1]):
            xi = X[:, i]
            knots = self._place_spanning_knots(
                xi, y, n_spanning, strategy, selector, self.task, min_interior, max_interior
            )
            self.knots_.append(knots)
            self.n_basis_.append(self._basis(xi, knots).shape[1])

        self.n_knots_ = [max(0, len(knots) - 2) for knots in self.knots_]
        return self

    def transform(self, X):
        check_is_fitted(self, "n_basis_")
        X = self._validate_allow_nan(X, reset=False)

        transformed = []
        for i in range(X.shape[1]):
            xi = X[:, i]
            basis = self._basis(xi, self.knots_[i])
            transformed.append(basis)

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
            Penalty matrix approximating the integrated squared second derivative
            of the natural cubic spline basis.
        """
        check_is_fitted(self, "knots_")
        knots = self.knots_[feature_index]
        x_grid = np.linspace(knots[0], knots[-1], 200)
        B = self._basis(x_grid, knots)
        B_d = np.gradient(B, x_grid, axis=0)
        B_dd = np.gradient(B_d, x_grid, axis=0)

        n_basis = B.shape[1]
        P = np.zeros((n_basis, n_basis))
        offset = 2 if self.include_bias else 1

        for i in range(offset, n_basis):
            for j in range(offset, n_basis):
                integrand = B_dd[:, i] * B_dd[:, j]
                P[i, j] = np.trapezoid(integrand, x_grid)

        return P
