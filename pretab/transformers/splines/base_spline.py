"""
Shared base class for spline basis expansions (B-spline, M-spline, I-spline).

The base class handles the parts that are common to every spline family:

- resolving the number of basis functions and the derived number of internal knots,
- placing knots per feature using one of three strategies with a fixed priority
  (``knot_locations`` > target-aware ``placement_strategy`` > automatic strategy),
- looping over columns so that multi-column input is supported,
- assembling the design matrix, optional bias column, penalty matrix and feature
  names.

Each concrete transformer only implements how a single column is turned into a
basis matrix through the ``_design_matrix`` hook.
"""

from typing import ClassVar, Literal

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ...core.base import BasePreTabTransformer
from ...core.exceptions import (
    IncompatibleParamsError,
    InvalidParamError,
    PretabDataError,
)
from ...core.knots import (
    basis_to_knots,
    generate_internal_knots,
    quantile_knots,
    select_knots,
    uniform_knots,
)
from ...core.params import UNSET, validate_placement
from .knot_selectors import BaseKnotSelector, build_knot_selector


class BaseSplineTransformer(BasePreTabTransformer):
    """
    Base class for spline basis expansions with target-aware knot placement.

    Let ``p`` be the spline ``degree`` and ``m := output_dim`` the number of
    non-bias output columns produced per input feature. A B/M/I spline basis with
    ``K`` interior knots has ``m = K + p + 1`` columns, so the requested
    ``output_dim`` is inverted to ``K = output_dim - p - 1`` interior knots
    (bracketed by ``p + 1`` repeated boundary knots on each side). ``include_bias``
    adds one further intercept column on top of ``output_dim``.

    Parameters
    ----------
    output_dim : int, default=6
        Number of non-bias output columns per feature (``m``). The number of
        interior knots is derived as ``output_dim - degree - 1``. Must be at least
        ``degree + 1`` and at most 50. Ignored when ``knot_locations`` is provided.

    degree : int, default=3
        Degree of the spline (3 is cubic, 2 quadratic, 1 linear).

    target_aware : bool, default=False
        If True, knots are placed by a target-aware selector built from
        ``placement_strategy`` (requires ``y`` during fit). If False, knots are
        placed by the unsupervised ``placement_strategy`` spacing.

    placement_strategy : {"cart", "lightgbm", "uniform", "quantile"}, default="quantile"
        When ``target_aware=True``, the selector: ``"cart"`` or ``"lightgbm"``.
        When ``target_aware=False``, the spacing rule: ``"uniform"`` spaces knots
        evenly across the range, ``"quantile"`` places them at data quantiles.

    include_bias : bool, default=False
        Whether to prepend a constant intercept column per feature.

    knot_locations : ndarray or None, default=None
        Explicit internal knot locations applied to every feature. Takes priority
        over ``target_aware`` placement and the automatic strategy, and overrides
        ``output_dim``.

    adaptive : bool, default=False
        If True, the per-feature output dimension may vary within
        ``[min_output_dim, max_output_dim]``.

    min_output_dim : int or None, default=None
        Lower bound on the per-feature output dimension used in adaptive mode.

    max_output_dim : int or None, default=None
        Upper bound on the per-feature output dimension used in adaptive mode.

    task : {"regression", "classification"} or None, default=None
        Task passed to the target-aware selector when ``target_aware=True``.

    random_state : int or None, default=None
        Random state forwarded to the target-aware selector for reproducibility.

    Attributes
    ----------
    knots_ : list of ndarray
        Full knot vector (with repeated boundary knots) for each feature.

    n_knots_ : list of int
        Number of interior knots placed for each feature
        (``output_dim - degree - 1`` on the default, non-selector path).

    n_basis_ : list of int
        Number of spline basis functions for each feature, excluding the bias term
        (equals ``output_dim`` on the default path).

    n_features_in_ : int
        Number of input features seen during ``fit``.

    total_output_dim_ : int
        Total number of output columns across all features (fitted).

    Notes
    -----
    Knot placement follows a fixed priority: explicit ``knot_locations`` take
    precedence over target-aware ``placement_strategy`` selection, which in turn
    takes precedence over the automatic ``output_dim`` strategy. Multi-column input
    is expanded column by column and stacked horizontally.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import BSplineTransformer
    >>> X = np.linspace(0, 1, 50).reshape(-1, 1)
    >>> BSplineTransformer(output_dim=8).fit_transform(X).shape
    (50, 9)
    """

    def __init__(
        self,
        output_dim=UNSET,
        degree: int = 3,
        target_aware: bool = False,
        placement_strategy: str = "quantile",
        include_bias: bool = False,
        knot_locations: np.ndarray | None = None,
        adaptive: bool = False,
        min_output_dim=UNSET,
        max_output_dim=UNSET,
        task: Literal["regression", "classification"] | None = None,
        random_state: int | None = None,
    ):
        self.output_dim = output_dim
        self.degree = degree
        self.target_aware = target_aware
        self.placement_strategy = placement_strategy
        self.include_bias = include_bias
        self.knot_locations = knot_locations
        self.adaptive = adaptive
        self.min_output_dim = min_output_dim
        self.max_output_dim = max_output_dim
        self.task: Literal["regression", "classification"] | None = task
        self.random_state = random_state

    _selector_spline_type: ClassVar[Literal["bspline", "mspline", "ispline"]] = "bspline"

    def _basis_to_knots(self, n_basis: int) -> int:
        """Convert a basis-function count to the number of internal knots."""
        return basis_to_knots(n_basis, self.degree)

    def _resolve_basis_bounds(
        self, n_basis: int, min_basis_req: int | None, max_basis_req: int | None
    ) -> tuple[int, int]:
        """Return the (min, max) number of basis functions to allow per feature."""
        return self._resolve_output_bounds(
            n_basis,
            min_basis_req,
            max_basis_req,
            floor=self.degree + 1,
            floor_label=f"degree + 1 = {self.degree + 1}",
            ceil=50,
        )

    def _generate_knots(self, x: np.ndarray, n_knots: int, strategy: str) -> np.ndarray:
        """Generate internal knots for a single feature using the chosen strategy."""
        return generate_internal_knots(x, n_knots, strategy)

    def _adjust_internal_knots(
        self, x: np.ndarray, internal_knots: np.ndarray, min_knots: int, max_knots: int
    ) -> np.ndarray:
        """Clip, deduplicate and rebalance internal knots to the allowed count."""
        internal_knots = np.clip(internal_knots, x.min(), x.max())
        internal_knots = np.unique(np.sort(internal_knots))
        if len(internal_knots) < min_knots:
            internal_knots = self._supplement_knots(x, internal_knots, min_knots)
        if len(internal_knots) > max_knots:
            internal_knots = select_knots(internal_knots, max_knots)
        return internal_knots

    def _supplement_knots(self, x: np.ndarray, internal_knots: np.ndarray, target_count: int) -> np.ndarray:
        """Add quantile and uniform candidates until ``target_count`` knots exist."""
        if target_count <= len(internal_knots):
            return internal_knots

        candidates = [internal_knots]
        if target_count > 0:
            candidates.append(quantile_knots(x, target_count))
            candidates.append(uniform_knots(x, target_count))

        combined = np.unique(np.concatenate(candidates))
        combined = np.sort(combined)
        if len(combined) < target_count:
            combined = uniform_knots(x, target_count)
        return select_knots(np.asarray(combined), target_count)

    def _column_knots(
        self,
        x_valid: np.ndarray,
        y_valid: np.ndarray | None,
        n_basis: int,
        strategy: str,
        selector: BaseKnotSelector | None,
        min_basis_req: int | None,
        max_basis_req: int | None,
    ) -> np.ndarray:
        """Build the full knot vector for a single feature."""
        min_basis, max_basis = self._resolve_basis_bounds(n_basis, min_basis_req, max_basis_req)
        min_knots = self._basis_to_knots(min_basis)
        max_knots = self._basis_to_knots(max_basis)
        if self.adaptive:
            max_knots = min(max_knots, max(0, np.unique(x_valid).size - 2))

        x_min = x_valid.min()
        x_max = x_valid.max()

        if self.knot_locations is not None:
            expected_knots = self._basis_to_knots(n_basis)
            if not self.adaptive and len(self.knot_locations) != expected_knots:
                raise IncompatibleParamsError("knot_locations length must match output_dim - degree - 1 when adaptive=False")
            internal_knots = self._adjust_internal_knots(x_valid, np.asarray(self.knot_locations), min_knots, max_knots)
        elif selector is not None:
            selected = selector.get_knot_locations(x_valid.reshape(-1, 1), y_valid, task=self.task)
            internal_knots = self._adjust_internal_knots(x_valid, np.asarray(selected), min_knots, max_knots)
        else:
            n_internal = self._basis_to_knots(n_basis)
            internal_knots = self._generate_knots(x_valid, n_internal, strategy)
            internal_knots = self._adjust_internal_knots(x_valid, internal_knots, min_knots, max_knots)

        internal_knots = np.clip(internal_knots, x_min, x_max)
        internal_knots = np.unique(np.sort(internal_knots))

        boundary_left = np.repeat(x_min, self.degree + 1)
        boundary_right = np.repeat(x_max, self.degree + 1)
        return np.concatenate([boundary_left, internal_knots, boundary_right])

    def fit(self, X, y=None):
        """Determine per-feature knot vectors."""
        validate_placement(self.target_aware, self.placement_strategy)
        n_basis = self._resolve_param("output_dim", default=6)
        min_basis_req = self._resolve_param("min_output_dim", default=None)
        max_basis_req = self._resolve_param("max_output_dim", default=None)
        if n_basis < self.degree + 1:
            raise InvalidParamError(f"output_dim must be >= degree + 1 = {self.degree + 1}, got {n_basis}")
        if n_basis > 50:
            raise InvalidParamError(f"output_dim should be <= 50, got {n_basis}")

        X = self._validate(X, reset=True)

        y_arr = None if y is None else np.asarray(y).ravel()

        # Knot placement priority: explicit knot_locations win, then a target-aware
        # selector built from placement_strategy, then the automatic (unsupervised)
        # spacing named by placement_strategy.
        if self.target_aware and self.knot_locations is None:
            selector = build_knot_selector(
                self.placement_strategy,
                degree=self.degree,
                spline_type=self._selector_spline_type,
                random_state=self.random_state,
            )
            strategy = "quantile"
        else:
            selector = None
            strategy = self.placement_strategy if not self.target_aware else "quantile"

        self.knots_ = []
        for i in range(X.shape[1]):
            xi = X[:, i]
            valid_mask = ~np.isnan(xi)
            xi_valid = xi[valid_mask]
            if xi_valid.size == 0:
                raise PretabDataError(f"Feature at index {i} has only NaN values")
            yi_valid = y_arr[valid_mask] if y_arr is not None else None
            self.knots_.append(
                self._column_knots(xi_valid, yi_valid, n_basis, strategy, selector, min_basis_req, max_basis_req)
            )

        self.n_basis_ = [len(knots) - self.degree - 1 for knots in self.knots_]
        self.n_knots_ = [len(knots) - 2 * (self.degree + 1) for knots in self.knots_]
        return self

    def transform(self, X):
        """Expand each feature into its spline basis and stack the results."""
        check_is_fitted(self, "knots_")
        X = self._validate(X, reset=False)

        transformed = []
        for i in range(X.shape[1]):
            knots = self.knots_[i]
            xi_clipped = np.clip(X[:, i], knots[0], knots[-1])
            design = self._design_matrix(xi_clipped, knots)
            if self.include_bias:
                design = np.hstack([np.ones((design.shape[0], 1)), design])
            transformed.append(design)

        return np.hstack(transformed)

    def fit_transform(self, X, y=None):
        """Fit to the data then return the transformed features."""
        return self.fit(X, y).transform(X)

    def _design_matrix(self, x: np.ndarray, knots: np.ndarray) -> np.ndarray:
        """Return the basis matrix for a single feature (without the bias column)."""
        raise NotImplementedError

    def _feature_suffix(self) -> str:
        """Suffix used when generating output feature names."""
        return "spline"

    def _output_sizes(self) -> list[int]:
        """Number of output columns per feature, including the optional bias."""
        bias = 1 if self.include_bias else 0
        return [int(n) + bias for n in self.n_basis_]

    def get_n_features_out(self) -> int:
        """Total number of output columns across all features."""
        check_is_fitted(self, "n_basis_")
        return int(sum(self._output_sizes()))

    def get_penalty_matrix(self, feature_index: int = 0, diff_order: int = 2):
        """
        Return a difference penalty matrix for the given feature.

        The penalty is ``D^T D`` where ``D`` is the ``diff_order`` difference
        operator on the spline coefficients. When ``include_bias`` is enabled the
        bias column is left unpenalized (a leading zero row and column).
        """
        check_is_fitted(self, "n_basis_")

        n_basis = self.n_basis_[feature_index]
        D = np.eye(n_basis)
        for _ in range(diff_order):
            D = np.diff(D, n=1, axis=0)
        penalty = D.T @ D

        if self.include_bias:
            full = np.zeros((n_basis + 1, n_basis + 1))
            full[1:, 1:] = penalty
            return full
        return penalty
