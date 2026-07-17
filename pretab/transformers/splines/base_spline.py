"""
Shared base class for spline basis expansions (B-spline, M-spline, I-spline).

The base class handles the parts that are common to every spline family:

- resolving the number of basis functions and the derived number of internal knots,
- placing knots per feature using one of three strategies with a fixed priority
  (``knot_selector`` > ``knot_locations`` > automatic ``n_basis_functions``),
- looping over columns so that multi-column input is supported,
- assembling the design matrix, optional bias column, penalty matrix and feature
  names.

Each concrete transformer only implements how a single column is turned into a
basis matrix through the ``_design_matrix`` hook.
"""

from typing import Literal

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ...core.base import BasePreTabTransformer
from ...core.knots import (
    basis_to_knots,
    generate_internal_knots,
    quantile_knots,
    select_knots,
    uniform_knots,
)
from .knot_selectors import BaseKnotSelector


class BaseSplineTransformer(BasePreTabTransformer):
    """
    Base class for spline basis expansions with target-aware knot placement.

    Parameters
    ----------
    n_basis_functions : int, default=5
        Number of basis functions per feature. The number of internal knots is
        derived as ``n_basis_functions - degree - 1``. Must be between 5 and 50.
        Ignored when ``knot_locations`` or ``knot_selector`` is provided.

    degree : int, default=3
        Degree of the spline (3 is cubic, 2 quadratic, 1 linear).

    knot_strategy : {"uniform", "quantile"}, default="quantile"
        Placement rule used for the automatic strategy. ``"uniform"`` spaces knots
        evenly across the range, ``"quantile"`` places them at data quantiles.

    include_bias : bool, default=False
        Whether to prepend a constant intercept column per feature.

    knot_locations : ndarray or None, default=None
        Explicit internal knot locations applied to every feature. Overrides
        ``n_basis_functions``.

    knot_selector : BaseKnotSelector or None, default=None
        Target-aware knot selector (for example ``CARTKnotSelector``). Takes
        priority over all other placement options and requires ``y`` during fit.

    adaptive : bool, default=False
        If True, the number of basis functions may vary per feature within
        ``[min_basis_functions, max_basis_functions]``.

    min_basis_functions : int or None, default=None
        Lower bound used in adaptive mode.

    max_basis_functions : int or None, default=None
        Upper bound used in adaptive mode.

    n_knots : int or None, default=None
        Compatibility alias for ``n_basis_functions``. When set, it takes
        precedence and is interpreted as the number of basis functions. This lets
        the Preprocessor keep passing ``n_knots`` to spline strategies.

    task : {"regression", "classification"} or None, default=None
        Task passed to a target-aware ``knot_selector``.

    Attributes
    ----------
    knots_ : list of ndarray
        Full knot vector (with repeated boundary knots) for each feature.

    n_basis_ : list of int
        Number of spline basis functions for each feature, excluding the bias term.

    n_features_in_ : int
        Number of input features seen during ``fit``.

    Notes
    -----
    Knot placement follows a fixed priority: a target-aware ``knot_selector``
    takes precedence over explicit ``knot_locations``, which in turn take
    precedence over the automatic ``n_basis_functions`` strategy. Multi-column
    input is expanded column by column and stacked horizontally.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import BSplineTransformer
    >>> X = np.linspace(0, 1, 50).reshape(-1, 1)
    >>> BSplineTransformer(n_basis_functions=8).fit_transform(X).shape
    (50, 9)
    """

    def __init__(
        self,
        n_basis_functions: int = 5,
        degree: int = 3,
        knot_strategy: str = "quantile",
        include_bias: bool = False,
        knot_locations: np.ndarray | None = None,
        knot_selector: BaseKnotSelector | None = None,
        adaptive: bool = False,
        min_basis_functions: int | None = None,
        max_basis_functions: int | None = None,
        n_knots: int | None = None,
        task: Literal["regression", "classification"] | None = None,
    ):
        self.n_basis_functions = n_basis_functions
        self.degree = degree
        self.knot_strategy = knot_strategy
        self.include_bias = include_bias
        self.knot_locations = knot_locations
        self.knot_selector = knot_selector
        self.adaptive = adaptive
        self.min_basis_functions = min_basis_functions
        self.max_basis_functions = max_basis_functions
        self.n_knots = n_knots
        self.task: Literal["regression", "classification"] | None = task

    def _effective_n_basis(self) -> int:
        """Resolve the requested number of basis functions, honouring ``n_knots``."""
        return self.n_knots if self.n_knots is not None else self.n_basis_functions

    def _basis_to_knots(self, n_basis: int) -> int:
        """Convert a basis-function count to the number of internal knots."""
        return basis_to_knots(n_basis, self.degree)

    def _resolve_basis_bounds(self, n_basis: int) -> tuple[int, int]:
        """Return the (min, max) number of basis functions to allow per feature."""
        if not self.adaptive:
            if self.min_basis_functions is not None and n_basis < self.min_basis_functions:
                raise ValueError("n_basis_functions must be >= min_basis_functions when adaptive=False")
            if self.max_basis_functions is not None and n_basis > self.max_basis_functions:
                raise ValueError("n_basis_functions must be <= max_basis_functions when adaptive=False")
            min_basis = n_basis
            max_basis = n_basis
        else:
            min_basis = self.min_basis_functions if self.min_basis_functions is not None else n_basis
            max_basis = self.max_basis_functions if self.max_basis_functions is not None else n_basis

        if min_basis < self.degree + 1:
            raise ValueError(f"min_basis_functions must be >= degree + 1 = {self.degree + 1}, got {min_basis}")
        if max_basis > 50:
            raise ValueError(f"max_basis_functions should be <= 50, got {max_basis}")
        if min_basis > max_basis:
            raise ValueError("min_basis_functions must be <= max_basis_functions")
        return min_basis, max_basis

    def _generate_knots(self, x: np.ndarray, n_knots: int) -> np.ndarray:
        """Generate internal knots for a single feature using the chosen strategy."""
        return generate_internal_knots(x, n_knots, self.knot_strategy)

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

    def _column_knots(self, x_valid: np.ndarray, y_valid: np.ndarray | None, n_basis: int) -> np.ndarray:
        """Build the full knot vector for a single feature."""
        min_basis, max_basis = self._resolve_basis_bounds(n_basis)
        min_knots = self._basis_to_knots(min_basis)
        max_knots = self._basis_to_knots(max_basis)
        if self.adaptive:
            max_knots = min(max_knots, max(0, np.unique(x_valid).size - 2))

        x_min = x_valid.min()
        x_max = x_valid.max()

        if self.knot_selector is not None:
            selected = self.knot_selector.get_knot_locations(x_valid.reshape(-1, 1), y_valid, task=self.task)
            internal_knots = self._adjust_internal_knots(x_valid, np.asarray(selected), min_knots, max_knots)
        elif self.knot_locations is not None:
            expected_knots = self._basis_to_knots(n_basis)
            if not self.adaptive and len(self.knot_locations) != expected_knots:
                raise ValueError("knot_locations length must match n_basis_functions when adaptive=False")
            internal_knots = self._adjust_internal_knots(x_valid, np.asarray(self.knot_locations), min_knots, max_knots)
        else:
            n_internal = self._basis_to_knots(n_basis)
            internal_knots = self._generate_knots(x_valid, n_internal)
            internal_knots = self._adjust_internal_knots(x_valid, internal_knots, min_knots, max_knots)

        internal_knots = np.clip(internal_knots, x_min, x_max)
        internal_knots = np.unique(np.sort(internal_knots))

        boundary_left = np.repeat(x_min, self.degree + 1)
        boundary_right = np.repeat(x_max, self.degree + 1)
        return np.concatenate([boundary_left, internal_knots, boundary_right])

    def fit(self, X, y=None):
        """Determine per-feature knot vectors."""
        n_basis = self._effective_n_basis()
        if n_basis < self.degree + 1:
            raise ValueError(f"n_basis_functions must be >= degree + 1 = {self.degree + 1}, got {n_basis}")
        if n_basis < 5:
            raise ValueError(f"n_basis_functions must be >= 5 to ensure at least 1 internal knot, got {n_basis}")
        if n_basis > 50:
            raise ValueError(f"n_basis_functions should be <= 50, got {n_basis}")

        X = self._validate(X, reset=True)

        y_arr = None if y is None else np.asarray(y).ravel()

        self.knots_ = []
        for i in range(X.shape[1]):
            xi = X[:, i]
            valid_mask = ~np.isnan(xi)
            xi_valid = xi[valid_mask]
            if xi_valid.size == 0:
                raise ValueError(f"Feature at index {i} has only NaN values")
            yi_valid = y_arr[valid_mask] if y_arr is not None else None
            self.knots_.append(self._column_knots(xi_valid, yi_valid, n_basis))

        self.n_basis_ = [len(knots) - self.degree - 1 for knots in self.knots_]
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
