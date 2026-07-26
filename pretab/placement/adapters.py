"""Family adapters: convert generic placement into family-specific locations.

The placement strategies in :mod:`pretab.placement.supervised` /
:mod:`pretab.placement.unsupervised` speak in *locations* and *unit counts*. Each
transformer family, though, has its own vocabulary and conventions:

* splines think in *basis functions* -> *internal knots* (a degree-dependent
  conversion) and place knots strictly interior to the data range;
* PLE thinks in *bins* -> *thresholds* (``bins - 1``) and is target-aware only;
* feature maps think in *centers* that span the range with the endpoints included.

These adapters own exactly that translation, so the placement strategies stay
family-neutral. Each is a faithful reimplementation of the historical per-family
selection code on top of the shared placement strategies, so knot / threshold /
center positions are numerically unchanged.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..core.knots import basis_to_knots
from ..core.selectors import Task
from ..exceptions import invalid_param_error
from .factory import create_placement_strategy

__all__ = [
    "PLEPlacementAdapter",
    "PeriodicPlacementAdapter",
    "RBFPlacementAdapter",
    "SplinePlacementAdapter",
]

# The spline knot selectors have always searched a fixed basis-function window,
# independent of the requested output_dim (the transformer clamps to output_dim
# afterwards). These reproduce ``CART/LightGBMKnotSelector``'s defaults.
_SPLINE_MIN_BASIS = 3
_SPLINE_MAX_BASIS = 15
# Historical default seed used by the spline knot selectors when random_state is
# left unset (feature maps / PLE forward their own default instead).
_SPLINE_DEFAULT_SEED = 51


class SplinePlacementAdapter:
    """Target-aware knot placement for the B/M/I spline families.

    A drop-in replacement for the old ``build_knot_selector(...)`` product: it
    exposes the same :meth:`get_knot_locations` signature the spline base and
    mixin call, but sources locations from a shared
    :class:`~pretab.placement.supervised` strategy. The basis-function search
    window (``min_basis_functions`` / ``max_basis_functions``) is converted to an
    internal-knot count via :func:`pretab.core.knots.basis_to_knots`, exactly as
    before.

    Parameters
    ----------
    degree : int
        Spline degree, used to convert basis functions into internal knots.
    placement_strategy : {"cart", "lightgbm"}
        Target-aware selector to place the knots.
    spline_type : {"bspline", "mspline", "ispline"}, default="bspline"
        Retained for parity with the previous selector API (the knot count depends
        only on ``degree``).
    random_state : int or None, default=None
        Random state forwarded to the strategy. When unset the historical spline
        default seed (51) is used.
    min_basis_functions, max_basis_functions : int
        Basis-function search window (defaults 3 and 15, matching the old
        selectors).
    """

    def __init__(
        self,
        *,
        degree: int,
        placement_strategy: str,
        spline_type: Literal["bspline", "mspline", "ispline"] = "bspline",
        random_state: int | None = None,
        min_basis_functions: int = _SPLINE_MIN_BASIS,
        max_basis_functions: int = _SPLINE_MAX_BASIS,
    ):
        if placement_strategy not in ("cart", "lightgbm"):
            raise invalid_param_error(
                type(self).__name__,
                "placement_strategy",
                placement_strategy,
                "must be 'cart' or 'lightgbm' when target_aware=True",
                valid={"cart", "lightgbm"},
            )
        self.degree = degree
        self.placement_strategy = placement_strategy
        self.spline_type = spline_type
        self.random_state = random_state
        self.min_knots = basis_to_knots(min_basis_functions, degree)
        self.max_knots = basis_to_knots(max_basis_functions, degree)

    def get_knot_locations(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        task: Task | None = "regression",
    ) -> np.ndarray:
        """Return sorted internal knot locations for a single feature."""
        seed = self.random_state if self.random_state is not None else _SPLINE_DEFAULT_SEED
        strategy = create_placement_strategy(
            target_aware=True,
            placement_strategy=self.placement_strategy,
            min_count=self.min_knots,
            max_count=self.max_knots,
            task=task,
            random_state=seed,
        )
        return strategy.fit(X, y).get_locations().locations


class PLEPlacementAdapter:
    """Target-aware threshold placement for Piecewise Linear Encoding.

    PLE is inherently target-aware: only the supervised strategies apply. The
    caller resolves the ``[min_count, max_count]`` *threshold* window (one fewer
    than the bin count) and this adapter returns the sorted thresholds.

    Parameters
    ----------
    placement_strategy : {"cart", "lightgbm"}
        Target-aware selector to place the thresholds.
    task : {"regression", "classification"}, default="regression"
        Prediction task passed to the selector.
    random_state : int or None, default=None
        Random state forwarded to the strategy as-is.
    """

    def __init__(
        self,
        *,
        placement_strategy: str,
        task: Task | None = "regression",
        random_state: int | None = None,
    ):
        if placement_strategy not in ("cart", "lightgbm"):
            raise invalid_param_error(
                type(self).__name__,
                "placement_strategy",
                placement_strategy,
                "must be 'cart' or 'lightgbm'",
                valid={"cart", "lightgbm"},
            )
        self.placement_strategy = placement_strategy
        self.task: Task | None = task
        self.random_state = random_state

    def get_thresholds(self, x: np.ndarray, y: np.ndarray, min_count: int, max_count: int) -> np.ndarray:
        """Return sorted bin thresholds for a single feature."""
        strategy = create_placement_strategy(
            target_aware=True,
            placement_strategy=self.placement_strategy,
            min_count=min_count,
            max_count=max_count,
            task=self.task,
            random_state=self.random_state,
        )
        return np.sort(strategy.fit(x, y).get_locations().locations)


class RBFPlacementAdapter:
    """Center placement for the center-based feature maps (RBF/ReLU/sigmoid/tanh).

    Feature-map centers span the feature range with the endpoints included, and
    may be placed either target-aware (CART / LightGBM) or unsupervised
    (uniform / quantile). The caller resolves the ``[min_count, max_count]``
    window (equal bounds on the non-adaptive path).

    Parameters
    ----------
    target_aware : bool
        Whether to use the supervised strategies.
    placement_strategy : {"cart", "lightgbm", "uniform", "quantile"}
        Placement strategy, validated against ``target_aware``.
    task : {"regression", "classification"}, default="regression"
        Prediction task for the supervised strategies.
    random_state : int or None, default=None
        Random state forwarded to the supervised strategies as-is.
    """

    def __init__(
        self,
        *,
        target_aware: bool,
        placement_strategy: str,
        task: Task | None = "regression",
        random_state: int | None = None,
    ):
        self.target_aware = target_aware
        self.placement_strategy = placement_strategy
        self.task: Task | None = task
        self.random_state = random_state

    def get_centers(self, x: np.ndarray, y: np.ndarray | None, min_count: int, max_count: int) -> np.ndarray:
        """Return sorted centers for a single feature."""
        strategy = create_placement_strategy(
            target_aware=self.target_aware,
            placement_strategy=self.placement_strategy,
            min_count=min_count,
            max_count=max_count,
            task=self.task,
            random_state=self.random_state,
            include_endpoints=True,
        )
        return strategy.fit(x, y).get_locations().locations


class PeriodicPlacementAdapter:
    """Forward-declared adapter for the periodic (cyclic) encoder.

    Periodic encoding is parameter-driven (``period`` and ``harmonics``) rather
    than placement-driven: it does not locate data-dependent knots or centers.
    This adapter exists so the capability registry and placement factory can name
    a placement entry for every family uniformly; its data-driven placement is
    reserved for a later phase (e.g. learned phase offsets or frequency
    selection) and raises until then.
    """

    def get_locations(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError(
            "Periodic encoding is parameter-driven (period, harmonics) and does not use "
            "data-dependent placement."
        )
