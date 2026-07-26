"""Unsupervised placement: locations from feature geometry alone.

:class:`UniformPlacement` spaces locations evenly across a feature's range;
:class:`QuantilePlacement` puts them at evenly spaced data quantiles. Neither
uses the target, so both fit without a ``y`` and their effective unit count
always equals the requested count.

The two endpoint conventions PreTab uses are exposed through ``include_endpoints``:

* ``include_endpoints=False`` (default) returns *interior* locations -- the
  B/M/I-spline internal-knot convention (:func:`pretab.core.knots.uniform_knots` /
  :func:`~pretab.core.knots.quantile_knots`).
* ``include_endpoints=True`` returns locations that span the full range, endpoints
  included -- the feature-map center and spanning-knot convention
  (:func:`pretab.core.knots.spanning_knots`).

Both paths delegate to the shared knot primitives so placement stays numerically
identical to the historical per-family code.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from ..core.knots import quantile_knots, spanning_knots, uniform_knots
from .base import BasePlacementStrategy, PlacementResult

__all__ = ["QuantilePlacement", "UniformPlacement"]


class _UnsupervisedPlacement(BasePlacementStrategy):
    """Shared machinery for the fixed-count, target-free placement strategies.

    Parameters
    ----------
    n_units : int
        Number of locations to place per feature (the requested and, for these
        deterministic strategies, effective count).
    include_endpoints : bool, default=False
        ``False`` returns interior locations (internal-knot convention); ``True``
        returns range-spanning locations with the endpoints included.
    """

    target_aware: ClassVar[bool] = False

    def __init__(self, n_units: int, *, include_endpoints: bool = False):
        self.n_units = n_units
        self.include_endpoints = include_endpoints

    def _place(self, x: np.ndarray, n_units: int) -> np.ndarray:
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> _UnsupervisedPlacement:
        x = np.asarray(x, dtype=float).ravel()
        x = x[~np.isnan(x)]
        self.locations_ = np.asarray(self._place(x, self.n_units))
        return self

    def get_locations(self) -> PlacementResult:
        return PlacementResult(
            locations=self.locations_,
            requested_units=self.n_units,
            effective_units=int(self.locations_.shape[0]),
            strategy=self.name,
            target_aware=False,
        )


class UniformPlacement(_UnsupervisedPlacement):
    """Evenly spaced locations across a feature's range."""

    name: ClassVar[str] = "uniform"

    def _place(self, x: np.ndarray, n_units: int) -> np.ndarray:
        if self.include_endpoints:
            return spanning_knots(x, n_units, "uniform")
        return uniform_knots(x, n_units)


class QuantilePlacement(_UnsupervisedPlacement):
    """Locations at evenly spaced data quantiles of a feature."""

    name: ClassVar[str] = "quantile"

    def _place(self, x: np.ndarray, n_units: int) -> np.ndarray:
        if self.include_endpoints:
            return spanning_knots(x, n_units, "quantile")
        return quantile_knots(x, n_units)
