"""Core placement contract: :class:`BasePlacementStrategy` and :class:`PlacementResult`.

A *placement strategy* answers the "where" question for a single feature: given
that feature's values (and optionally a target), it produces a sorted array of
locations along the feature -- spline knots, feature-map centers, or PLE
thresholds. It is deliberately unit-agnostic (it returns *locations*, not basis
functions); converting a requested number of basis functions into a number of
locations is the job of the family adapters in :mod:`pretab.placement.adapters`.

The contract is intentionally tiny so both the unsupervised (uniform / quantile)
and supervised (CART / LightGBM) families, and the family adapters, can share it:

* :meth:`~BasePlacementStrategy.fit` looks at one feature and stores the chosen
  locations, and
* :meth:`~BasePlacementStrategy.get_locations` returns a frozen
  :class:`PlacementResult` describing them (locations plus the requested and
  effective unit counts, the strategy name, and whether the target was used).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

__all__ = ["BasePlacementStrategy", "PlacementResult"]


@dataclass(frozen=True)
class PlacementResult:
    """Immutable description of the locations a strategy placed for one feature.

    Parameters
    ----------
    locations : np.ndarray
        Sorted array of placed locations along the feature.
    requested_units : int
        Number of locations the caller asked for (the resolved upper bound of the
        ``[min_count, max_count]`` window).
    effective_units : int
        Number of locations actually produced. Equals ``requested_units`` for the
        fixed-count unsupervised families; may be smaller for the data-driven
        supervised families, which can find fewer informative splits.
    strategy : str
        Name of the strategy that produced the locations (``"uniform"``,
        ``"quantile"``, ``"cart"``, ``"lightgbm"``).
    target_aware : bool
        Whether the placement used the target ``y``.
    """

    locations: np.ndarray
    requested_units: int
    effective_units: int
    strategy: str
    target_aware: bool


class BasePlacementStrategy(ABC):
    """Abstract base class for single-feature location placement strategies.

    Subclasses set the ``name`` and ``target_aware`` class attributes and
    implement :meth:`fit` (store the chosen locations on ``self``) and
    :meth:`get_locations` (return the frozen :class:`PlacementResult`).

    The strategy is stateful and single-feature: call :meth:`fit` with one
    feature's values (and its target when ``target_aware``), then
    :meth:`get_locations`. :meth:`place`, provided here, chains the two for
    callers that do not need to keep the fitted strategy around.
    """

    name: ClassVar[str] = ""
    target_aware: ClassVar[bool] = False

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> BasePlacementStrategy:
        """Look at one feature (and optional target) and store the locations.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples,) or (n_samples, 1)
            Values of a single feature.
        y : np.ndarray of shape (n_samples,), optional
            Target values. Required by the supervised strategies.

        Returns
        -------
        self : BasePlacementStrategy
            The fitted strategy.
        """
        raise NotImplementedError

    @abstractmethod
    def get_locations(self) -> PlacementResult:
        """Return the :class:`PlacementResult` produced by the last :meth:`fit`."""
        raise NotImplementedError

    def place(self, x: np.ndarray, y: np.ndarray | None = None) -> PlacementResult:
        """Convenience: :meth:`fit` on ``x``/``y`` then return :meth:`get_locations`."""
        return self.fit(x, y).get_locations()
