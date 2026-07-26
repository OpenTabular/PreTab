"""Supervised placement: locations where the feature's effect on the target changes.

:class:`CARTPlacement` fits a single decision tree (scikit-learn only, always
available); :class:`LightGBMPlacement` fits a gradient-boosted ensemble and needs
the optional ``lightgbm`` dependency. Both look at one feature against the target
and return the split thresholds -- spaced out, ranked (impurity for CART, gain for
LightGBM), and topped up / trimmed to land in ``[min_count, max_count]``. Because
placement is data-driven, the effective unit count can be smaller than requested.

Both strategies delegate to the count-based selectors in
:mod:`pretab.core.selectors`, so placement stays numerically identical to the
historical per-family code.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from ..core.selectors import (
    BaseLocationSelector,
    CARTLocationSelector,
    LightGBMLocationSelector,
    Task,
)
from ..exceptions import IncompatibleParamsError
from .base import BasePlacementStrategy, PlacementResult

__all__ = ["CARTPlacement", "LightGBMPlacement"]


class _SupervisedPlacement(BasePlacementStrategy):
    """Shared machinery for the target-aware placement strategies.

    Parameters
    ----------
    min_count, max_count : int
        Inclusive bounds on the number of locations to return.
    task : {"regression", "classification"}, default="regression"
        Prediction task passed to the underlying tree model.
    random_state : int or None, default=None
        Forwarded to the selector for reproducibility (only when set, so an unset
        value keeps the selector's own default seed).
    """

    target_aware: ClassVar[bool] = True

    def __init__(
        self,
        *,
        min_count: int,
        max_count: int,
        task: Task | None = "regression",
        random_state: int | None = None,
    ):
        self.min_count = min_count
        self.max_count = max_count
        self.task: Task | None = task
        self.random_state = random_state
        self._selector = self._build_selector()

    def _build_selector(self) -> BaseLocationSelector:
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> _SupervisedPlacement:
        if y is None:
            raise IncompatibleParamsError(f"{type(self).__name__} requires y to place locations.")
        self.locations_ = self._selector.select(
            x,
            y,
            task=self.task,
            min_count=self.min_count,
            max_count=self.max_count,
        )
        return self

    def get_locations(self) -> PlacementResult:
        return PlacementResult(
            locations=self.locations_,
            requested_units=self.max_count,
            effective_units=int(self.locations_.shape[0]),
            strategy=self.name,
            target_aware=True,
        )


class CARTPlacement(_SupervisedPlacement):
    """Target-aware placement from a single decision tree's split points."""

    name: ClassVar[str] = "cart"

    def _build_selector(self) -> BaseLocationSelector:
        return CARTLocationSelector(random_state=self.random_state)


class LightGBMPlacement(_SupervisedPlacement):
    """Target-aware placement from a LightGBM ensemble's gain-ranked split points."""

    name: ClassVar[str] = "lightgbm"

    def _build_selector(self) -> BaseLocationSelector:
        return LightGBMLocationSelector(random_state=self.random_state)
