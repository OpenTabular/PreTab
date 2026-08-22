"""Factory for building placement strategies from the public parameter vocabulary.

:func:`create_placement_strategy` is the single entry point transformers use to
turn the user-facing ``target_aware`` / ``placement_strategy`` pair into a
concrete :class:`~pretab.placement.base.BasePlacementStrategy`. It validates the
``target_aware`` / ``placement_strategy`` combination up front via
:func:`pretab.core.parameters.validate_placement`, so an invalid pairing fails
with one clear error instead of surfacing deep inside a family.

The count window (``min_count`` / ``max_count``) and endpoint convention
(``include_endpoints``) are resolved by the caller, typically a family adapter
in :mod:`pretab.placement.adapters`, and passed straight through.
"""

from __future__ import annotations

from ..core.parameters import validate_placement
from ..core.selectors import Task
from ..exceptions import invalid_param_error
from .base import BasePlacementStrategy
from .supervised import CARTPlacement, LightGBMPlacement
from .unsupervised import QuantilePlacement, UniformPlacement

__all__ = ["create_placement_strategy"]


def create_placement_strategy(
    *,
    target_aware: bool,
    placement_strategy: str,
    min_count: int,
    max_count: int,
    task: Task | None = "regression",
    random_state: int | None = None,
    include_endpoints: bool = False,
) -> BasePlacementStrategy:
    """Build a placement strategy from ``target_aware`` / ``placement_strategy``.

    Parameters
    ----------
    target_aware : bool
        Whether the target ``y`` is used to place locations. Selects the
        supervised (``True``) or unsupervised (``False``) family.
    placement_strategy : {"cart", "lightgbm", "uniform", "quantile"}
        The strategy name. Must be a supervised selector (``"cart"`` /
        ``"lightgbm"``) when ``target_aware`` is True, or an unsupervised spacing
        rule (``"uniform"`` / ``"quantile"``) when False.
    min_count, max_count : int
        Inclusive count window. The supervised strategies place a data-driven
        count inside it; the unsupervised strategies place exactly ``max_count``
        locations (callers pass ``min_count == max_count`` for a fixed width).
    task : {"regression", "classification"}, optional
        Task forwarded to the supervised strategies. Ignored when unsupervised.
    random_state : int or None, default=None
        Forwarded to the supervised strategies for reproducibility.
    include_endpoints : bool, default=False
        Endpoint convention for the unsupervised strategies (``False`` -> interior
        locations, ``True`` -> range-spanning). Ignored when supervised.

    Returns
    -------
    BasePlacementStrategy
        A ready-to-fit placement strategy.

    Raises
    ------
    InvalidParamError
        If ``placement_strategy`` is not valid for the chosen ``target_aware``.
    """
    validate_placement(target_aware, placement_strategy)

    if target_aware:
        if placement_strategy == "cart":
            return CARTPlacement(min_count=min_count, max_count=max_count, task=task, random_state=random_state)
        return LightGBMPlacement(min_count=min_count, max_count=max_count, task=task, random_state=random_state)

    if placement_strategy == "uniform":
        return UniformPlacement(max_count, include_endpoints=include_endpoints)
    if placement_strategy == "quantile":
        return QuantilePlacement(max_count, include_endpoints=include_endpoints)

    # Unreachable: validate_placement already rejected any other name.
    raise invalid_param_error(
        "create_placement_strategy",
        "placement_strategy",
        placement_strategy,
        "must be one of 'cart', 'lightgbm', 'uniform', 'quantile'",
        valid={"cart", "lightgbm", "quantile", "uniform"},
    )
