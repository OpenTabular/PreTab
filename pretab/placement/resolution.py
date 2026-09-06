"""Resolution policies: *how many* units to place, separate from *where*.

Every PreTab expansion exposes the same sizing vocabulary: a fixed ``output_dim``
plus an optional adaptive window ``[min_output_dim, max_output_dim]``. Resolving
that vocabulary into an inclusive ``(lo, hi)`` count window (and validating it
against a family floor / ceiling) is a single concern that does not depend on
*where* the units land. Keeping it here, apart from the placement strategies in
:mod:`pretab.placement.unsupervised` / :mod:`pretab.placement.supervised`, lets a
family combine any resolution policy with any placement strategy.

:class:`FixedResolution` implements the ``output_dim`` / ``[min, max]`` contract
shared by every family today. :class:`CardinalityAwareResolution` and
:class:`DataSizeAwareResolution` are internal, not-yet-implemented stubs for a
future data-driven ``(lo, hi)`` policy; every method on both currently raises
``NotImplementedError``, and neither is part of the public ``pretab.placement``
API (they are not re-exported from :mod:`pretab.placement`'s ``__all__``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..exceptions import IncompatibleParamsError, InvalidParamError

__all__ = [
    "BaseResolutionPolicy",
    "FixedResolution",
]


class BaseResolutionPolicy(ABC):
    """Abstract base class for "how many units" policies.

    A policy turns the user-facing sizing parameters (``output_dim`` and the
    optional ``[min_output_dim, max_output_dim]`` window) into an inclusive
    ``(lo, hi)`` bound on the per-feature unit count, validated against a
    family-specific ``floor`` (and optional ``ceil``).
    """

    @abstractmethod
    def resolve(
        self,
        output_dim: int,
        min_req: int | None,
        max_req: int | None,
        *,
        floor: int,
        floor_label: str | None = None,
        ceil: int | None = None,
    ) -> tuple[int, int]:
        """Return the inclusive ``(lo, hi)`` per-feature unit-count window."""
        raise NotImplementedError


class FixedResolution(BaseResolutionPolicy):
    """The ``output_dim`` / ``[min, max]`` resolution shared by every family.

    With ``adaptive=False`` each feature is expanded to exactly ``output_dim``
    units (``lo == hi == output_dim``), after checking ``output_dim`` is
    consistent with any explicitly supplied ``min``/``max`` request. With
    ``adaptive=True`` the window comes from the requested ``min``/``max`` (each
    falling back to ``output_dim`` when unset). The resolved window is validated
    against the family ``floor`` and optional ``ceil``.

    This reproduces the historical ``AdaptiveResolutionMixin._resolve_output_bounds``
    behaviour exactly.
    """

    def __init__(self, adaptive: bool):
        self.adaptive = adaptive

    def resolve(
        self,
        output_dim: int,
        min_req: int | None,
        max_req: int | None,
        *,
        floor: int,
        floor_label: str | None = None,
        ceil: int | None = None,
    ) -> tuple[int, int]:
        if not self.adaptive:
            if min_req is not None and output_dim < min_req:
                raise IncompatibleParamsError(
                    "output_dim must be >= min_output_dim when adaptive=False "
                    f"(got output_dim={output_dim}, min_output_dim={min_req}).\n"
                    "Fix: raise output_dim, lower min_output_dim, or set adaptive=True."
                )
            if max_req is not None and output_dim > max_req:
                raise IncompatibleParamsError(
                    "output_dim must be <= max_output_dim when adaptive=False "
                    f"(got output_dim={output_dim}, max_output_dim={max_req}).\n"
                    "Fix: lower output_dim, raise max_output_dim, or set adaptive=True."
                )
            lo = hi = output_dim
        else:
            lo = min_req if min_req is not None else output_dim
            hi = max_req if max_req is not None else output_dim

        label = floor_label if floor_label is not None else str(floor)
        if lo < floor:
            name = "min_output_dim" if self.adaptive and min_req is not None else "output_dim"
            raise InvalidParamError(
                f"{name} must be >= {label}, got {lo}.\nFix: raise {name} to at least the family minimum."
            )
        if ceil is not None and hi > ceil:
            name = "max_output_dim" if self.adaptive and max_req is not None else "output_dim"
            raise InvalidParamError(f"{name} should be <= {ceil}, got {hi}.\nFix: lower {name} to at most {ceil}.")
        if lo > hi:
            raise IncompatibleParamsError(
                f"min_output_dim must be <= max_output_dim (got min_output_dim={lo}, max_output_dim={hi})."
            )
        return lo, hi


class CardinalityAwareResolution(BaseResolutionPolicy):
    """Stub: cap the unit count by the feature's distinct-value count.

    Internal, not-yet-implemented placeholder for a future phase; not part of the
    public ``pretab.placement`` API.
    """

    def resolve(
        self,
        output_dim: int,
        min_req: int | None,
        max_req: int | None,
        *,
        floor: int,
        floor_label: str | None = None,
        ceil: int | None = None,
    ) -> tuple[int, int]:
        raise NotImplementedError("CardinalityAwareResolution is not implemented yet.")

    def clamp_to_cardinality(self, hi: int, x: np.ndarray) -> int:
        """Placeholder for the future distinct-value clamp."""
        raise NotImplementedError("CardinalityAwareResolution is not implemented yet.")


class DataSizeAwareResolution(BaseResolutionPolicy):
    """Stub: scale the unit count with the number of samples.

    Internal, not-yet-implemented placeholder for a future phase; not part of the
    public ``pretab.placement`` API.
    """

    def resolve(
        self,
        output_dim: int,
        min_req: int | None,
        max_req: int | None,
        *,
        floor: int,
        floor_label: str | None = None,
        ceil: int | None = None,
    ) -> tuple[int, int]:
        raise NotImplementedError("DataSizeAwareResolution is not implemented yet.")
