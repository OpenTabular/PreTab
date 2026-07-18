"""Shared adaptive / fixed output-dimension resolution for expansion families.

Every PreTab expansion (splines, feature maps, piecewise-linear encoding,
tree-driven binning) exposes the same sizing vocabulary: a fixed ``output_dim``
plus an optional adaptive window ``[min_output_dim, max_output_dim]``. When
``adaptive`` is ``False`` each feature is expanded to exactly ``output_dim``
columns; when ``True`` the per-feature count may vary within the window and is
chosen by the family's own data-driven placement (decision-tree splits, CART
knot selection, quantile spacing, ...), then clamped into that window.

``AdaptiveResolutionMixin`` holds the *one* implementation of that window
resolution and its validation so no family reimplements it. Families call
:meth:`AdaptiveResolutionMixin._resolve_output_bounds`, passing only the
family-specific floor (and optional ceiling) on the count.
"""

__all__ = ["AdaptiveResolutionMixin"]


class AdaptiveResolutionMixin:
    """Resolve the ``(min, max)`` per-feature output window shared by all families.

    Subclasses expose an ``adaptive`` attribute (set from the ``adaptive``
    constructor argument). :meth:`_resolve_output_bounds` returns the inclusive
    ``(lo, hi)`` bounds on the per-feature output dimension:

    * ``adaptive is False`` -> ``lo == hi == output_dim`` (a fixed width), after
      checking ``output_dim`` is consistent with any explicitly supplied
      ``min``/``max`` request.
    * ``adaptive is True`` -> ``lo``/``hi`` come from the requested
      ``min``/``max`` (each falling back to ``output_dim`` when unset).

    The resolved window is then validated against a family-specific ``floor``
    (minimum admissible count, e.g. ``degree + 1`` basis functions or ``1`` bin)
    and optional ``ceil`` (maximum admissible count, e.g. ``50`` spline bases).
    """

    adaptive: bool

    def _resolve_output_bounds(
        self,
        output_dim: int,
        min_req: int | None,
        max_req: int | None,
        *,
        floor: int,
        floor_label: str | None = None,
        ceil: int | None = None,
    ) -> tuple[int, int]:
        """Return the inclusive ``(lo, hi)`` per-feature output-dimension window.

        Parameters
        ----------
        output_dim:
            The resolved fixed output dimension (already read from the estimator's
            ``output_dim`` parameter, with its family default applied).
        min_req, max_req:
            The requested ``min_output_dim`` / ``max_output_dim`` (``None`` when
            the caller left them unset).
        floor:
            Smallest admissible per-feature count for this family (e.g.
            ``degree + 1`` for a spline basis, ``1`` for a bin count).
        floor_label:
            Human-readable form of ``floor`` used in the error message (defaults
            to ``str(floor)``). Lets splines report ``"degree + 1 = 4"``.
        ceil:
            Largest admissible per-feature count, or ``None`` for no upper limit.
        """
        if not self.adaptive:
            if min_req is not None and output_dim < min_req:
                raise ValueError("output_dim must be >= min_output_dim when adaptive=False")
            if max_req is not None and output_dim > max_req:
                raise ValueError("output_dim must be <= max_output_dim when adaptive=False")
            lo = hi = output_dim
        else:
            lo = min_req if min_req is not None else output_dim
            hi = max_req if max_req is not None else output_dim

        label = floor_label if floor_label is not None else str(floor)
        if lo < floor:
            raise ValueError(f"min_output_dim must be >= {label}, got {lo}")
        if ceil is not None and hi > ceil:
            raise ValueError(f"max_output_dim should be <= {ceil}, got {hi}")
        if lo > hi:
            raise ValueError("min_output_dim must be <= max_output_dim")
        return lo, hi
