"""Shared scikit-learn contract helpers for the spline transformers.

The spline families in this package differ in how they build a basis, but they
share the same scikit-learn plumbing: NaN-aware input validation, a fitted guard,
output feature names, and the estimator tag that lets missing values pass through
to a later imputer. That plumbing now lives in
:class:`pretab.core.base.BasePreTabTransformer`; ``SplineBasisMixin`` is a thin
shim that adapts it to the spline transformers (per-feature ``n_basis_`` output
sizes and the legacy ``_validate_allow_nan`` entry point) so each concrete
transformer only has to implement its own basis math.
"""

import numpy as np

from ...core.base import BasePreTabTransformer
from ...core.exceptions import IncompatibleParamsError
from ...core.knots import generate_internal_knots, select_knots, spanning_knots


class SplineBasisMixin(BasePreTabTransformer):
    """Spline-specific adapter over :class:`BasePreTabTransformer`.

    Concrete transformers combine this with ``BaseEstimator`` and
    ``TransformerMixin`` and place it first in the base list so its
    ``get_feature_names_out`` and ``__sklearn_tags__`` take precedence::

        class MySpline(SplineBasisMixin, TransformerMixin, BaseEstimator):
            _feature_suffix_value = "ms"

    Subclasses expose one output size per input feature through the ``n_basis_``
    attribute (a list with one entry per column). Transformers whose output is not
    a simple per-feature concatenation, such as an interaction basis, can override
    ``_output_sizes`` or ``get_feature_names_out`` directly.
    """

    _feature_suffix_value = "spline"

    n_basis_: list[int]

    def _validate_allow_nan(self, X, *, reset: bool):
        """Validate ``X`` while letting NaN values pass through.

        Thin backward-compatible wrapper around
        :meth:`BasePreTabTransformer._validate`.
        """
        return self._validate(X, reset=reset)

    def _output_sizes(self) -> list[int]:
        """Number of output columns contributed by each input feature."""
        return [int(n) for n in self.n_basis_]

    def _place_spanning_knots(self, x, y, n_basis, strategy, selector, task, min_interior=None, max_interior=None):
        """Return a spanning knot vector (endpoints included) for one feature.

        When ``selector`` is provided the internal knots come from the target-aware
        selector and are bracketed by the feature's min/max; otherwise
        ``n_basis`` knots are placed with :func:`pretab.core.knots.spanning_knots`.
        ``strategy="uniform"`` reproduces the legacy ``np.linspace(min, max, n_basis)``
        placement exactly. When ``min_interior`` / ``max_interior`` are given (the
        adaptive selector path) the number of interior knots is clamped into that
        window before bracketing.
        """
        x = np.asarray(x)
        if selector is not None:
            interior = self._place_interior_knots(
                x, y, 0, strategy, selector, task, min_interior, max_interior
            )
            x_min, x_max = x.min(), x.max()
            return np.concatenate([[x_min], interior, [x_max]])
        return spanning_knots(x, n_basis, strategy)

    def _place_interior_knots(self, x, y, n_interior, strategy, selector, task, min_interior=None, max_interior=None):
        """Return the interior knots (endpoints excluded) for one feature.

        Unlike :meth:`_place_spanning_knots`, no boundary points are appended: the
        returned array holds only the strictly-interior knots that the B/M/I,
        cubic, p-spline and tensor-product families use to reach an exact
        ``output_dim``. When ``selector`` is provided the interior knots come from
        the target-aware selector (their count is data-driven and may differ from
        ``n_interior``); otherwise ``n_interior`` knots are placed with
        :func:`pretab.core.knots.generate_internal_knots`. When ``min_interior`` /
        ``max_interior`` are given (the adaptive selector path) the data-driven
        count is clamped into that window.
        """
        x = np.asarray(x)
        if selector is not None:
            if y is None:
                raise IncompatibleParamsError(
                    "A knot selector requires y during fit for target-aware knot placement."
                )
            selected = np.asarray(
                selector.get_knot_locations(x.reshape(-1, 1), y, task=task), dtype=float
            )
            x_min, x_max = x.min(), x.max()
            selected = np.unique(selected[(selected > x_min) & (selected < x_max)])
            if min_interior is not None or max_interior is not None:
                selected = self._clamp_interior_knots(x, selected, min_interior, max_interior, strategy)
            return selected
        return generate_internal_knots(x, n_interior, strategy)

    def _clamp_interior_knots(self, x, knots, min_count, max_count, strategy):
        """Clamp a data-driven set of interior knots into ``[min_count, max_count]``.

        Down-samples with :func:`pretab.core.knots.select_knots` when there are too
        many knots and supplements with quantile / uniform interior candidates when
        there are too few. Endpoints are never added -- the result stays strictly
        interior.
        """
        x = np.asarray(x)
        knots = np.unique(np.sort(np.asarray(knots, dtype=float)))
        if max_count is not None and len(knots) > max_count:
            knots = select_knots(knots, max_count)
        if min_count is not None and len(knots) < min_count:
            x_min, x_max = x.min(), x.max()
            candidates = [
                knots,
                generate_internal_knots(x, min_count, "quantile"),
                generate_internal_knots(x, min_count, "uniform"),
            ]
            combined = np.unique(np.concatenate(candidates))
            combined = combined[(combined > x_min) & (combined < x_max)]
            if len(combined) > min_count:
                combined = select_knots(combined, min_count)
            knots = combined
        return knots

    def _adaptive_interior_bounds(self, output_dim, selector, *, floor, offset):
        """Return ``(min_interior, max_interior)`` for the adaptive selector path.

        Adaptive sizing only takes effect on the target-aware selector path; on the
        fixed path (``adaptive`` False or no ``selector``) this returns
        ``(None, None)`` so knot placement reproduces the non-adaptive output. The
        ``[min_output_dim, max_output_dim]`` window (validated in output-dimension
        space via :meth:`_resolve_output_bounds`) is translated into interior-knot
        counts by subtracting the family ``offset`` (``output_dim - offset`` interior
        knots).
        """
        if not (self.adaptive and selector is not None):
            return None, None
        min_req = self._resolve_param("min_output_dim", default=None)
        max_req = self._resolve_param("max_output_dim", default=None)
        lo, hi = self._resolve_output_bounds(output_dim, min_req, max_req, floor=floor)
        return lo - offset, hi - offset

    def _place_bspline_knots(self, x, y, output_dim, degree, strategy, selector, task,
                             min_interior=None, max_interior=None):
        """Return the full padded B-spline knot vector for one feature.

        Places ``output_dim - degree - 1`` interior knots (via
        :meth:`_place_interior_knots`) and brackets them with ``degree + 1``
        repeated boundary knots on each side -- the B/M/I convention used by
        :class:`~pretab.transformers.splines.base_spline.BaseSplineTransformer`.
        The resulting marginal B-spline basis then has exactly ``output_dim``
        (non-bias) columns: ``len(knots) - degree - 1 == output_dim``. On the
        adaptive selector path ``min_interior`` / ``max_interior`` clamp the
        interior-knot count.
        """
        x = np.asarray(x)
        n_interior = output_dim - degree - 1
        interior = self._place_interior_knots(
            x, y, n_interior, strategy, selector, task, min_interior, max_interior
        )
        x_min, x_max = x.min(), x.max()
        boundary_left = np.repeat(x_min, degree + 1)
        boundary_right = np.repeat(x_max, degree + 1)
        return np.concatenate([boundary_left, interior, boundary_right])
