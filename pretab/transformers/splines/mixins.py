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
from ...core.knots import spanning_knots


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

    def _place_spanning_knots(self, x, y, n_basis, strategy, selector, task):
        """Return a spanning knot vector (endpoints included) for one feature.

        When ``selector`` is provided the internal knots come from the target-aware
        selector and are bracketed by the feature's min/max; otherwise
        ``n_basis`` knots are placed with :func:`pretab.core.knots.spanning_knots`.
        ``strategy="uniform"`` reproduces the legacy ``np.linspace(min, max, n_basis)``
        placement exactly.
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
            return np.concatenate([[x_min], selected, [x_max]])
        return spanning_knots(x, n_basis, strategy)
