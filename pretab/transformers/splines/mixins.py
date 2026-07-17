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

from ...core.base import BasePreTabTransformer


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
