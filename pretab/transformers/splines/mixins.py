"""Shared scikit-learn contract helpers for the spline transformers.

The spline families in this package differ in how they build a basis, but they
share the same scikit-learn plumbing: NaN-aware input validation, a fitted guard,
output feature names, and the estimator tag that lets missing values pass through
to a later imputer. ``SplineBasisMixin`` collects that plumbing in one place so
each concrete transformer only has to implement its own basis math.
"""

import warnings
from typing import Literal

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted


class SplineBasisMixin:
    """Mixin holding the shared scikit-learn contract for spline transformers.

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

    def _feature_suffix(self) -> str:
        """Suffix used when generating output feature names."""
        return self._feature_suffix_value

    def _validate_allow_nan(self, X, *, reset: bool):
        """Validate ``X`` while letting NaN values pass through.

        Coerces ``X`` to a 2D float array, keeps missing values so a later imputer
        can handle them, warns if validation drops columns, and records
        ``n_features_in_`` when ``reset`` is True (during ``fit``).
        """
        original_dim = np.shape(X)[1] if np.ndim(X) == 2 else 1
        finite_policy: Literal["allow-nan"] | bool = "allow-nan"
        X = check_array(X, dtype=np.float64, ensure_2d=True, ensure_all_finite=finite_policy)
        if X.shape[1] < original_dim:
            warnings.warn(
                "Some input features were dropped during check_array validation.",
                UserWarning,
                stacklevel=2,
            )
        if reset:
            self.n_features_in_ = X.shape[1]
        return X

    def _output_sizes(self) -> list[int]:
        """Number of output columns contributed by each input feature."""
        return [int(n) for n in self.n_basis_]

    def get_feature_names_out(self, input_features=None):
        """Return output feature names of the form ``{feature}_{suffix}{j}``."""
        check_is_fitted(self, "n_basis_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        suffix = self._feature_suffix()
        names = []
        for feature, n_cols in zip(input_features, self._output_sizes(), strict=False):
            for j in range(n_cols):
                names.append(f"{feature}_{suffix}{j}")
        return np.asarray(names, dtype=object)

    def __sklearn_tags__(self):
        """Declare that these transformers can pass NaN through to an imputer."""
        tags = super().__sklearn_tags__()  # type: ignore
        tags.input_tags.allow_nan = True
        return tags
