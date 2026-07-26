"""Shared base class implementing the common scikit-learn transformer contract.

``BasePreTabTransformer`` centralizes NaN-aware validation, the ``allow_nan`` /
``requires_y`` estimator tags, a ``check_is_fitted`` guard, and default
``get_feature_names_out`` generation, so concrete transformers only implement
their own basis / encoding math.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .adaptive import AdaptiveResolutionMixin
from .parameters import AliasResolverMixin
from .representation import RepresentationSpecMixin
from .validation import validate_2d_allow_nan

__all__ = ["BasePreTabTransformer"]


class BasePreTabTransformer(
    RepresentationSpecMixin, AdaptiveResolutionMixin, AliasResolverMixin, TransformerMixin, BaseEstimator
):
    """Base class carrying the shared scikit-learn contract for PreTab transformers.

    Subclasses set ``_allow_nan`` / ``_requires_y`` / ``_feature_suffix_value`` as
    needed and implement ``_output_sizes`` (the number of output columns each
    input feature contributes) to get automatic ``get_feature_names_out`` support.
    Transformers whose output is not a simple per-feature concatenation can
    override ``get_feature_names_out`` directly.
    """

    _allow_nan: bool = True
    _requires_y: bool = False
    _feature_suffix_value: str = "f"

    n_features_in_: int

    def _validate(self, X, *, reset: bool):
        """Validate ``X`` through the shared NaN-aware validator."""
        return validate_2d_allow_nan(X, allow_nan=self._allow_nan, reset=reset, estimator=self)

    def _feature_suffix(self) -> str:
        """Suffix used when generating output feature names."""
        return self._feature_suffix_value

    def _output_sizes(self) -> list[int]:
        """Number of output columns contributed by each input feature."""
        raise NotImplementedError

    def get_feature_names_out(self, input_features=None):
        """Return output feature names of the form ``{feature}_{suffix}{j}``."""
        check_is_fitted(self, "n_features_in_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        suffix = self._feature_suffix()
        names = []
        for feature, n_cols in zip(input_features, self._output_sizes(), strict=False):
            for j in range(n_cols):
                names.append(f"{feature}_{suffix}{j}")
        return np.asarray(names, dtype=object)

    @property
    def total_output_dim_(self) -> int:
        """Total number of output columns produced across all input features.

        Fitted attribute (available only after ``fit``). Defined as
        ``len(self.get_feature_names_out())`` so it always equals the true width
        of the transformed array, including any bias columns and, for interaction
        bases such as the tensor-product spline, the full product of the marginal
        sizes.
        """
        check_is_fitted(self, "n_features_in_")
        return len(self.get_feature_names_out())

    def __sklearn_tags__(self):
        """Declare NaN-passthrough and target-requirement estimator tags."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = self._allow_nan
        tags.target_tags.required = self._requires_y
        return tags
