"""Shared base for the center-placed feature-map expansions.

The RBF / ReLU / sigmoid / tanh transformers differ only in the per-column kernel
they apply; everything else -- parameter handling, NaN-aware validation, center
placement (decision tree / quantile / uniform), the transform loop, feature
names, and estimator tags -- is identical. ``BaseCenterExpansion`` holds that
shared machinery so each concrete transformer only implements ``_expand_column``.
"""

from typing import ClassVar

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..core.base import BasePreTabTransformer
from ..core.centers import center_identification_using_decision_tree
from ..core.exceptions import (
    IncompatibleParamsError,
    InvalidParamError,
    PretabDataError,
)
from ..core.params import UNSET, is_set


class BaseCenterExpansion(BasePreTabTransformer):
    """Base class for feature maps that expand each column around fixed centers.

    Subclasses set ``_feature_suffix_value`` and implement
    :meth:`_expand_column`; they typically add a single kernel parameter (such as
    ``gamma`` or ``scale``) in their own ``__init__``.

    Centers are placed either from decision-tree split thresholds (when
    ``use_decision_tree`` is True, which then requires ``y``) or from ``quantile``
    / ``uniform`` spacing.
    """

    centers_: list

    _param_aliases: ClassVar[dict[str, str]] = {
        "n_centers": "n_basis",
        "use_decision_tree": "use_target",
    }

    def __init__(
        self,
        n_basis=UNSET,
        use_target=UNSET,
        task: str = "regression",
        strategy="uniform",
        n_centers=UNSET,
        use_decision_tree=UNSET,
    ):
        self.n_basis = n_basis
        self.use_target = use_target
        self.task = task
        self.strategy = strategy
        self.n_centers = n_centers
        self.use_decision_tree = use_decision_tree

        if self.strategy not in ("uniform", "quantile"):
            raise InvalidParamError(
                f"Invalid strategy. Choose 'uniform' or 'quantile'. Got {self.strategy!r}."
            )
        if self.task not in ("regression", "classification"):
            raise InvalidParamError(
                f"Invalid task. Choose 'regression' or 'classification'. Got {self.task!r}."
            )

    def _expand_column(self, x_col, centers):
        """Expand a single column ``x_col`` (shape ``(n, 1)``) against ``centers``.

        Returns an array of shape ``(n, len(centers))``. Implemented by subclasses.
        """
        raise NotImplementedError

    def fit(self, X, y=None):
        """Place per-feature centers from a decision tree or quantile/uniform spacing."""
        n_centers = self._resolve_param("n_basis", default=10)
        use_target = self._resolve_param("use_target", default=True)
        X = self._validate(X, reset=True)

        if use_target and y is None:
            raise IncompatibleParamsError(
                "Target variable 'y' must be provided when use_decision_tree=True."
            )

        if use_target:
            centers_list = center_identification_using_decision_tree(
                X, y, self.task, n_centers
            )
        elif self.strategy == "quantile":
            centers_list = [
                np.percentile(X[:, i], np.linspace(0, 100, n_centers))
                for i in range(X.shape[1])
            ]
        else:  # uniform
            centers_list = [
                np.linspace(X[:, i].min(), X[:, i].max(), n_centers)
                for i in range(X.shape[1])
            ]

        self.centers_ = centers_list
        return self

    def transform(self, X):
        """Expand every feature against its centers and stack the results."""
        check_is_fitted(self, "centers_")
        X = self._validate(X, reset=False)

        if len(self.centers_) != X.shape[1]:
            raise PretabDataError("X and centers must have the same number of features.")

        transformed = []
        for i in range(X.shape[1]):
            centers = np.asarray(self.centers_[i])
            transformed.append(self._expand_column(X[:, [i]], centers))

        return np.hstack(transformed)

    def _output_sizes(self) -> list[int]:
        """Number of output columns contributed by each input feature."""
        return [int(np.asarray(centers).shape[0]) for centers in self.centers_]

    def __sklearn_tags__(self):
        """Require ``y`` only when centers are placed with a decision tree."""
        tags = super().__sklearn_tags__()
        if is_set(self.use_decision_tree):
            use_target = self.use_decision_tree
        elif is_set(self.use_target):
            use_target = self.use_target
        else:
            use_target = True
        tags.target_tags.required = bool(use_target)
        return tags
