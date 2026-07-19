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

from ...core.base import BasePreTabTransformer
from ...core.centers import center_identification_using_decision_tree
from ...core.exceptions import (
    IncompatibleParamsError,
    InvalidParamError,
    PretabDataError,
)
from ...core.knots import select_knots, spanning_knots
from ...core.params import UNSET, is_set


class BaseCenterExpansion(BasePreTabTransformer):
    """Base class for feature maps that expand each column around fixed centers.

    Subclasses set ``_feature_suffix_value`` and implement
    :meth:`_expand_column`; they typically add a single kernel parameter (such as
    ``gamma`` or ``scale``) in their own ``__init__``.

    Centers are placed either from decision-tree split thresholds (when
    ``use_decision_tree`` is True, which then requires ``y``) or from ``quantile``
    / ``uniform`` spacing.

    Adaptive sizing (``adaptive=True``) only takes effect on the target-aware
    decision-tree path: the tree is grown up to ``max_output_dim`` leaves and the
    resulting per-feature centers are clamped into
    ``[min_output_dim, max_output_dim]``. On the ``quantile`` / ``uniform`` paths
    (and whenever ``adaptive`` is False) each feature keeps exactly ``output_dim``
    centers, reproducing the non-adaptive behavior.
    """

    centers_: list

    _param_aliases: ClassVar[dict[str, str]] = {
        "use_decision_tree": "use_target",
    }

    def __init__(
        self,
        output_dim=UNSET,
        use_target=UNSET,
        task: str = "regression",
        strategy="uniform",
        use_decision_tree=UNSET,
        adaptive: bool = False,
        min_output_dim=UNSET,
        max_output_dim=UNSET,
        random_state: int | None = None,
    ):
        self.output_dim = output_dim
        self.use_target = use_target
        self.task = task
        self.strategy = strategy
        self.use_decision_tree = use_decision_tree
        self.adaptive = adaptive
        self.min_output_dim = min_output_dim
        self.max_output_dim = max_output_dim
        self.random_state = random_state

    def _expand_column(self, x_col, centers):
        """Expand a single column ``x_col`` (shape ``(n, 1)``) against ``centers``.

        Returns an array of shape ``(n, len(centers))``. Implemented by subclasses.
        """
        raise NotImplementedError

    def fit(self, X, y=None):
        """Place per-feature centers from a decision tree or quantile/uniform spacing."""
        if self.strategy not in ("uniform", "quantile"):
            raise InvalidParamError(
                f"Invalid strategy. Choose 'uniform' or 'quantile'. Got {self.strategy!r}."
            )
        if self.task not in ("regression", "classification"):
            raise InvalidParamError(
                f"Invalid task. Choose 'regression' or 'classification'. Got {self.task!r}."
            )
        n_centers = self._resolve_param("output_dim", default=10)
        use_target = self._resolve_param("use_target", default=True)
        min_req = self._resolve_param("min_output_dim", default=None)
        max_req = self._resolve_param("max_output_dim", default=None)
        X = self._validate(X, reset=True)

        if n_centers < 1:
            raise InvalidParamError(f"output_dim must be >= 1, got {n_centers}")

        if use_target and y is None:
            raise IncompatibleParamsError(
                "Target variable 'y' must be provided when use_decision_tree=True."
            )

        if use_target and self.adaptive:
            # Adaptive sizing only applies on the target-aware tree path: grow the
            # tree up to ``max`` leaves, then clamp each feature into [min, max].
            min_centers, max_centers = self._resolve_output_bounds(
                n_centers, min_req, max_req, floor=1
            )
            raw_centers = center_identification_using_decision_tree(
                X, y, self.task, max_centers, random_state=self.random_state
            )
            centers_list = [
                self._adjust_centers(X[:, i], centers, min_centers, max_centers)
                for i, centers in enumerate(raw_centers)
            ]
        elif use_target:
            centers_list = center_identification_using_decision_tree(
                X, y, self.task, n_centers, random_state=self.random_state
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

    def _adjust_centers(
        self, x: np.ndarray, centers: np.ndarray, min_centers: int, max_centers: int
    ) -> np.ndarray:
        """Clamp a data-driven set of centers into the ``[min, max]`` window."""
        centers = np.unique(np.sort(np.asarray(centers, dtype=float)))
        if len(centers) > max_centers:
            centers = select_knots(centers, max_centers)
        if len(centers) < min_centers:
            centers = self._supplement_centers(x, centers, min_centers)
        return centers

    def _supplement_centers(
        self, x: np.ndarray, centers: np.ndarray, target: int
    ) -> np.ndarray:
        """Add quantile / uniform candidates until ``target`` centers exist."""
        if target <= len(centers):
            return centers
        candidates = [np.asarray(centers, dtype=float)]
        candidates.append(spanning_knots(x, target, "quantile"))
        candidates.append(spanning_knots(x, target, "uniform"))
        combined = np.unique(np.concatenate(candidates))
        if len(combined) > target:
            combined = select_knots(combined, target)
        return combined

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
