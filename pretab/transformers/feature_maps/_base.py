"""Shared base for the center-placed feature-map expansions.

The RBF / ReLU / sigmoid / tanh transformers differ only in the per-column kernel
they apply; everything else -- parameter handling, NaN-aware validation, center
placement (decision tree / quantile / uniform), the transform loop, feature
names, and estimator tags -- is identical. ``BaseCenterExpansion`` holds that
shared machinery so each concrete transformer only implements ``_expand_column``.
"""

from typing import cast

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ...core.base import BasePreTabTransformer
from ...core.exceptions import (
    IncompatibleParamsError,
    InvalidParamError,
    PretabDataError,
)
from ...core.params import UNSET, validate_placement
from ...core.selectors import CARTLocationSelector, LightGBMLocationSelector


class BaseCenterExpansion(BasePreTabTransformer):
    """Base class for feature maps that expand each column around fixed centers.

    Subclasses set ``_feature_suffix_value`` and implement
    :meth:`_expand_column`; they typically add a single kernel parameter (such as
    ``gamma`` or ``scale``) in their own ``__init__``.

    Centers are placed either from a target-aware location selector (when
    ``target_aware`` is True, which then requires ``y``) or from ``quantile`` /
    ``uniform`` spacing (when ``target_aware`` is False, the default).
    ``placement_strategy`` selects the mechanism: ``"cart"`` or ``"lightgbm"``
    when target-aware, otherwise ``"uniform"`` or ``"quantile"``. When left unset
    it resolves to ``"cart"`` on the target-aware path and ``"quantile"``
    otherwise. Defaulting to unsupervised placement lets these expansions fit
    without a target; pass ``target_aware=True`` (with ``y``) to place centers
    where they best separate it. (PLE, by contrast, is inherently target-aware.)

    Adaptive sizing (``adaptive=True``) only takes effect on the target-aware
    path: each feature's centers are clamped into
    ``[min_output_dim, max_output_dim]``. On the ``quantile`` / ``uniform`` paths
    (and whenever ``adaptive`` is False) each feature keeps exactly ``output_dim``
    centers, reproducing the non-adaptive behavior.
    """

    centers_: list

    def __init__(
        self,
        output_dim=UNSET,
        target_aware: bool = False,
        placement_strategy=UNSET,
        task: str = "regression",
        adaptive: bool = False,
        min_output_dim=UNSET,
        max_output_dim=UNSET,
        random_state: int | None = None,
    ):
        self.output_dim = output_dim
        self.target_aware = target_aware
        self.placement_strategy = placement_strategy
        self.task = task
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
        """Place per-feature centers from a target-aware selector or quantile/uniform spacing."""
        placement_strategy = self._resolve_placement_strategy()
        validate_placement(self.target_aware, placement_strategy)
        if self.task not in ("regression", "classification"):
            raise InvalidParamError(
                f"Invalid task. Choose 'regression' or 'classification'. Got {self.task!r}."
            )
        n_centers = self._resolve_param("output_dim", default=6)
        min_req = self._resolve_param("min_output_dim", default=None)
        max_req = self._resolve_param("max_output_dim", default=None)
        X = self._validate(X, reset=True)

        if n_centers < 1:
            raise InvalidParamError(f"output_dim must be >= 1, got {n_centers}")

        if self.target_aware and y is None:
            raise IncompatibleParamsError(
                "Target variable 'y' must be provided when target_aware=True."
            )

        if self.target_aware:
            # Centers come from a target-aware location selector (CART by default,
            # optionally LightGBM): split points spaced out and ranked by impurity
            # / gain. Adaptive sizing clamps each feature into [min, max]; otherwise
            # each feature keeps exactly ``output_dim`` centers.
            selector = self._build_selector(placement_strategy)
            if self.adaptive:
                min_centers, max_centers = self._resolve_output_bounds(
                    n_centers, min_req, max_req, floor=1
                )
            else:
                min_centers = max_centers = n_centers
            centers_list = [
                selector.select(
                    X[:, i], y, task=self.task,
                    min_count=min_centers, max_count=max_centers,
                )
                for i in range(X.shape[1])
            ]
        elif placement_strategy == "quantile":
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

    def _resolve_placement_strategy(self) -> str:
        """Resolve ``placement_strategy``, defaulting by ``target_aware`` when unset.

        Leaving ``placement_strategy`` unset selects ``"cart"`` on the target-aware
        path and ``"quantile"`` on the unsupervised path, so ``target_aware`` alone
        always yields a valid pairing.
        """
        if self.placement_strategy is not UNSET:
            return cast(str, self.placement_strategy)
        return "cart" if self.target_aware else "quantile"

    def _build_selector(self, placement_strategy):
        """Construct the target-aware location selector named by ``placement_strategy``."""
        if placement_strategy == "cart":
            return CARTLocationSelector(random_state=self.random_state)
        if placement_strategy == "lightgbm":
            return LightGBMLocationSelector(random_state=self.random_state)
        raise InvalidParamError(
            f"Invalid placement_strategy. Choose 'cart' or 'lightgbm'. Got {placement_strategy!r}."
        )

    def __sklearn_tags__(self):
        """Require ``y`` only when centers are placed by a target-aware selector."""
        tags = super().__sklearn_tags__()
        tags.target_tags.required = bool(self.target_aware)
        return tags
