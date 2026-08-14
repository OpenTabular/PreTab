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
from ...core.parameters import UNSET, validate_placement
from ...core.supervised import warn_target_leakage
from ...exceptions import (
    IncompatibleParamsError,
    InvalidParamError,
    PretabDataError,
)
from ...placement.adapters import RBFPlacementAdapter


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

    _representation_component_kind = "center"
    _representation_supervision = "optional"

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
        warn_target_leakage(self, y)
        placement_strategy = self._resolve_placement_strategy()
        validate_placement(self.target_aware, placement_strategy)
        if self.task not in ("regression", "classification"):
            raise InvalidParamError(f"Invalid task. Choose 'regression' or 'classification'. Got {self.task!r}.")
        n_centers = self._resolve_param("output_dim", default=6)
        min_req = self._resolve_param("min_output_dim", default=None)
        max_req = self._resolve_param("max_output_dim", default=None)
        X = self._validate(X, reset=True)

        if n_centers < 1:
            raise InvalidParamError(f"output_dim must be >= 1, got {n_centers}")

        if self.target_aware and y is None:
            raise IncompatibleParamsError("Target variable 'y' must be provided when target_aware=True.")

        # Centers come from the placement subsystem: a target-aware selector
        # (CART / LightGBM) when ``target_aware``, otherwise quantile / uniform
        # spacing across the range with the endpoints included. Adaptive sizing
        # only takes effect on the target-aware path, clamping each feature into
        # [min, max]; otherwise each feature keeps exactly ``output_dim`` centers.
        adapter = RBFPlacementAdapter(
            target_aware=self.target_aware,
            placement_strategy=placement_strategy,
            task=self.task,
            random_state=self.random_state,
        )
        if self.target_aware and self.adaptive:
            min_centers, max_centers = self._resolve_output_bounds(n_centers, min_req, max_req, floor=1)
        else:
            min_centers = max_centers = n_centers
        y_place = y if self.target_aware else None
        self.centers_ = []
        for i in range(X.shape[1]):
            if np.isnan(X[:, i]).all():
                raise PretabDataError(f"Feature at index {i} has only NaN values")
            self.centers_.append(adapter.get_centers(X[:, i], y_place, min_centers, max_centers))
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

    def __sklearn_tags__(self):
        """Require ``y`` only when centers are placed by a target-aware selector."""
        tags = super().__sklearn_tags__()
        tags.target_tags.required = bool(self.target_aware)
        return tags
