"""Target-aware knot selection strategies for spline transformers.

A knot selector looks at a single feature and its target, then returns the
internal knot positions a spline basis should use. Placing knots where the
feature actually changes its relationship with the target usually produces a more
faithful basis than spreading knots uniformly.

Two strategies are provided:

- :class:`CARTKnotSelector` uses a single decision tree and needs only
  scikit-learn, so it is always available.
- :class:`LightGBMKnotSelector` uses a gradient boosted ensemble and requires the
  optional ``lightgbm`` dependency (``pip install pretab[knots]``).

Both are thin, spline-aware adapters over the degree-agnostic count-based
selectors in :mod:`pretab.core.selectors`. The adapter converts a number of
spline basis functions into a number of internal knots (which depends on the
spline degree) and then asks the underlying location selector for that many
locations.
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from ...core.exceptions import IncompatibleParamsError, invalid_param_error
from ...core.knots import basis_to_knots
from ...core.selectors import CARTLocationSelector, LightGBMLocationSelector


class BaseKnotSelector(ABC):
    """Abstract base class for knot selection strategies.

    Subclasses implement :meth:`get_knot_locations`, returning the internal knot
    positions (boundary knots are added later by the spline transformer). The
    basis-to-knot conversion, which depends on the spline degree, lives here so
    the concrete selectors stay small.

    The following attributes are expected to be set by every subclass:
    ``degree``, ``spline_type``, ``min_knot_spacing``, ``min_knots`` and
    ``max_knots``.
    """

    degree: int
    spline_type: Literal["bspline", "mspline", "ispline"]
    min_knot_spacing: float
    min_knots: int
    max_knots: int

    @abstractmethod
    def get_knot_locations(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        task: Literal["regression", "classification"] | None = None,
    ) -> np.ndarray:
        """Return internal knot locations for a single feature.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples,) or (n_samples, 1)
            Input feature values for one feature.
        y : np.ndarray of shape (n_samples,), optional
            Target values. May be None for selectors that are not target aware.
        task : {"regression", "classification"}, optional
            Type of prediction task.

        Returns
        -------
        knot_locations : np.ndarray
            Sorted array of internal knot locations.
        """
        raise NotImplementedError

    def _basis_to_knots(self, n_basis: int) -> int:
        """Convert a number of basis functions into a number of internal knots."""
        if self.spline_type in ("bspline", "mspline", "ispline"):
            return basis_to_knots(n_basis, self.degree)
        raise invalid_param_error(
            type(self).__name__,
            "spline_type",
            self.spline_type,
            "must be one of 'bspline', 'mspline', 'ispline'",
            valid={"bspline", "mspline", "ispline"},
        )


class CARTKnotSelector(BaseKnotSelector):
    """Select knots from the split points of a single decision tree.

    A ``DecisionTreeRegressor`` or ``DecisionTreeClassifier`` is fitted to the
    feature against the target, and its split thresholds become the candidate
    knots. Candidates are spaced out, and if there are too many they are ranked
    by weighted impurity decrease so the most informative splits are kept.

    Parameters
    ----------
    max_tree_depth : int, default=6
        Maximum depth of the decision tree.
    min_samples_split : int, default=20
        Minimum samples required to split a node.
    min_samples_leaf : int, default=10
        Minimum samples required in a leaf.
    min_knot_spacing : float, default=0.01
        Minimum distance between adjacent knots, as a fraction of the feature range.
    min_basis_functions : int, default=3
        Minimum basis functions. Falls back to quantile knots if the tree yields
        fewer splits.
    max_basis_functions : int, default=15
        Maximum basis functions. The top splits are kept if the tree exceeds this.
    degree : int, default=3
        Spline degree, used to convert basis functions into internal knots.
    spline_type : {"bspline", "mspline", "ispline"}, default="bspline"
        Spline family the knots are intended for.
    random_state : int or None, default=51
        Random state for reproducibility.
    """

    def __init__(
        self,
        max_tree_depth: int = 6,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        min_knot_spacing: float = 0.01,
        min_basis_functions: int = 3,
        max_basis_functions: int = 15,
        degree: int = 3,
        spline_type: Literal["bspline", "mspline", "ispline"] = "bspline",
        random_state: int | None = 51,
    ):
        self.max_tree_depth = max_tree_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_knot_spacing = min_knot_spacing
        self.min_basis_functions = min_basis_functions
        self.max_basis_functions = max_basis_functions
        self.degree = degree
        self.spline_type = spline_type
        self.random_state = random_state

        self.min_knots = self._basis_to_knots(min_basis_functions)
        self.max_knots = self._basis_to_knots(max_basis_functions)

        self._selector = CARTLocationSelector(
            max_tree_depth=max_tree_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_location_spacing=min_knot_spacing,
            random_state=random_state,
        )

    def get_knot_locations(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        task: Literal["regression", "classification"] | None = "regression",
    ) -> np.ndarray:
        if y is None:
            raise IncompatibleParamsError("CARTKnotSelector requires y to select knots.")
        return self._selector.select(X, y, task=task, min_count=self.min_knots, max_count=self.max_knots)


class LightGBMKnotSelector(BaseKnotSelector):
    """Select knots from the split points of a LightGBM ensemble.

    A gradient boosted ensemble is fitted to the feature against the target, and
    split thresholds are ranked by their cumulative gain across all trees. This
    tends to find informative knots that a single tree can miss.

    Requires the optional ``lightgbm`` dependency, installable with
    ``pip install pretab[knots]``.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds.
    max_depth : int, default=3
        Maximum depth of each tree.
    learning_rate : float, default=0.1
        Boosting learning rate.
    min_child_samples : int, default=20
        Minimum samples in a leaf.
    min_knot_spacing : float, default=0.01
        Minimum distance between adjacent knots, as a fraction of the feature range.
    min_basis_functions : int, default=3
        Minimum basis functions. Falls back to quantile knots if fewer splits found.
    max_basis_functions : int, default=15
        Maximum basis functions. The top-gain splits are kept if more are found.
    degree : int, default=3
        Spline degree, used to convert basis functions into internal knots.
    spline_type : {"bspline", "mspline", "ispline"}, default="bspline"
        Spline family the knots are intended for.
    random_state : int or None, default=51
        Random state for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_child_samples: int = 20,
        min_knot_spacing: float = 0.01,
        min_basis_functions: int = 3,
        max_basis_functions: int = 15,
        degree: int = 3,
        spline_type: Literal["bspline", "mspline", "ispline"] = "bspline",
        random_state: int | None = 51,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.min_knot_spacing = min_knot_spacing
        self.min_basis_functions = min_basis_functions
        self.max_basis_functions = max_basis_functions
        self.degree = degree
        self.spline_type = spline_type
        self.random_state = random_state

        self.min_knots = self._basis_to_knots(min_basis_functions)
        self.max_knots = self._basis_to_knots(max_basis_functions)

        self._selector = LightGBMLocationSelector(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            min_location_spacing=min_knot_spacing,
            random_state=random_state,
        )

    def get_knot_locations(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        task: Literal["regression", "classification"] | None = "regression",
    ) -> np.ndarray:
        if y is None:
            raise IncompatibleParamsError("LightGBMKnotSelector requires y to select knots.")
        return self._selector.select(X, y, task=task, min_count=self.min_knots, max_count=self.max_knots)


def build_knot_selector(
    placement_strategy: str,
    *,
    degree: int,
    spline_type: Literal["bspline", "mspline", "ispline"] = "bspline",
    random_state: int | None = None,
) -> BaseKnotSelector:
    """Build a target-aware knot selector from a ``placement_strategy`` name.

    ``placement_strategy`` must be ``"cart"`` (a single decision tree, always
    available) or ``"lightgbm"`` (a gradient-boosted ensemble, requires the
    optional ``lightgbm`` dependency). ``random_state`` is only forwarded when
    set, so an unset value keeps each selector's own default seed.
    """
    kwargs: dict = {"degree": degree, "spline_type": spline_type}
    if random_state is not None:
        kwargs["random_state"] = random_state
    if placement_strategy == "cart":
        return CARTKnotSelector(**kwargs)
    if placement_strategy == "lightgbm":
        return LightGBMKnotSelector(**kwargs)
    raise invalid_param_error(
        "build_knot_selector",
        "placement_strategy",
        placement_strategy,
        "must be 'cart' or 'lightgbm' when target_aware=True",
        valid={"cart", "lightgbm"},
    )
