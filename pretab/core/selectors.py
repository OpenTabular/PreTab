"""Count-based, target-aware location selectors.

A *location selector* looks at a single feature and its target and returns a
sorted array of interesting locations along that feature -- places where the
relationship between the feature and the target changes. It is deliberately
degree-agnostic: callers ask for a number of locations (``min_count`` /
``max_count``) rather than a number of spline basis functions, so the same
selector can drive spline knots, feature-map centers, or PLE thresholds.

Two strategies are provided:

- :class:`CARTLocationSelector` fits a single decision tree and needs only
  scikit-learn, so it is always available.
- :class:`LightGBMLocationSelector` fits a gradient boosted ensemble and requires
  the optional ``lightgbm`` dependency (``pip install pretab[knots]``).

Both share the :class:`BaseLocationSelector` template, which handles input
validation, small-sample quantile fallbacks, minimum spacing, and topping up or
trimming the candidate set to fit within ``[min_count, max_count]``.
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .exceptions import IncompatibleParamsError, OptionalDependencyError
from .knots import quantile_knots

Task = Literal["regression", "classification"]


class BaseLocationSelector(ABC):
    """Abstract base class for count-based, target-aware location selectors.

    Subclasses implement :meth:`_ordered_candidates` (fit a model and return the
    candidate locations in the selector's preferred order together with any
    context needed for trimming) and :meth:`_trim_over_max` (reduce an overfull
    candidate set to ``max_count`` using the selector's importance ranking).

    The shared :meth:`select` template validates and cleans the inputs, falls
    back to quantile locations when there are too few samples or no candidates,
    enforces a minimum spacing, and tops up or trims the result so it lands
    inside ``[min_count, max_count]``.

    The following attributes are expected to be set by every subclass:
    ``min_location_spacing`` and ``min_samples_floor``.
    """

    min_location_spacing: float
    min_samples_floor: int

    def select(
        self,
        x: np.ndarray,
        y: np.ndarray | None,
        *,
        task: Task | None = "regression",
        min_count: int,
        max_count: int,
    ) -> np.ndarray:
        """Return sorted target-aware locations for a single feature.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples,) or (n_samples, 1)
            Input feature values for one feature.
        y : np.ndarray of shape (n_samples,)
            Target values. Required -- these selectors are target aware.
        task : {"regression", "classification"}, optional
            Type of prediction task. Defaults to ``"regression"``.
        min_count : int
            Minimum number of locations to return.
        max_count : int
            Maximum number of locations to return.

        Returns
        -------
        locations : np.ndarray
            Sorted array of selected locations.
        """
        if y is None:
            raise IncompatibleParamsError(
                f"{type(self).__name__} requires y to select locations."
            )
        task = task or "regression"

        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        y = np.asarray(y).ravel()

        # Only float targets can carry NaN; for integer / object (e.g. string
        # class label) targets, leave the target validation to the fitted model.
        y_missing = np.isnan(y) if y.dtype.kind == "f" else np.zeros(len(y), dtype=bool)
        valid_mask = ~(np.isnan(x).any(axis=1) | y_missing)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        if len(x_valid) < self.min_samples_floor:
            return quantile_knots(x_valid, min_count)

        points, context = self._ordered_candidates(x_valid, y_valid, task)
        if len(points) == 0:
            return quantile_knots(x_valid, min_count)

        locations = self._enforce_spacing(points, x_valid)

        if len(locations) < min_count:
            locations = self._supplement(locations, x_valid, min_count)
        elif len(locations) > max_count:
            locations = self._trim_over_max(locations, context, max_count)

        return np.array(sorted(locations))

    @abstractmethod
    def _ordered_candidates(
        self, x_valid: np.ndarray, y_valid: np.ndarray, task: Task
    ) -> tuple[list[float], object]:
        """Fit a model and return candidate locations plus trimming context.

        The candidates must be returned in the selector's preferred order (the
        order :meth:`_enforce_spacing` should honour): location order for a
        single tree, gain-descending order for a boosted ensemble. ``context`` is
        an opaque object passed straight through to :meth:`_trim_over_max`.
        """
        raise NotImplementedError

    @abstractmethod
    def _trim_over_max(self, points: list[float], context: object, max_count: int) -> list[float]:
        """Reduce ``points`` to ``max_count`` using the selector's ranking."""
        raise NotImplementedError

    def _enforce_spacing(self, split_points: list[float], x: np.ndarray) -> list[float]:
        """Drop locations closer than ``min_location_spacing`` of the range.

        The distance test runs against *every* location kept so far rather than
        only the most recent one, which makes the filter independent of the
        order in which candidates arrive. That matters because subclasses
        deliberately order their candidates differently -- location order for a
        single tree, gain-descending order for a boosted ensemble -- and an
        order-sensitive test silently dropped every candidate positioned below
        the previously kept one, collapsing a gain-ranked set into a small
        clustered subsequence.

        For already-ascending input (the single-tree path) this is equivalent to
        comparing against the last kept location, so that behaviour is unchanged.
        """
        if len(split_points) <= 1:
            return split_points

        x_range = float(x.max() - x.min())
        min_distance = self.min_location_spacing * x_range

        spaced: list[float] = [split_points[0]]
        for point in split_points[1:]:
            if all(abs(point - kept) >= min_distance for kept in spaced):
                spaced.append(point)

        return spaced

    def _supplement(self, existing: list[float], x: np.ndarray, target_count: int) -> list[float]:
        """Top up an under-filled location set with quantile locations."""
        if target_count - len(existing) <= 0:
            return existing

        quantile_candidates = quantile_knots(x, target_count)
        all_locations = set(existing) | set(quantile_candidates.tolist())
        return sorted(all_locations)[:target_count]


class CARTLocationSelector(BaseLocationSelector):
    """Select locations from the split points of a single decision tree.

    A ``DecisionTreeRegressor`` or ``DecisionTreeClassifier`` is fitted to the
    feature against the target, and its split thresholds become the candidate
    locations. Candidates are spaced out, and if there are too many they are
    ranked by weighted impurity decrease so the most informative splits are kept.

    Parameters
    ----------
    max_tree_depth : int, default=6
        Maximum depth of the decision tree.
    min_samples_split : int, default=20
        Minimum samples required to split a node. Also the small-sample floor
        below which quantile locations are returned.
    min_samples_leaf : int, default=10
        Minimum samples required in a leaf.
    min_location_spacing : float, default=0.01
        Minimum distance between adjacent locations, as a fraction of the range.
    random_state : int or None, default=51
        Random state for reproducibility.
    """

    def __init__(
        self,
        max_tree_depth: int = 6,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        min_location_spacing: float = 0.01,
        random_state: int | None = 51,
    ):
        self.max_tree_depth = max_tree_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_location_spacing = min_location_spacing
        self.random_state = random_state
        self.min_samples_floor = min_samples_split

    def _ordered_candidates(
        self, x_valid: np.ndarray, y_valid: np.ndarray, task: Task
    ) -> tuple[list[float], object]:
        if task == "regression":
            tree = DecisionTreeRegressor(
                max_depth=self.max_tree_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
        else:
            tree = DecisionTreeClassifier(
                max_depth=self.max_tree_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )

        tree.fit(x_valid, y_valid)
        split_points = self._extract_split_points(tree, x_valid)
        return split_points, tree

    def _trim_over_max(self, points: list[float], context: object, max_count: int) -> list[float]:
        return self._select_top_locations(points, context, max_count)

    def _extract_split_points(self, tree, x: np.ndarray) -> list[float]:
        """Collect in-range split thresholds from a fitted decision tree."""
        tree_structure = tree.tree_
        split_points = []

        x_min, x_max = float(x.min()), float(x.max())

        for node_id in range(tree_structure.node_count):
            is_split = tree_structure.children_left[node_id] != tree_structure.children_right[node_id]
            if not is_split:
                continue
            if tree_structure.feature[node_id] != 0:
                continue
            threshold = tree_structure.threshold[node_id]
            if x_min < threshold < x_max:
                split_points.append(threshold)

        return sorted(set(split_points))

    def _select_top_locations(self, candidates: list[float], tree, max_count: int) -> list[float]:
        """Keep the locations whose splits reduce impurity the most."""
        if len(candidates) <= max_count:
            return candidates

        tree_structure = tree.tree_
        split_importance = {}

        for node_id in range(tree_structure.node_count):
            is_split = tree_structure.children_left[node_id] != tree_structure.children_right[node_id]
            if not is_split:
                continue

            threshold = tree_structure.threshold[node_id]
            if threshold not in candidates:
                continue

            n_samples = tree_structure.n_node_samples[node_id]
            impurity = tree_structure.impurity[node_id]

            left_child = tree_structure.children_left[node_id]
            right_child = tree_structure.children_right[node_id]
            n_left = tree_structure.n_node_samples[left_child]
            n_right = tree_structure.n_node_samples[right_child]
            impurity_left = tree_structure.impurity[left_child]
            impurity_right = tree_structure.impurity[right_child]

            split_importance[threshold] = n_samples * impurity - (n_left * impurity_left + n_right * impurity_right)

        top = sorted(split_importance, key=lambda k: split_importance[k], reverse=True)[:max_count]
        return sorted(top)


class LightGBMLocationSelector(BaseLocationSelector):
    """Select locations from the split points of a LightGBM ensemble.

    A gradient boosted ensemble is fitted to the feature against the target, and
    split thresholds are ranked by their cumulative gain across all trees. This
    tends to find informative locations that a single tree can miss.

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
        Minimum samples in a leaf. Also the small-sample floor below which
        quantile locations are returned.
    min_location_spacing : float, default=0.01
        Minimum distance between adjacent locations, as a fraction of the range.
    random_state : int or None, default=51
        Random state for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_child_samples: int = 20,
        min_location_spacing: float = 0.01,
        random_state: int | None = 51,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.min_location_spacing = min_location_spacing
        self.random_state = random_state
        self.min_samples_floor = min_child_samples

    @staticmethod
    def _import_lightgbm():
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise OptionalDependencyError(
                "LightGBMLocationSelector requires the optional 'lightgbm' dependency. "
                "Install it with: pip install pretab[knots]"
            ) from exc
        return lgb

    def _ordered_candidates(
        self, x_valid: np.ndarray, y_valid: np.ndarray, task: Task
    ) -> tuple[list[float], object]:
        lgb = self._import_lightgbm()

        params = {
            "objective": "regression" if task == "regression" else "binary",
            "metric": "rmse" if task == "regression" else "binary_logloss",
            "num_leaves": 2**self.max_depth,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_samples": self.min_child_samples,
            "verbose": -1,
            "random_state": self.random_state,
        }

        train_data = lgb.Dataset(x_valid, label=y_valid)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            callbacks=[lgb.log_evaluation(period=0)],
        )

        split_points = self._extract_split_points_with_gains(model, x_valid)
        if len(split_points) == 0:
            return [], None

        sorted_splits = sorted(split_points.items(), key=lambda item: item[1], reverse=True)
        thresholds = [split for split, _ in sorted_splits]
        return thresholds, None

    def _trim_over_max(self, points: list[float], context: object, max_count: int) -> list[float]:
        return points[:max_count]

    def _extract_split_points_with_gains(self, model, x: np.ndarray) -> dict[float, float]:
        """Collect split thresholds and their cumulative gains from a model."""
        x_min, x_max = float(x.min()), float(x.max())
        split_importance: dict[float, float] = {}

        model_dict = model.dump_model()
        for tree_info in model_dict["tree_info"]:
            self._traverse_tree(tree_info["tree_structure"], split_importance, x_min, x_max)

        return split_importance

    def _traverse_tree(self, node: dict, split_importance: dict, x_min: float, x_max: float):
        """Recursively accumulate split gains for the single feature."""
        if "split_feature" not in node:
            return

        if node["split_feature"] == 0:
            threshold = node["threshold"]
            gain = node.get("split_gain", 0.0)
            if x_min < threshold < x_max:
                split_importance[threshold] = split_importance.get(threshold, 0.0) + gain

        if "left_child" in node:
            self._traverse_tree(node["left_child"], split_importance, x_min, x_max)
        if "right_child" in node:
            self._traverse_tree(node["right_child"], split_importance, x_min, x_max)
