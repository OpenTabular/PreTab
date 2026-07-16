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
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class BaseKnotSelector(ABC):
    """Abstract base class for knot selection strategies.

    Subclasses implement :meth:`get_knot_locations`, returning the internal knot
    positions (boundary knots are added later by the spline transformer). Common
    helpers for spacing, quantile fallbacks, and basis-to-knot conversion live
    here so the concrete selectors stay small.

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
            return max(0, n_basis - self.degree - 1)
        raise ValueError(f"Unknown spline_type: {self.spline_type}")

    def _enforce_spacing(self, split_points: list[float], X: np.ndarray) -> list[float]:
        """Drop knots that sit closer than ``min_knot_spacing`` of the range."""
        if len(split_points) <= 1:
            return split_points

        x_range = float(X.max() - X.min())
        min_distance = self.min_knot_spacing * x_range

        spaced_knots = [split_points[0]]
        for knot in split_points[1:]:
            if knot - spaced_knots[-1] >= min_distance:
                spaced_knots.append(knot)

        return spaced_knots

    def _get_quantile_knots(self, X: np.ndarray, n_knots: int) -> np.ndarray:
        """Return quantile-spaced knots, used as a fallback."""
        if n_knots <= 0:
            return np.array([])
        quantiles = np.linspace(0, 1, n_knots + 2)[1:-1]
        return np.quantile(X, quantiles)

    def _supplement_knots(self, existing_knots: list[float], X: np.ndarray, target_count: int) -> list[float]:
        """Top up an under-filled knot set with quantile knots."""
        if target_count - len(existing_knots) <= 0:
            return existing_knots

        quantile_knots = self._get_quantile_knots(X, target_count)
        all_knots = set(existing_knots) | set(quantile_knots.tolist())
        return sorted(all_knots)[:target_count]


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

    def get_knot_locations(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        task: Literal["regression", "classification"] | None = "regression",
    ) -> np.ndarray:
        if y is None:
            raise ValueError("CARTKnotSelector requires y to select knots.")
        task = task or "regression"

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y).ravel()

        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        if len(X_valid) < self.min_samples_split:
            return self._get_quantile_knots(X_valid, self.min_knots)

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

        tree.fit(X_valid, y_valid)

        split_points = self._extract_split_points(tree, X_valid)
        knot_locations = self._enforce_spacing(split_points, X_valid)

        if len(knot_locations) < self.min_knots:
            knot_locations = self._supplement_knots(knot_locations, X_valid, self.min_knots)
        elif len(knot_locations) > self.max_knots:
            knot_locations = self._select_top_knots(knot_locations, tree, self.max_knots)

        return np.array(sorted(knot_locations))

    def _extract_split_points(self, tree, X: np.ndarray) -> list[float]:
        """Collect in-range split thresholds from a fitted decision tree."""
        tree_structure = tree.tree_
        split_points = []

        x_min, x_max = float(X.min()), float(X.max())

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

    def _select_top_knots(self, knot_candidates: list[float], tree, max_count: int) -> list[float]:
        """Keep the knots whose splits reduce impurity the most."""
        if len(knot_candidates) <= max_count:
            return knot_candidates

        tree_structure = tree.tree_
        split_importance = {}

        for node_id in range(tree_structure.node_count):
            is_split = tree_structure.children_left[node_id] != tree_structure.children_right[node_id]
            if not is_split:
                continue

            threshold = tree_structure.threshold[node_id]
            if threshold not in knot_candidates:
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

        top_knots = sorted(split_importance, key=lambda k: split_importance[k], reverse=True)[:max_count]
        return sorted(top_knots)


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

    @staticmethod
    def _import_lightgbm():
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "LightGBMKnotSelector requires the optional 'lightgbm' dependency. "
                "Install it with: pip install pretab[knots]"
            ) from exc
        return lgb

    def get_knot_locations(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        task: Literal["regression", "classification"] | None = "regression",
    ) -> np.ndarray:
        if y is None:
            raise ValueError("LightGBMKnotSelector requires y to select knots.")
        task = task or "regression"
        lgb = self._import_lightgbm()

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y).ravel()

        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        if len(X_valid) < self.min_child_samples:
            return self._get_quantile_knots(X_valid, self.min_knots)

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

        train_data = lgb.Dataset(X_valid, label=y_valid)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            callbacks=[lgb.log_evaluation(period=0)],
        )

        split_points = self._extract_split_points_with_gains(model, X_valid)
        if len(split_points) == 0:
            return self._get_quantile_knots(X_valid, self.min_knots)

        sorted_splits = sorted(split_points.items(), key=lambda item: item[1], reverse=True)
        thresholds = [split for split, _ in sorted_splits]

        knot_locations = self._enforce_spacing(thresholds, X_valid)

        if len(knot_locations) < self.min_knots:
            knot_locations = self._supplement_knots(knot_locations, X_valid, self.min_knots)
        elif len(knot_locations) > self.max_knots:
            knot_locations = knot_locations[: self.max_knots]

        return np.array(sorted(knot_locations))

    def _extract_split_points_with_gains(self, model, X: np.ndarray) -> dict[float, float]:
        """Collect split thresholds and their cumulative gains from a model."""
        x_min, x_max = float(X.min()), float(X.max())
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
