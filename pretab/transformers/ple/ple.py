"""Piecewise Linear Encoding (PLE) transformer.

Encodes numerical features into a piecewise linear representation using bin
edges learned from a decision tree. Thresholds are read directly from the fitted
tree structure and the encoding is fully vectorized, so there is no ``eval`` of
generated strings and no regular-expression parsing of split conditions.
"""

import warnings
from typing import ClassVar, Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_array, check_is_fitted

from ...core.params import UNSET, AliasResolverMixin


def extract_thresholds_from_tree(tree) -> np.ndarray:
    """Extract split thresholds directly from a fitted decision tree.

    Reads the threshold values stored on the tree's internal (split) nodes.
    Leaf nodes carry a sentinel feature index of ``-2`` and are ignored.

    Parameters
    ----------
    tree : DecisionTreeRegressor or DecisionTreeClassifier
        A fitted decision tree.

    Returns
    -------
    thresholds : ndarray
        Sorted, unique threshold values used by the split nodes.
    """
    tree_ = tree.tree_

    # Split nodes have a valid feature index (>= 0); leaf nodes use -2.
    split_mask = tree_.feature >= 0
    thresholds = tree_.threshold[split_mask]

    return np.unique(thresholds)


class PLETransformer(AliasResolverMixin, BaseEstimator, TransformerMixin):
    """Piecewise Linear Encoding (PLE) transformer for numerical features.

    Each feature is discretized with a decision tree, and the split thresholds
    define the bin edges. Every input feature is expanded into one column per
    bin. Within the bin that a value falls into, the value is encoded linearly;
    all lower bins are set to ``1`` and all higher bins to ``0``. This preserves
    an ordinal, piecewise linear signal that downstream models can exploit while
    keeping the representation interpretable.

    Parameters
    ----------
    output_dim : int, default=20
        Maximum number of bins per feature, i.e. an upper bound on the number of
        output columns produced for each feature. This caps the number of leaf
        nodes (and therefore thresholds) the decision tree may produce. The
        actual number of output columns is data-dependent and may be smaller
        (see Notes).
    task : {"regression", "classification"}, default="regression"
        Whether to fit a ``DecisionTreeRegressor`` or ``DecisionTreeClassifier``
        to locate the split thresholds.
    max_depth : int or None, default=None
        Maximum depth of the decision tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required at a leaf node.
    random_state : int or None, default=51
        Random state for reproducible tree fitting.
    handle_missing : {"error", "median"}, default="median"
        How to handle NaN values.

        - ``"error"``: raise an error when a NaN is encountered.
        - ``"median"``: drop NaN rows during ``fit`` and, at ``transform`` time,
          replace NaN with the median of that feature's thresholds (or ``0`` when
          the feature produced no thresholds).
    adaptive : bool, default=False
        If True, allow the number of bins per feature to be data-driven, bounded
        by ``min_output_dim`` and ``max_output_dim``. If False, every feature is
        capped by ``output_dim``.
    min_output_dim : int or None, default=None
        Minimum number of bins per feature when ``adaptive=True``.
    max_output_dim : int or None, default=None
        Maximum number of bins per feature when ``adaptive=True``.

    Attributes
    ----------
    thresholds_ : list of ndarray
        Sorted threshold values for each feature.
    n_features_in_ : int
        Number of features seen during ``fit``.
    n_bins_per_feature_ : list of int
        Number of output bins produced for each feature (``len(thresholds) + 1``).
    total_output_dim_ : int
        Total number of output columns across all features (fitted); equals
        ``sum(n_bins_per_feature_)``.
    fill_values_ : list of float
        Per-feature fill value used to replace NaN during ``transform``.

    Notes
    -----
    The number of output columns per feature equals ``len(thresholds) + 1`` and
    is therefore data-dependent: a feature whose tree finds fewer splits will
    expand into fewer columns than a feature with many splits. ``output_dim`` is
    an upper bound (bin cap), not an exact width; this is a documented exception
    to the exact-width contract that the fixed-basis families follow.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=51)
    >>> ple = PLETransformer(output_dim=10, task="regression")
    >>> X_transformed = ple.fit_transform(X, y)
    """

    _param_aliases: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        output_dim=UNSET,
        task: Literal["regression", "classification"] = "regression",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int | None = 51,
        handle_missing: Literal["error", "median"] = "median",
        adaptive: bool = False,
        min_output_dim=UNSET,
        max_output_dim=UNSET,
    ):
        self.output_dim = output_dim
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.handle_missing = handle_missing
        self.adaptive = adaptive
        self.min_output_dim = min_output_dim
        self.max_output_dim = max_output_dim

    def fit(self, X, y):
        """Fit the transformer by learning per-feature bin thresholds.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values used to grow the per-feature decision trees.

        Returns
        -------
        self : PLETransformer
            The fitted transformer.
        """
        finite_policy: Literal["allow-nan"] | bool = "allow-nan" if self.handle_missing == "median" else True
        X = check_array(
            X,
            dtype=np.float64,
            ensure_2d=True,
            ensure_all_finite=finite_policy,
        )
        y = np.asarray(y).ravel()

        if len(X) != len(y):
            raise ValueError(f"X and y must have same length. Got {len(X)} and {len(y)}")

        if self.handle_missing == "median":
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            if not valid_mask.all():
                n_removed = int((~valid_mask).sum())
                warnings.warn(
                    f"Removed {n_removed} samples with NaN values during fit",
                    UserWarning,
                    stacklevel=2,
                )
                X = X[valid_mask]
                y = y[valid_mask]

        if len(X) == 0:
            raise ValueError("All samples contain NaN values")

        self.n_features_in_ = X.shape[1]
        self.thresholds_ = []
        self.n_bins_per_feature_ = []
        self.fill_values_ = []

        n_bins = self._resolve_param("output_dim", default=20)
        min_bins_req = self._resolve_param("min_output_dim", default=None)
        max_bins_req = self._resolve_param("max_output_dim", default=None)
        min_bins, max_bins = self._resolve_bin_bounds(n_bins, min_bins_req, max_bins_req)

        for i in range(X.shape[1]):
            x_feat = X[:, [i]]

            tree_max_leaf_nodes = max_bins if self.adaptive else n_bins

            if self.task == "regression":
                dt = DecisionTreeRegressor(
                    max_leaf_nodes=tree_max_leaf_nodes,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                )
            elif self.task == "classification":
                dt = DecisionTreeClassifier(
                    max_leaf_nodes=tree_max_leaf_nodes,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                )
            else:
                raise ValueError(f"Unsupported task: {self.task}. Use 'regression' or 'classification'.")

            dt.fit(x_feat, y)

            thresholds = extract_thresholds_from_tree(dt)
            thresholds = self._adjust_thresholds(x_feat.ravel(), thresholds, min_bins=min_bins, max_bins=max_bins)

            self.thresholds_.append(thresholds)
            self.n_bins_per_feature_.append(len(thresholds) + 1)

            if len(thresholds) > 0:
                self.fill_values_.append(float(np.median(thresholds)))
            else:
                self.fill_values_.append(0.0)

        self.total_output_dim_ = int(sum(self.n_bins_per_feature_))

        return self

    def transform(self, X):
        """Encode features with vectorized piecewise linear binning.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features_out)
            The piecewise linear encoding, cast to ``float32``.
        """
        check_is_fitted(self, ["thresholds_", "n_features_in_"])

        finite_policy: Literal["allow-nan"] | bool = "allow-nan" if self.handle_missing == "median" else True
        X = check_array(
            X,
            dtype=np.float64,
            ensure_2d=True,
            ensure_all_finite=finite_policy,
        )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but PLETransformer was fitted with {self.n_features_in_} features."
            )

        all_transformed = []

        for col in range(X.shape[1]):
            feature = X[:, col].copy()
            thresholds = self.thresholds_[col]

            nan_mask = np.isnan(feature)
            if nan_mask.any():
                if self.handle_missing == "error":
                    raise ValueError(f"Feature {col} contains NaN values")
                feature[nan_mask] = self.fill_values_[col]

            ple_encoded = self._apply_piecewise_linear_vectorized(feature, thresholds)
            all_transformed.append(ple_encoded)

        return np.hstack(all_transformed).astype(np.float32)

    def _apply_piecewise_linear_vectorized(self, feature: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """Apply the vectorized piecewise linear encoding for one feature.

        The encoding for each sample works as follows:

        - First bin (below ``thresholds[0]``): the raw value.
        - Middle bins: the value normalized to ``[0, 1]`` within the bin.
        - Last bin (above ``thresholds[-1]``): the raw value.

        Every bin below the active bin is filled with ``1.0``; bins above it stay
        ``0.0``.
        """
        n_samples = len(feature)
        n_bins = len(thresholds) + 1

        if len(thresholds) == 0:
            return feature.reshape(-1, 1).astype(np.float32)

        ple_encoded = np.zeros((n_samples, n_bins), dtype=np.float32)

        bin_indices = np.searchsorted(thresholds, feature, side="right")

        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx

            if not mask.any():
                continue

            values = feature[mask]

            if bin_idx == 0:
                # First bin: raw value, no lower bins to fill.
                ple_encoded[mask, bin_idx] = values

            elif bin_idx == n_bins - 1:
                # Last bin: raw value, all lower bins set to 1.
                ple_encoded[mask, bin_idx] = values
                ple_encoded[mask, :bin_idx] = 1.0

            else:
                # Middle bin: normalize the value to [0, 1] within the bin.
                lower_threshold = thresholds[bin_idx - 1]
                upper_threshold = thresholds[bin_idx]
                bin_width = upper_threshold - lower_threshold

                if bin_width > 1e-10:
                    ple_encoded[mask, bin_idx] = (values - lower_threshold) / bin_width
                else:
                    ple_encoded[mask, bin_idx] = 0.5

                ple_encoded[mask, :bin_idx] = 1.0

        return ple_encoded

    def get_n_features_out(self) -> int:
        """Return the total number of output columns across all features."""
        check_is_fitted(self, ["n_bins_per_feature_"])
        return int(sum(self.n_bins_per_feature_))

    def get_feature_names_out(self, input_features=None):
        """Return output feature names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names. If None, generic names ``x0``, ``x1`` ... are used.

        Returns
        -------
        feature_names_out : ndarray of str
            One name per output column, formatted ``{name}_ple_piece{j}``.
        """
        check_is_fitted(self, ["thresholds_", "n_features_in_"])

        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]

        feature_names_out = []
        for name, n_bins in zip(input_features, self.n_bins_per_feature_, strict=False):
            for j in range(n_bins):
                feature_names_out.append(f"{name}_ple_piece{j}")

        return np.array(feature_names_out)

    def _resolve_bin_bounds(self, n_bins: int, min_bins_req, max_bins_req) -> tuple[int, int]:
        if not self.adaptive:
            if min_bins_req is not None and n_bins < min_bins_req:
                raise ValueError("output_dim must be >= min_output_dim when adaptive=False")
            if max_bins_req is not None and n_bins > max_bins_req:
                raise ValueError("output_dim must be <= max_output_dim when adaptive=False")
            min_bins = n_bins
            max_bins = n_bins
        else:
            min_bins = min_bins_req if min_bins_req is not None else n_bins
            max_bins = max_bins_req if max_bins_req is not None else n_bins

        if min_bins < 1:
            raise ValueError(f"min_output_dim must be >= 1, got {min_bins}")
        if max_bins < 1:
            raise ValueError(f"max_output_dim must be >= 1, got {max_bins}")
        if min_bins > max_bins:
            raise ValueError("min_output_dim must be <= max_output_dim")

        return min_bins, max_bins

    def _adjust_thresholds(
        self, feature: np.ndarray, thresholds: np.ndarray, min_bins: int, max_bins: int
    ) -> np.ndarray:
        thresholds = np.sort(np.asarray(thresholds))
        min_thresholds = max(0, min_bins - 1)
        max_thresholds = max(0, max_bins - 1)

        if len(thresholds) < min_thresholds:
            thresholds = self._supplement_thresholds(feature, thresholds, min_thresholds)
        if len(thresholds) > max_thresholds:
            thresholds = self._select_thresholds(thresholds, max_thresholds)

        return np.sort(thresholds)

    def _supplement_thresholds(self, feature: np.ndarray, thresholds: np.ndarray, target_count: int) -> np.ndarray:
        if target_count <= len(thresholds):
            return thresholds

        candidates = [thresholds]
        if target_count > 0:
            quantiles = np.linspace(0, 100, target_count + 2)[1:-1]
            candidates.append(np.percentile(feature, quantiles))
            candidates.append(np.linspace(feature.min(), feature.max(), target_count + 2)[1:-1])

        combined = np.sort(np.unique(np.concatenate(candidates)))
        if len(combined) < target_count:
            combined = np.linspace(feature.min(), feature.max(), target_count + 2)[1:-1]

        return self._select_thresholds(np.asarray(combined), target_count)

    @staticmethod
    def _select_thresholds(thresholds: np.ndarray, count: int) -> np.ndarray:
        if len(thresholds) <= count:
            return thresholds
        idx = np.linspace(0, len(thresholds) - 1, count).round().astype(int)
        return thresholds[idx]
