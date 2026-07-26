"""Piecewise Linear Encoding (PLE) transformer.

Encodes numerical features into a piecewise linear representation using bin
edges from a target-aware location selector (CART by default, optionally
LightGBM). The encoding is fully vectorized, so there is no ``eval`` of generated
strings and no regular-expression parsing of split conditions.
"""

from typing import ClassVar, Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ...core.adaptive import AdaptiveResolutionMixin
from ...core.parameters import UNSET, AliasResolverMixin
from ...core.representation import RepresentationSpecMixin
from ...core.supervised import warn_target_leakage
from ...exceptions import (
    IncompatibleParamsError,
    InvalidParamError,
    PretabDataError,
)
from ...placement.adapters import PLEPlacementAdapter


class PLETransformer(
    RepresentationSpecMixin, AdaptiveResolutionMixin, AliasResolverMixin, TransformerMixin, BaseEstimator
):
    """Piecewise Linear Encoding (PLE) transformer for numerical features.

    Each feature is discretized by a target-aware location selector (``"cart"``
    by default, optionally ``"lightgbm"``) whose split thresholds define the bin
    edges. Every input feature is expanded into one column per bin. Within the
    bin that a value falls into, the value is encoded linearly;
    all lower bins are set to ``1`` and all higher bins to ``0``. This preserves
    an ordinal, piecewise linear signal that downstream models can exploit while
    keeping the representation interpretable.

    Parameters
    ----------
    output_dim : int, default=6
        Maximum number of bins per feature, i.e. an upper bound on the number of
        output columns produced for each feature. This caps the number of leaf
        nodes (and therefore thresholds) the decision tree may produce. The
        actual number of output columns is data-dependent and may be smaller
        (see Notes).
    placement_strategy : {"cart", "lightgbm"}, default="cart"
        Which target-aware location selector places the bin thresholds. PLE is
        inherently target-aware, so only the supervised selectors apply.
        ``"cart"`` fits a single decision tree (always available); ``"lightgbm"``
        fits a gradient-boosted ensemble and requires the optional ``lightgbm``
        dependency (``pip install pretab[lightgbm]``).
    task : {"regression", "classification"}, default="regression"
        Whether to fit a ``DecisionTreeRegressor`` or ``DecisionTreeClassifier``
        to locate the split thresholds.
    adaptive : bool, default=False
        If True, allow the number of bins per feature to be data-driven, bounded
        by ``min_output_dim`` and ``max_output_dim``. If False, every feature is
        capped by ``output_dim``.
    min_output_dim : int or None, default=None
        Minimum number of bins per feature when ``adaptive=True``.
    max_output_dim : int or None, default=None
        Maximum number of bins per feature when ``adaptive=True``.
    random_state : int or None, default=51
        Random state for reproducible tree fitting.
    max_depth : int or None, default=None
        Maximum depth of the decision tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required at a leaf node.

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

    Notes
    -----
    The number of output columns per feature equals ``len(thresholds) + 1`` and
    is therefore data-dependent: a feature whose tree finds fewer splits will
    expand into fewer columns than a feature with many splits. ``output_dim`` is
    an upper bound (bin cap), not an exact width; this is a documented exception
    to the exact-width contract that the fixed-basis families follow.

    PLE requires finite input: NaN values raise an error. Missing-value handling
    is the responsibility of an upstream imputation step (for example the
    ``Preprocessor`` imputation parameters), not of this transformer.

    The ``max_depth`` / ``min_samples_split`` / ``min_samples_leaf`` parameters
    are retained for backward-compatible construction but no longer affect
    threshold placement: the ``placement_strategy`` selector fits its own model
    with its own settings. They are slated for reconciliation in a later cleanup.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=51)
    >>> ple = PLETransformer(output_dim=10, task="regression")
    >>> X_transformed = ple.fit_transform(X, y)
    """

    _param_aliases: ClassVar[dict[str, str]] = {}
    _representation_family = "piecewise_linear"
    _representation_component_kind = "interval"
    _representation_supervision = "supervised"
    _representation_local_support = True

    def __init__(
        self,
        output_dim=UNSET,
        placement_strategy: str = "cart",
        task: Literal["regression", "classification"] = "regression",
        adaptive: bool = False,
        min_output_dim=UNSET,
        max_output_dim=UNSET,
        random_state: int | None = 51,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        self.output_dim = output_dim
        self.placement_strategy = placement_strategy
        self.task = task
        self.adaptive = adaptive
        self.min_output_dim = min_output_dim
        self.max_output_dim = max_output_dim
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def __sklearn_tags__(self):
        """Declare the required-target tag; PLE requires finite input."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        tags.target_tags.required = True
        return tags

    def fit(self, X, y=None):
        """Fit the transformer by learning per-feature bin thresholds.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Must be finite; NaN values raise an error.
        y : array-like of shape (n_samples,)
            Target values used to grow the per-feature decision trees. PLE is
            always target-aware, so ``y`` is required.

        Returns
        -------
        self : PLETransformer
            The fitted transformer.
        """
        warn_target_leakage(self, y)
        if y is None:
            raise IncompatibleParamsError(
                "PLETransformer is always target-aware and requires y at fit time; got y=None."
            )

        X = check_array(
            X,
            dtype=np.float64,
            ensure_2d=True,
            ensure_all_finite=True,
        )
        y = np.asarray(y).ravel()

        if len(X) != len(y):
            raise PretabDataError(f"X and y must have same length. Got {len(X)} and {len(y)}")

        self.n_features_in_ = X.shape[1]
        self.thresholds_ = []
        self.n_bins_per_feature_ = []

        n_bins = self._resolve_param("output_dim", default=6)
        min_bins_req = self._resolve_param("min_output_dim", default=None)
        max_bins_req = self._resolve_param("max_output_dim", default=None)
        min_bins, max_bins = self._resolve_bin_bounds(n_bins, min_bins_req, max_bins_req)

        if self.task not in ("regression", "classification"):
            raise InvalidParamError(f"Unsupported task: {self.task}. Use 'regression' or 'classification'.")

        if self.placement_strategy not in ("cart", "lightgbm"):
            raise InvalidParamError(
                f"Invalid placement_strategy. Choose 'cart' or 'lightgbm'. Got {self.placement_strategy!r}."
            )

        # Thresholds come from the placement subsystem's target-aware adapter
        # (CART by default, optionally LightGBM): split points spaced out and
        # ranked by impurity / gain, then trimmed / topped up to fit the bin-count
        # window. Each feature produces ``len(thresholds) + 1`` bins, so we ask for
        # one fewer location than bins: the non-adaptive window pins the count to
        # exactly ``output_dim`` bins, adaptive clamps it into ``[min, max]``.
        adapter = PLEPlacementAdapter(
            placement_strategy=self.placement_strategy,
            task=self.task,
            random_state=self.random_state,
        )
        min_thresholds = max(0, min_bins - 1)
        max_thresholds = max(0, max_bins - 1)

        for i in range(X.shape[1]):
            thresholds = adapter.get_thresholds(X[:, i], y, min_thresholds, max_thresholds)

            self.thresholds_.append(thresholds)
            self.n_bins_per_feature_.append(len(thresholds) + 1)

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

        X = check_array(
            X,
            dtype=np.float64,
            ensure_2d=True,
            ensure_all_finite=True,
        )

        if X.shape[1] != self.n_features_in_:
            raise PretabDataError(
                f"X has {X.shape[1]} features, but {type(self).__name__} "
                f"is expecting {self.n_features_in_} features as input."
            )

        all_transformed = []

        for col in range(X.shape[1]):
            feature = X[:, col].copy()
            thresholds = self.thresholds_[col]

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
        return self._resolve_output_bounds(n_bins, min_bins_req, max_bins_req, floor=1)
