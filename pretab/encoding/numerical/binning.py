from typing import ClassVar

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...core.parameters import UNSET, AliasResolverMixin
from ...core.representation import RepresentationSpecMixin
from ...exceptions import InsufficientSamplesError, InvalidParamError, PretabDataError

_VALID_ENCODINGS = ("ordinal", "onehot", "soft")
_VALID_STRATEGIES = ("uniform", "quantile")


class NumericBinningTransformer(RepresentationSpecMixin, AliasResolverMixin, TransformerMixin, BaseEstimator):
    """Stateful binning transformer for numerical features.

    The bin edges are learned once in :meth:`fit` and reused at
    :meth:`transform` time, so the discretization never leaks information from
    the data being transformed. Edges are placed with equal width
    (``placement_strategy="uniform"``) or on empirical quantiles
    (``placement_strategy="quantile"``); an explicit array of edges can also be
    passed through ``output_dim``. Each feature is binned independently.

    Parameters
    ----------
    output_dim : int or array-like
        If an int, the number of bins to place from the fitted data range /
        quantiles. If array-like, the bin edges to use directly
        (``placement_strategy`` is then ignored). ``output_dim`` is the number of
        *bins*, not the number of output columns: with ``encode="ordinal"`` each
        feature emits a single column, whereas ``encode="onehot"`` /
        ``encode="soft"`` emit one column per bin.
    encode : {"ordinal", "onehot", "soft"}, default="ordinal"
        How to represent the bin assignment:

        * ``"ordinal"`` -- a single integer column of bin indices per feature.
        * ``"onehot"`` -- a 0/1 indicator column per bin.
        * ``"soft"`` -- triangular membership to the two nearest bin centers;
          the per-row weights are non-negative and sum to 1 across the bins.
    placement_strategy : {"uniform", "quantile"}, default="uniform"
        How to place the learned bin edges when ``output_dim`` is an int:
        equal-width (``"uniform"``) or equal-frequency (``"quantile"``). Ignored
        when ``output_dim`` is an explicit array of edges.

    Attributes
    ----------
    n_features_in_ : int
        The number of input features seen during :meth:`fit`.
    bin_edges_ : list of ndarray
        The sorted, de-duplicated bin edges learned per feature.
    n_bins_ : list of int
        The number of bins per feature (``len(edges) - 1``).
    total_output_dim_ : int
        Total number of output columns. Equal to ``n_features_in_`` for
        ``encode="ordinal"``; otherwise the sum of ``n_bins_``.

    Notes
    -----
    - The input must be numeric: string / categorical data raises a
      :class:`~pretab.exceptions.PretabDataError`. Encode such columns with a
      categorical method (e.g. ``"int"`` or ``"one-hot"``) before binning. Values
      seen at transform time that fall outside the fitted range are clamped into
      the outer bins.
    - Unlike scikit-learn's ``KBinsDiscretizer``, this transformer can also accept
      explicit user-defined bin edges, which is useful when bins should follow
      domain-specific boundaries rather than being learned from the data.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import NumericBinningTransformer
    >>> X = np.linspace(0, 1, 10).reshape(-1, 1)
    >>> NumericBinningTransformer(output_dim=4).fit_transform(X).shape
    (10, 1)
    >>> NumericBinningTransformer(output_dim=4, encode="onehot").fit_transform(X).shape
    (10, 4)
    """

    _param_aliases: ClassVar[dict[str, str]] = {}
    _representation_family = "binning"
    _representation_component_kind = "interval"
    _representation_local_support = True

    def __init__(self, output_dim=UNSET, encode="ordinal", placement_strategy="uniform"):
        # An int yields learned bins; an array-like is used as fixed bin edges.
        self.output_dim = output_dim
        self.encode = encode
        self.placement_strategy = placement_strategy

    def _check_array(self, X, *, reset):
        """Validate ``X`` is a 2D numeric array and (re)set the feature count."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise PretabDataError("Input must be a 2D array of shape (n_samples, n_features).")
        if not np.issubdtype(X.dtype, np.number):
            try:
                X = X.astype(np.float64)
            except (ValueError, TypeError) as exc:
                raise PretabDataError(
                    "NumericBinningTransformer requires numeric input: it bins continuous "
                    "values into intervals and cannot process string/categorical data. "
                    "Encode string columns with a categorical method (e.g. 'int' or "
                    "'one-hot') before binning."
                ) from exc
        else:
            X = X.astype(np.float64, copy=False)
        if np.isinf(X).any():
            raise PretabDataError(
                "NumericBinningTransformer received infinite values, which cannot be "
                "placed into finite bins. Clean or clip the input before binning."
            )
        if np.isnan(X).any():
            raise PretabDataError(
                "NumericBinningTransformer received missing values (NaN), which cannot be "
                "binned. Impute missing values (e.g. via the Preprocessor pipeline or a "
                "SimpleImputer) before binning."
            )
        if reset:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise PretabDataError(
                f"Input has {X.shape[1]} features, but NumericBinningTransformer was fitted with {self.n_features_in_}."
            )
        return X

    def _resolve_edges(self, column, bins_spec):
        """Return the sorted, de-duplicated bin edges for a single feature."""
        if isinstance(bins_spec, (int, np.integer)):
            n_bins = int(bins_spec)
            if n_bins < 1:
                raise InvalidParamError("output_dim must be a positive integer bin count.")
            lo = float(np.min(column))
            hi = float(np.max(column))
            if self.placement_strategy == "uniform":
                edges = np.linspace(lo, hi, n_bins + 1)
            else:  # quantile
                edges = np.quantile(column, np.linspace(0.0, 1.0, n_bins + 1))
        else:
            edges = np.asarray(bins_spec, dtype=np.float64).ravel()
            if edges.size < 2:
                raise InvalidParamError("Explicit bin edges must contain at least two values.")
        edges = np.unique(edges)  # sorted + de-duplicated
        if edges.size < 2:
            # Constant feature (or fully-tied quantiles): fall back to one bin.
            edges = np.array([edges[0], edges[0] + 1.0])
        return edges

    def fit(self, X, y=None):
        """Learn the per-feature bin edges.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = self._check_array(X, reset=True)
        if X.shape[0] <= 2:
            raise InsufficientSamplesError("Input must have more than 2 observations.")
        if self.encode not in _VALID_ENCODINGS:
            raise InvalidParamError(f"encode must be one of {_VALID_ENCODINGS}; got {self.encode!r}.")
        if self.placement_strategy not in _VALID_STRATEGIES:
            raise InvalidParamError(
                f"placement_strategy must be one of {_VALID_STRATEGIES}; got {self.placement_strategy!r}."
            )

        bins_spec = self._resolve_param("output_dim", default=UNSET)
        if bins_spec is UNSET:
            raise InvalidParamError("NumericBinningTransformer requires 'output_dim'.")

        self.bin_edges_ = [self._resolve_edges(X[:, j], bins_spec) for j in range(X.shape[1])]
        self.n_bins_ = [edges.size - 1 for edges in self.bin_edges_]
        self.total_output_dim_ = self.n_features_in_ if self.encode == "ordinal" else int(sum(self.n_bins_))
        return self

    @staticmethod
    def _bin_indices(column, edges):
        """Assign each value to a bin using ``(a, b]`` intervals with a closed right edge."""
        idx = np.searchsorted(edges, column, side="left") - 1
        return np.clip(idx, 0, edges.size - 2).astype(int)

    @staticmethod
    def _soft_membership(column, edges):
        """Return triangular membership weights to the two nearest bin centers."""
        centers = 0.5 * (edges[:-1] + edges[1:])
        n_bins = centers.size
        col = np.clip(column, centers[0], centers[-1])
        weights = np.zeros((col.size, n_bins), dtype=np.float64)
        if n_bins == 1:
            weights[:, 0] = 1.0
            return weights
        right = np.clip(np.searchsorted(centers, col, side="left"), 1, n_bins - 1)
        left = right - 1
        span = centers[right] - centers[left]
        frac = np.where(span > 0, (col - centers[left]) / span, 0.0)
        rows = np.arange(col.size)
        weights[rows, left] = 1.0 - frac
        weights[rows, right] += frac
        return weights

    def transform(self, X):
        """Bin the data using the edges learned during :meth:`fit`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        X_binned : ndarray of shape (n_samples, total_output_dim_)
            The encoded bin assignments.
        """
        check_is_fitted(self, "bin_edges_")
        X = self._check_array(X, reset=False)
        blocks = []
        for j in range(X.shape[1]):
            edges = self.bin_edges_[j]
            n_bins = self.n_bins_[j]
            if self.encode == "ordinal":
                blocks.append(self._bin_indices(X[:, j], edges).reshape(-1, 1))
            elif self.encode == "onehot":
                onehot = np.zeros((X.shape[0], n_bins), dtype=np.float64)
                onehot[np.arange(X.shape[0]), self._bin_indices(X[:, j], edges)] = 1.0
                blocks.append(onehot)
            else:  # soft
                blocks.append(self._soft_membership(X[:, j], edges))
        return np.hstack(blocks)

    def get_feature_names_out(self, input_features=None):
        """Return the names of the transformed features.

        Parameters
        ----------
        input_features : list of str or None
            The names of the input features. When ``None``, names of the form
            ``x0, x1, ...`` are generated.

        Returns
        -------
        feature_names : ndarray of shape (total_output_dim_,)
            One name per input feature for ``encode="ordinal"``; otherwise one
            ``"{feature}_bin{k}"`` name per bin.
        """
        check_is_fitted(self, "n_features_in_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        if self.encode == "ordinal":
            return np.asarray(input_features, dtype=object)
        check_is_fitted(self, "n_bins_")
        names = []
        for feature, n_bins in zip(input_features, self.n_bins_, strict=False):
            names.extend(f"{feature}_bin{k}" for k in range(n_bins))
        return np.asarray(names, dtype=object)
