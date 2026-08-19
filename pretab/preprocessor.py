import hashlib
import inspect
import json
import os
import time
import warnings
from typing import cast

import numpy as np
from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils._set_output import _get_output_config
from sklearn.utils.validation import check_is_fitted

from .compose.config import PreprocessorConfig
from .compose.factory import build_column_transformer
from .compose.feature_detection import detect_column_types, to_dataframe
from .compose.inspection import (
    build_feature_info,
    build_feature_lineage,
    build_transformer_summary,
    clean_feature_names,
    get_output_slices,
)
from .compose.output import compute_output_report, format_output, to_dataframe_output
from .compose.serialize import SCHEMA_VERSION, preprocessor_from_spec, preprocessor_to_spec
from .core.logging import configure_logging, get_logger
from .core.policy import RepresentationPolicy, apply_constant_policy
from .exceptions import (
    ConfigWarning,
    FrozenRepresentationError,
    OutputBudgetError,
    PretabDataError,
    PretabSerializationError,
    invalid_param_error,
)

logger = get_logger(__name__)

#: Named parameter bundles exposed through ``Preprocessor(preset=...)``. Each
#: preset supplies values only for the listed parameters; any parameter the caller
#: sets explicitly (i.e. away from its ``__init__`` default) overrides the preset.
PRESETS = {
    "standard": {
        "numerical_method": "ple",
        "categorical_method": "int",
        "output_dim": 7,
        "adaptive": False,
    },
    "expanded": {
        "numerical_method": "ple",
        "categorical_method": "one-hot",
        "output_dim": 16,
        "adaptive": False,
    },
    "adaptive": {
        "numerical_method": "ple",
        "categorical_method": "int",
        "adaptive": True,
        "min_output_dim": 5,
        "max_output_dim": 16,
    },
}


class Preprocessor(TransformerMixin, BaseEstimator):
    r"""
    Preprocessor class for automated tabular feature preprocessing using scikit-learn-compatible pipelines.

    This class provides a flexible interface for preprocessing tabular datasets containing numerical and
    categorical features. It automatically detects feature types, applies user-defined or default preprocessing
    strategies, and supports both dictionary and array-style outputs. It also supports integration with external
    embedding vectors.

    Features
    --------
    - Supports a wide range of preprocessing methods for numerical and categorical features.
    - Automatically detects feature types (numerical vs. categorical).
    - Compatible with both pandas DataFrames and NumPy arrays.
    - Handles external embedding arrays for models that require learned representations.
    - Returns either a dictionary of transformed feature blocks or a single NumPy array.
    - Fully compatible with scikit-learn transformers and pipelines.

    Parameters
    ----------
    numerical_method : str, default="ple"
        Preprocessing strategy applied to every numerical column unless overridden per feature.
        Common choices: ``"ple"`` (piecewise-linear encoding), ``"minmax"`` / ``"standardization"``
        / ``"robust"`` (scaling), ``"quantile"`` (rank transform), ``"rbf"`` / ``"relu"`` /
        ``"sigmoid"`` / ``"tanh"`` (feature maps), and ``"cubicspline"`` / ``"naturalspline"`` /
        ``"pspline"`` / ``"bspline"`` (spline bases). Pass ``None`` (resolved to ``"none"``) to leave
        numerical columns unchanged. See *Notes* for the complete list.
    categorical_method : str, default="int"
        Preprocessing strategy applied to every categorical column unless overridden per feature.
        Choices: ``"int"`` (contiguous integer codes), ``"one-hot"`` (dummy columns),
        ``"onehot_from_ordinal"`` (one-hot from an already integer-coded column; raises if the
        input is not already ordinal-encoded), ``"pretrained"`` (sentence-transformer language
        embeddings), and ``"custombin"`` (discretized bin codes). Pass ``None`` (resolved to
        ``"none"``) to leave categorical columns unchanged.
    feature_preprocessing : dict, optional
        Mapping of individual column names to a method, overriding the global ``numerical_method`` /
        ``categorical_method`` for those columns only, e.g.
        ``{"age": "cubicspline", "city": "pretrained"}``. Columns absent from the dict fall back to
        the global defaults.
    output_dim : int, default=7
        The single width knob shared by every numerical method: the number of non-bias output
        columns produced per input feature (bins for PLE/binning, centers for the feature maps,
        basis functions for the splines). The B/M/I splines clamp it into their supported
        ``[5, 50]`` range. Used as the fixed per-feature width when ``adaptive`` is False.
    degree : int, default=3
        Polynomial / spline basis degree, used by ``"polynomial"`` and the spline methods
        (``"cubicspline"``, ``"pspline"``, ``"bspline"``, ...). Ignored by methods without a degree.
    target_aware : bool, default=True
        Whether target-aware methods (feature maps and splines) use ``y`` to place their basis
        units, e.g. decision-tree knot/center selection. Requires ``y`` to be passed to ``fit``;
        set to False for a purely unsupervised, ``y``-free fit. Pairs with ``placement_strategy``:
        when True the strategy must be ``"cart"`` or ``"lightgbm"``; when False it must be
        ``"uniform"`` or ``"quantile"``.
    placement_strategy : str, default="cart"
        How basis units / knots are placed, interpreted according to ``target_aware``. When
        ``target_aware`` is True: ``"cart"`` (a single decision tree, always available) or
        ``"lightgbm"`` (a gradient-boosted ensemble, requires the optional ``lightgbm``
        dependency). When ``target_aware`` is False: ``"uniform"`` (evenly spaced across the
        feature range) or ``"quantile"`` (spaced by the data quantiles). Applies to the feature
        maps, PLE, and the knot-based splines (``"bspline"`` / ``"mspline"`` / ``"ispline"`` /
        ``"cubicspline"`` / ``"naturalspline"``); the always-target-aware ``"ple"`` only honors the
        supervised strategies, while the penalized ``"pspline"`` / ``"tensorspline"`` (which assume
        equally-spaced knots) and the kernel-based ``"tprs"`` only honor the unsupervised ones.
    task : str, default="regression"
        Supervised task (``"regression"`` or ``"classification"``) used by target-aware methods to
        place basis units / knots against ``y``. Only consulted when ``target_aware`` is True.
    adaptive : bool, default=False
        Whether adaptive-capable methods size each feature's output dimension from the data
        (within ``[min_output_dim, max_output_dim]``) instead of using the fixed ``output_dim``.
        Fixed-width methods (e.g. plain scalers) ignore this flag.
    min_output_dim : int, default=5
        Lower bound on the per-feature output dimension when ``adaptive`` is True. Ignored by
        fixed-width methods and when ``adaptive`` is False.
    max_output_dim : int, default=10
        Upper bound on the per-feature output dimension when ``adaptive`` is True. Ignored by
        fixed-width methods and when ``adaptive`` is False.
    random_state : int or None, default=None
        Global seed forwarded to every stochastic numerical method (PLE and feature-map
        decision trees, the ``quantile`` transformer, and target-aware spline knot
        selectors) to make ``fit`` reproducible. When ``None`` (the default) each transformer
        keeps its own default seed, so the value is only propagated when explicitly set --
        preserving prior behavior while giving a single knob to pin reproducibility. Forwarded
        by ``get_params`` / ``clone`` so an embedding host (e.g. DeepTab) can pass it through.
    scaling : str, default="minmax"
        Optional scaler inserted *before* the numerical method: ``"minmax"`` (rescale to ``[-1, 1]``)
        or ``"standardization"`` (zero mean, unit variance). Skipped automatically when the chosen
        method already scales (e.g. ``numerical_method="minmax"``).
    cat_cutoff : float or int, default=0.03
        Threshold deciding whether an integer column is treated as categorical. A float is a
        unique-ratio cutoff (``n_unique / n_rows < cat_cutoff`` -> categorical); an int is an
        absolute unique-count cutoff (``n_unique < cat_cutoff`` -> categorical).
    treat_all_integers_as_numerical : bool, default=False
        If True, every integer-typed column is treated as numerical regardless of cardinality,
        bypassing the ``cat_cutoff`` heuristic.
    numerical_imputation : str or None, default="median"
        Strategy for the ``SimpleImputer`` that runs *before* every numerical method. Accepts
        any ``sklearn`` strategy (``"median"``, ``"mean"``, ``"most_frequent"``, ``"constant"``).
        ``None`` disables imputation, so NaNs reach the numerical transformers unchanged and the
        finite-input methods (all numerical methods, including PLE) raise on missing values.
    categorical_imputation : str or None, default="most_frequent"
        Strategy for the ``SimpleImputer`` that runs *before* every categorical method. ``None``
        disables imputation for categorical columns.
    add_missing_indicator : bool, default=False
        If True, append a binary missing-value indicator column for each imputed feature (via the
        imputer's ``add_indicator``; a standalone ``MissingIndicator`` is used when imputation is
        disabled). Applies to both numerical and categorical pipelines.
    missing_policy : {"error", "propagate", "impute", "impute_with_indicator", "separate_state"} or None, default=None
        High-level missing-value strategy. ``None`` (default) keeps the explicit
        ``numerical_imputation`` / ``categorical_imputation`` / ``add_missing_indicator``
        parameters authoritative. When set it overrides them:

        - ``"error"`` -- raise :class:`~pretab.exceptions.PretabDataError` if any missing
          value is present at ``fit`` or ``transform``.
        - ``"propagate"`` -- disable imputation so NaNs reach the transformers unchanged
          (each family applies its own missing-value contract).
        - ``"impute"`` -- impute with the configured strategy (no indicator).
        - ``"impute_with_indicator"`` -- impute and append a missing indicator column.
        - ``"separate_state"`` -- impute for the representation *and* emit a dedicated
          ``__missing`` column per feature that stays outside the ordinary basis, so a
          downstream model can learn a separate response to missingness.
    policy : RepresentationPolicy or dict or None, default=None
        Central edge-case policy (see :class:`~pretab.RepresentationPolicy`) governing how
        constant columns, out-of-range values, missing values, and non-finite inputs are
        handled. ``None`` uses the default policy, which reproduces the library's historical
        behaviour. Pass a mapping such as ``{"constant": "error"}`` to tighten a single axis.
    max_output_features : int or None, default=None
        Upper bound on the total number of output columns produced across all input
        features. ``None`` disables the check. A violation is handled per
        ``overflow_policy``.
    max_features_per_input : int or None, default=None
        Upper bound on the number of output columns any single input feature may
        expand to. ``None`` disables the check. A violation is handled per
        ``overflow_policy``.
    max_dense_memory : int or None, default=None
        Upper bound, in bytes, on the estimated dense output footprint
        (``n_rows * total_output_dim_ * itemsize``) evaluated against the training
        data at ``fit``. ``None`` disables the check. A violation is handled per
        ``overflow_policy``. See :meth:`estimate_memory` to estimate this for any
        input.
    overflow_policy : {"error", "warn", "ignore"}, default="error"
        What to do when a configured output budget is exceeded: ``"error"`` raises
        :class:`~pretab.exceptions.OutputBudgetError`, ``"warn"`` emits a
        :class:`~pretab.exceptions.ConfigWarning`, ``"ignore"`` proceeds silently.
        Only takes effect when at least one budget parameter above is set.
    output_format : {"dense", "sparse", "auto"}, default="dense"
        Container used for the transformed output. ``"dense"`` (the default, for
        backward compatibility) returns NumPy arrays; ``"sparse"`` returns SciPy
        CSR matrices (a single stacked CSR when ``return_array=True``, otherwise CSR
        blocks in the output dict); ``"auto"`` selects ``"sparse"`` when the output
        density falls below ``0.3`` and ``"dense"`` otherwise. Ignored when
        :meth:`set_output` requests a pandas or polars DataFrame. Every ``transform``
        records the resolved choice and its memory footprint in ``output_report_``.
    dtype : numpy dtype or None, default=None
        Optional dtype to cast the transformed output to (e.g. ``numpy.float32`` to
        halve memory). ``None`` keeps the native ``float64`` output.
    verbose : int, default=0
        Verbosity level controlling ``fit``-time logging, applied through the shared
        ``"pretab"`` logger so a single setting on this entry point governs the whole
        package (including the individual numerical/categorical transformers). Accepts an
        int ``0``-``3`` and also ``bool`` (``True`` -> ``1``, ``False`` -> ``0``):

        - ``0`` -- silent on the happy path (only :class:`~pretab.PretabWarning` data warnings).
        - ``1`` -- one fit-summary line (feature counts, resolved methods, total output width, duration).
        - ``2`` -- the per-feature table that :meth:`get_feature_info` builds.
        - ``3`` -- internal decisions (fitted bins / knots / centers).

        Stored verbatim and forwarded by ``get_params`` / ``clone``, so an embedding host
        (e.g. DeepTab) can pass it straight through ``Preprocessor(**kwargs)``. PreTab never
        configures the root logger or attaches a handler when the host already owns one, so
        ``verbose=0`` keeps PreTab silent under a host's own logging.
    preset : {"standard", "expanded", "adaptive"} or None, default=None
        Optional named configuration bundle applied as a transparent alias. A preset only
        fills in parameters left at their defaults; any parameter set explicitly always wins.
        ``"standard"`` is the PLE + integer-code baseline, ``"expanded"`` widens the
        numerical basis and one-hot encodes categoricals, and ``"adaptive"`` sizes each
        feature's width from the data. Call :meth:`get_resolved_config` to see the effective
        parameters. ``None`` (default) uses the individual parameters unchanged.

    Attributes
    ----------
    column_transformer\_ : ColumnTransformer
        The internal scikit-learn column transformer that handles feature-wise preprocessing.
        Set when the preprocessor is fitted.
    n_features_in\_ : int
        Number of input features seen during ``fit``.
    total_output_dim\_ : int
        Total number of output columns produced across all input features
        (equals the width of ``transform(..., return_array=True)``).
    output_dims\_ : dict
        Per-feature expanded output-column counts, keyed by input feature name.
        The values sum to ``total_output_dim_``.
    output_report\_ : dict
        Memory report for the most recent ``transform``, with keys ``format``
        (``"dense"`` or ``"sparse"``), ``shape``, ``density``, ``dense_bytes``,
        ``actual_bytes``, and ``memory_saved_bytes``. Set on every ``transform``.
    embeddings\_ : bool
        Whether embedding vectors were provided at ``fit`` time and are expected in transformation.
    embedding_dimensions\_ : dict
        Dictionary of embedding feature names to their expected dimensionality.

    Notes
    -----
    Available ``numerical_method`` values: ``"none"``, ``"minmax"``, ``"standardization"``,
    ``"robust"``, ``"quantile"``, ``"polynomial"``, ``"box-cox"``, ``"yeo-johnson"``, ``"ple"``,
    ``"custombin"``, ``"rbf"``, ``"relu"``, ``"sigmoid"``, ``"tanh"``, ``"cubicspline"``,
    ``"naturalspline"``, ``"pspline"``, ``"tensorspline"``, ``"tprs"``, ``"bspline"``,
    ``"mspline"``, ``"ispline"``.

    Available ``categorical_method`` values: ``"int"``, ``"one-hot"``, ``"onehot_from_ordinal"``,
    ``"pretrained"``, ``"custombin"``, ``"none"``. The ``"pretrained"`` method requires the optional
    ``sentence-transformers`` dependency (``pip install "pretab[embeddings]"``).

    Method names are resolved case-insensitively and ignore ``-`` / ``_`` / space separators, so
    ``"one-hot"``, ``"one_hot"`` and ``"OneHot"`` are equivalent. Common synonyms and abbreviations
    are also accepted, e.g. ``"std"`` / ``"standard"`` -> ``"standardization"``, ``"ohe"`` /
    ``"dummy"`` -> ``"one-hot"``, ``"ordinal"`` / ``"label"`` -> ``"int"``, ``"poly"`` ->
    ``"polynomial"``, ``"thin-plate"`` -> ``"tprs"``, and ``"passthrough"`` -> ``"none"``.

    ``transform`` returns a dict of per-feature blocks keyed ``num_<col>`` / ``cat_<col>`` by
    default, or a single stacked array when ``return_array=True``.

    Examples
    --------
    Basic usage -- PLE for numerics and integer codes for categoricals (the defaults):

    >>> import pandas as pd
    >>> from pretab import Preprocessor
    >>> df = pd.DataFrame({"age": [25, 32, 47, 51], "gender": ["M", "F", "F", "M"]})
    >>> y = [0.1, 0.4, 0.9, 1.2]
    >>> pre = Preprocessor()
    >>> out = pre.fit_transform(df, y)
    >>> sorted(out.keys())
    ['cat_gender', 'num_age']

    Cubic-spline basis for numerics with one-hot encoded categoricals:

    >>> pre = Preprocessor(numerical_method="cubicspline", categorical_method="one-hot",
    ...                    output_dim=10, degree=3)
    >>> out = pre.fit_transform(df, y)

    Radial-basis feature maps with target-aware center placement:

    >>> pre = Preprocessor(numerical_method="rbf", target_aware=True, task="regression",
    ...                    output_dim=8)
    >>> out = pre.fit_transform(df, y)

    A different method per column via ``feature_preprocessing``:

    >>> pre = Preprocessor(feature_preprocessing={"age": "pspline", "gender": "one-hot"})
    >>> out = pre.fit_transform(df, y)

    Data-driven (adaptive) width, returned as a single stacked array:

    >>> pre = Preprocessor(numerical_method="ple", adaptive=True,
    ...                    min_output_dim=4, max_output_dim=12)
    >>> arr = pre.fit_transform(df, y, return_array=True)
    >>> arr.ndim
    2
    """

    def __init__(
        self,
        numerical_method="ple",
        categorical_method="int",
        feature_preprocessing=None,
        output_dim=7,
        degree=3,
        target_aware=True,
        placement_strategy="cart",
        task="regression",
        adaptive=False,
        min_output_dim=5,
        max_output_dim=10,
        random_state=None,
        scaling="minmax",
        cat_cutoff=0.03,
        treat_all_integers_as_numerical=False,
        numerical_imputation: str | None = "median",
        categorical_imputation: str | None = "most_frequent",
        add_missing_indicator=False,
        missing_policy=None,
        policy=None,
        max_output_features=None,
        max_features_per_input=None,
        max_dense_memory=None,
        overflow_policy="error",
        output_format="dense",
        dtype=None,
        verbose=0,
        preset=None,
    ):
        """
        Initialize the Preprocessor with various transformation options for tabular data.

        See the :class:`Preprocessor` class docstring for the full parameter reference,
        available ``numerical_method`` / ``categorical_method`` values, and usage examples.
        """

        self.numerical_method = numerical_method
        self.categorical_method = categorical_method
        self.feature_preprocessing = feature_preprocessing
        self.output_dim = output_dim
        self.degree = degree
        self.target_aware = target_aware
        self.placement_strategy = placement_strategy
        self.task = task
        self.adaptive = adaptive
        self.min_output_dim = min_output_dim
        self.max_output_dim = max_output_dim
        self.random_state = random_state
        self.scaling = scaling
        self.cat_cutoff = cat_cutoff
        self.treat_all_integers_as_numerical = treat_all_integers_as_numerical
        self.numerical_imputation = numerical_imputation
        self.categorical_imputation = categorical_imputation
        self.add_missing_indicator = add_missing_indicator
        self.missing_policy = missing_policy
        self.policy = policy
        self.max_output_features = max_output_features
        self.max_features_per_input = max_features_per_input
        self.max_dense_memory = max_dense_memory
        self.overflow_policy = overflow_policy
        self.output_format = output_format
        self.dtype = dtype
        self.verbose = verbose
        self.preset = preset

    def fit(self, X, y=None, embeddings=None):
        """
        Fit the preprocessor to the input data and target labels.

        Parameters
        ----------
        X : pandas.DataFrame, numpy.ndarray, or dict
            The input features.
        y : array-like, default=None
            Target values (used for decision tree-based methods).
        embeddings : np.ndarray or list of np.ndarray, optional
            External embedding arrays to be passed and validated.

        Returns
        -------
        self : Preprocessor
            Fitted instance of the preprocessor.
        """

        verbose = int(self.verbose or 0)
        if verbose > 0:
            configure_logging(verbose)
        start_time = time.perf_counter()

        resolved = self._resolved_params()
        config = PreprocessorConfig.from_params(
            numerical_method=resolved["numerical_method"],
            categorical_method=resolved["categorical_method"],
            feature_preprocessing=resolved["feature_preprocessing"],
            output_dim=resolved["output_dim"],
            degree=resolved["degree"],
            target_aware=resolved["target_aware"],
            placement_strategy=resolved["placement_strategy"],
            task=resolved["task"],
            adaptive=resolved["adaptive"],
            min_output_dim=resolved["min_output_dim"],
            max_output_dim=resolved["max_output_dim"],
            random_state=resolved["random_state"],
            scaling=resolved["scaling"],
            cat_cutoff=resolved["cat_cutoff"],
            treat_all_integers_as_numerical=resolved["treat_all_integers_as_numerical"],
            numerical_imputation=resolved["numerical_imputation"],
            categorical_imputation=resolved["categorical_imputation"],
            add_missing_indicator=resolved["add_missing_indicator"],
            missing_policy=resolved["missing_policy"],
            verbose=resolved["verbose"],
        )

        X = to_dataframe(X)

        if self.missing_policy == "error":
            self._reject_missing(X)

        self.embeddings_ = False
        self.embedding_dimensions_ = {}
        if embeddings is not None:
            self.embeddings_ = True
            if isinstance(embeddings, np.ndarray):
                self.embedding_dimensions_["embedding_1"] = embeddings.shape[1]
            elif isinstance(embeddings, list):
                for i, e in enumerate(embeddings):
                    self.embedding_dimensions_[f"embedding_{i + 1}"] = e.shape[1]

        numerical_features, categorical_features = detect_column_types(
            X,
            cat_cutoff=resolved["cat_cutoff"],
            treat_all_integers_as_numerical=resolved["treat_all_integers_as_numerical"],
            estimator_name=type(self).__name__,
        )

        self.policy_ = RepresentationPolicy.resolve(self.policy)
        self.numerical_features_ = list(numerical_features)
        self.categorical_features_ = list(categorical_features)
        if numerical_features and self.policy_.constant != "allow":
            numeric_values = X[numerical_features].to_numpy(dtype=np.float64, na_value=np.nan)
            apply_constant_policy(numeric_values, self.policy_, estimator=self)

        self.column_transformer_ = build_column_transformer(config, numerical_features, categorical_features)
        self.column_transformer_.fit(X, y)
        self.n_features_in_ = X.shape[1]

        valid_formats = ("auto", "dense", "sparse")
        if self.output_format not in valid_formats:
            raise invalid_param_error(
                type(self).__name__,
                "output_format",
                self.output_format,
                "must be one of 'auto', 'dense', 'sparse'",
                valid=set(valid_formats),
            )

        self._enforce_output_budget(X.shape[0])

        if verbose >= 1:
            logger.info(
                "fit complete: %d numerical (%s) + %d categorical (%s) feature(s) -> %d output columns in %.3fs",
                len(numerical_features),
                config.numerical_method,
                len(categorical_features),
                config.categorical_method,
                len(self.get_feature_names_out()),
                time.perf_counter() - start_time,
            )
        if verbose >= 2:
            info = self.get_feature_info(verbose=False)
            for line in build_transformer_summary(*info):
                logger.debug(line)
        if verbose >= 3:
            self._log_internal_decisions()

        return self

    def transform(self, X, embeddings=None, return_array=False):
        """
        Transform the input data using the fitted column transformer.

        Parameters
        ----------
        X : pandas.DataFrame, numpy.ndarray, or dict
            Input features to transform.
        embeddings : np.ndarray or list of np.ndarray, optional
            Optional external embeddings to attach to the transformation.
        return_array : bool, default=False
            If True, return a single stacked NumPy array. If False, return a dict of transformed arrays.

        Returns
        -------
        dict, np.ndarray, scipy.sparse matrix, or DataFrame
            Transformed data. By default a dictionary of per-feature blocks; a
            single stacked array when ``return_array=True``; a SciPy CSR matrix (or
            CSR blocks) when ``output_format`` resolves to ``"sparse"``; or a pandas
            / polars DataFrame when configured via :meth:`set_output`.
        """

        check_is_fitted(self)

        X = to_dataframe(X, copy=True)

        if self.missing_policy == "error":
            self._reject_missing(X)

        transformed_X = self.column_transformer_.transform(X)
        if sp.issparse(transformed_X):
            transformed_X = transformed_X.toarray()  # type: ignore
        transformed_X = np.asarray(transformed_X)
        if self.dtype is not None:
            transformed_X = transformed_X.astype(self.dtype, copy=False)

        fmt, self.output_report_ = compute_output_report(transformed_X, self.output_format)

        container = _get_output_config("transform", self)["dense"]
        if container in ("pandas", "polars"):
            return to_dataframe_output(transformed_X, self.get_feature_names_out(), container)

        slices = None if return_array else get_output_slices(self.column_transformer_)
        return format_output(
            transformed_X,
            return_array=return_array,
            slices=slices,
            embeddings=embeddings,
            embeddings_expected=self.embeddings_,
            embedding_dimensions=self.embedding_dimensions_,
            output_format=fmt,
        )

    def fit_transform(self, X, y=None, embeddings=None, return_array=False):
        """
        Convenience method that fits the preprocessor and transforms the data.

        Parameters
        ----------
        X : pandas.DataFrame, numpy.ndarray, or dict
            Input features.
        y : array-like, optional
            Target values.
        embeddings : np.ndarray or list of np.ndarray, optional
            Optional embedding arrays.
        return_array : bool, default=False
            Whether to return a stacked NumPy array or a dictionary of arrays.

        Returns
        -------
        dict or np.ndarray
            Transformed dataset in the specified output format.
        """

        return self.fit(X, y, embeddings=embeddings).transform(X, embeddings, return_array)

    @classmethod
    def _param_defaults(cls):
        """Return the ``__init__`` parameter defaults, keyed by name."""
        signature = inspect.signature(cls.__init__)
        return {
            name: parameter.default
            for name, parameter in signature.parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }

    def _resolved_params(self):
        """Return the effective parameters after expanding ``preset``.

        A preset fills in only the parameters left at their ``__init__`` default;
        explicitly-set parameters always take precedence. The ``preset`` key is
        dropped from the returned mapping.
        """
        params = self.get_params(deep=False)
        preset = params.pop("preset", None)
        if preset is None:
            return params
        if preset not in PRESETS:
            raise invalid_param_error(
                type(self).__name__,
                "preset",
                preset,
                "must be one of " + ", ".join(repr(name) for name in sorted(PRESETS)),
                valid=set(PRESETS),
            )
        defaults = self._param_defaults()
        resolved = dict(params)
        for key, preset_value in PRESETS[preset].items():
            if key in defaults and params.get(key) == defaults[key]:
                resolved[key] = preset_value
        return resolved

    def get_resolved_config(self):
        """Return the effective parameter mapping after ``preset`` expansion.

        When ``preset`` is set, its bundled values fill in every parameter the
        caller left at its default while explicitly-set parameters win; the
        ``preset`` key itself is removed. When ``preset`` is ``None`` this is simply
        :meth:`get_params` without the ``preset`` entry. The returned dict is the
        configuration ``fit`` builds from, so it makes a preset's effect inspectable
        before fitting.

        Returns
        -------
        dict
            The resolved parameter mapping.
        """
        return self._resolved_params()

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Delegates to the fitted internal :class:`~sklearn.compose.ColumnTransformer`,
        returning one name per output column of the stacked array produced by
        ``transform(..., return_array=True)``.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names. Passed through to the underlying column transformer.

        Returns
        -------
        feature_names_out : numpy.ndarray of str
            Transformed feature names.
        """

        check_is_fitted(self)
        raw_names = self.column_transformer_.get_feature_names_out(input_features)
        return np.array(clean_feature_names(self.column_transformer_, raw_names))

    def get_feature_lineage(self):
        """Return per-output-column provenance for the fitted preprocessor.

        Each :class:`~pretab.core.representation.FeatureLineage` record maps one
        output column back to its source feature(s), representation family, and
        component, covering 100% of the columns produced by
        :meth:`get_feature_names_out` (and in the same order).

        Returns
        -------
        lineage : list of FeatureLineage
            One record per output column of the transformed array.
        """
        check_is_fitted(self)
        return build_feature_lineage(self.column_transformer_)

    @property
    def total_output_dim_(self) -> int:
        """Total number of output columns produced across all input features.

        Fitted attribute (available only after ``fit``). Defined as
        ``len(self.get_feature_names_out())`` so it always equals the true width
        of the array returned by ``transform(..., return_array=True)``.
        """
        check_is_fitted(self)
        return len(self.get_feature_names_out())

    @property
    def output_dims_(self) -> dict:
        """Per-feature expanded output-column counts.

        Fitted attribute (available only after ``fit``). Maps each input feature
        (by name) to the number of columns it contributes to the transformed
        array, complementing the scalar :attr:`total_output_dim_`. The values sum
        to :attr:`total_output_dim_`. Useful when features get different
        ``output_dim`` values (via ``feature_preprocessing``) or expand to a
        non-uniform width (e.g. one-hot encoded categoricals).
        """
        check_is_fitted(self)
        column_transformer = self.column_transformer_
        output_indices = column_transformer.output_indices_
        dims: dict = {}
        for name, _transformer, columns in column_transformer.transformers_:
            span = output_indices[name]
            width = span.stop - span.start
            if width == 0:
                continue
            if name == "remainder":
                # Passthrough columns: one output column per untransformed feature.
                for full in column_transformer.get_feature_names_out()[span]:
                    feature = full.split("__", 1)[-1] if "__" in full else full
                    dims[feature] = dims.get(feature, 0) + 1
            else:
                dims[columns[0]] = width
        return dims

    def _output_itemsize(self) -> int:
        """Bytes per element of the dense transformed array (float64 for now)."""
        return np.dtype(np.float64).itemsize

    def estimate_output_shape(self, X) -> tuple:
        """Estimate the shape of the dense transformed array for ``X``.

        Fitted method. Returns ``(n_rows, total_output_dim_)`` where ``n_rows`` is
        the number of rows in ``X`` and the column count is the fitted output width
        (the same width :meth:`transform` would produce with ``return_array=True``).

        Parameters
        ----------
        X : pandas.DataFrame, numpy.ndarray, or dict
            Input whose row count drives the estimate; not transformed.

        Returns
        -------
        tuple of int
            ``(n_rows, n_output_columns)``.
        """
        check_is_fitted(self)
        n_rows = to_dataframe(X).shape[0]
        return (int(n_rows), int(self.total_output_dim_))

    def estimate_memory(self, X) -> int:
        """Estimate the dense-array memory footprint (in bytes) of transforming ``X``.

        Fitted method. Computed as ``n_rows * total_output_dim_ * itemsize`` for the
        dense output dtype, without materialising the transform.

        Parameters
        ----------
        X : pandas.DataFrame, numpy.ndarray, or dict
            Input whose row count drives the estimate; not transformed.

        Returns
        -------
        int
            Estimated number of bytes for the dense transformed array.
        """
        n_rows, n_cols = self.estimate_output_shape(X)
        return int(n_rows * n_cols * self._output_itemsize())

    def _reject_missing(self, X) -> None:
        """Raise when ``missing_policy="error"`` but ``X`` contains missing values."""
        na_columns = [col for col in X.columns if X[col].isna().any()]
        if na_columns:
            raise PretabDataError(
                f"missing_policy='error' but missing values were found in columns {na_columns}.\n"
                "Fix: impute the data first, or choose a different missing_policy "
                "('propagate', 'impute', 'impute_with_indicator', 'separate_state')."
            )

    def _enforce_output_budget(self, n_rows: int) -> None:
        """Check the fitted output width against the configured output budget.

        Runs at the end of :meth:`fit`. When no budget parameter is set this is a
        no-op (the historical behaviour). Any violation is handled according to
        ``overflow_policy``: ``"error"`` raises
        :class:`~pretab.exceptions.OutputBudgetError`, ``"warn"`` emits a
        :class:`~pretab.exceptions.ConfigWarning`, and ``"ignore"`` proceeds
        silently.
        """
        valid_policies = ("error", "warn", "ignore")
        if self.overflow_policy not in valid_policies:
            raise invalid_param_error(
                type(self).__name__,
                "overflow_policy",
                self.overflow_policy,
                "must be one of 'error', 'warn', 'ignore'",
                valid=set(valid_policies),
            )

        violations: list[str] = []

        total = int(self.total_output_dim_)
        if self.max_output_features is not None and total > self.max_output_features:
            violations.append(f"total output columns ({total}) exceed max_output_features ({self.max_output_features})")

        if self.max_features_per_input is not None:
            for feature, width in self.output_dims_.items():
                if width > self.max_features_per_input:
                    violations.append(
                        f"feature {feature!r} expands to {width} columns, "
                        f"exceeding max_features_per_input ({self.max_features_per_input})"
                    )

        if self.max_dense_memory is not None:
            estimated = n_rows * total * self._output_itemsize()
            if estimated > self.max_dense_memory:
                violations.append(
                    f"dense output for {n_rows} row(s) needs ~{estimated} bytes, "
                    f"exceeding max_dense_memory ({self.max_dense_memory})"
                )

        if not violations:
            return

        message = "Output budget exceeded: " + "; ".join(violations) + "."
        if self.overflow_policy == "error":
            raise OutputBudgetError(message)
        if self.overflow_policy == "warn":
            warnings.warn(message, ConfigWarning, stacklevel=2)

    def get_feature_info(self, verbose=True):
        """
        Retrieves metadata about the transformed features.

        Provides detailed information for each input feature, including:
        - preprocessing applied
        - output dimensionality
        - number of categories (for categorical features)
        - embedding dimensions (if any)

        Parameters
        ----------
        verbose : bool, default=True
            If True, renders an aligned per-feature table through the ``pretab``
            logger (attaching a stream handler when none is configured). If False,
            the info dicts are returned silently.

        Returns
        -------
        tuple of dicts
            numerical_feature_info : dict
                Metadata for numerical features.
            categorical_feature_info : dict
                Metadata for categorical features.
            embedding_feature_info : dict
                Metadata for embedding features, if used.
        """

        check_is_fitted(self)

        numerical_feature_info, categorical_feature_info, embedding_feature_info = build_feature_info(
            self.column_transformer_,
            embeddings=self.embeddings_,
            embedding_dimensions=self.embedding_dimensions_,
        )

        if verbose:
            configure_logging(1)
            for line in build_transformer_summary(
                numerical_feature_info,
                categorical_feature_info,
                embedding_feature_info,
            ):
                logger.info(line)

        return numerical_feature_info, categorical_feature_info, embedding_feature_info

    def _log_internal_decisions(self):
        """Log fitted internal decisions (bins / knots / centers) at DEBUG."""
        for name, transformer, _columns in self.column_transformer_.transformers_:
            last_step = transformer.steps[-1][1] if hasattr(transformer, "steps") else transformer
            for attr in (
                "thresholds_",
                "knots_",
                "centers_",
                "n_knots_",
                "total_output_dim_",
            ):
                if hasattr(last_step, attr):
                    logger.debug("%s.%s = %r", name, attr, getattr(last_step, attr))

    # --- Portable serialization (P9.1) ---
    def to_spec(self, path=None) -> dict:
        """Serialize the fitted preprocessor to a portable, versioned spec.

        Produces a self-describing JSON-compatible dictionary (schema version,
        PreTab / numpy / scipy / scikit-learn versions, resolved parameters, a
        per-representation summary, the output-column order, and the encoded
        fitted state) that reconstructs this estimator bit-for-bit via
        :meth:`from_spec`. Unlike :mod:`pickle`, loading a spec never executes
        estimator code and only imports an allow-listed set of library modules.

        Parameters
        ----------
        path : str, os.PathLike, or None, default=None
            When given, the spec is also written to this path as UTF-8 JSON.

        Returns
        -------
        dict
            The spec dictionary (always returned, whether or not ``path`` is set).
        """
        check_is_fitted(self)
        spec = preprocessor_to_spec(self)
        if path is not None:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(spec, handle, indent=2)
        return spec

    @classmethod
    def from_spec(cls, source) -> "Preprocessor":
        """Reconstruct a fitted preprocessor from a spec created by :meth:`to_spec`.

        Parameters
        ----------
        source : str, os.PathLike, or dict
            A path to a JSON spec file, or the spec dictionary itself.

        Returns
        -------
        Preprocessor
            A fitted preprocessor equivalent to the one that produced the spec;
            ``transform`` reproduces the original output bit-for-bit.
        """
        if isinstance(source, dict):
            data = source
        elif isinstance(source, (str, os.PathLike)):
            with open(source, encoding="utf-8") as handle:
                data = json.load(handle)
        else:
            raise PretabSerializationError("from_spec expects a spec dict or a path to a JSON spec file.")
        obj = preprocessor_from_spec(data)
        if not isinstance(obj, cls):
            raise PretabSerializationError(f"Spec reconstructed a {type(obj).__name__}, expected {cls.__name__}.")
        return obj

    # --- Fingerprint & reproducibility (P9.2) ---
    def _canonical_spec(self) -> dict:
        """Deterministic subset of the spec used for fingerprinting."""
        spec = preprocessor_to_spec(self)
        return {
            "schema_version": spec["schema_version"],
            "pretab_version": spec["pretab_version"],
            "library_versions": spec["library_versions"],
            "feature_names_out": spec["feature_names_out"],
            "state": spec["state"],
        }

    @property
    def fingerprint_(self) -> str:
        """Stable SHA-256 digest identifying this fitted preprocessor.

        Fitted attribute. Computed over a canonical JSON view of the resolved
        configuration, dependency versions, output-column order, random seeds, and
        the fitted state (knot / center / bin locations, scaler statistics, encoder
        categories). The digest is deterministic across processes and machines, so
        two preprocessors share a fingerprint iff they transform identically.
        """
        check_is_fitted(self)
        canonical = json.dumps(self._canonical_spec(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def reproducibility_report(self) -> dict:
        """Return a machine-readable reproducibility summary for this fitted preprocessor.

        Returns
        -------
        dict
            Fingerprint, schema / library versions, random seed, output dtype and
            format, input/output widths, and the per-feature representation
            families -- everything needed to audit or reproduce the fit.
        """
        check_is_fitted(self)
        spec = preprocessor_to_spec(self)
        representations = {
            entry["columns"][0]: entry.get("family") for entry in spec["representations"] if entry.get("columns")
        }
        return {
            "fingerprint": self.fingerprint_,
            "schema_version": SCHEMA_VERSION,
            "pretab_version": spec["pretab_version"],
            "library_versions": spec["library_versions"],
            "random_state": self.random_state,
            "output_format": self.output_format,
            "dtype": None if self.dtype is None else str(self.dtype),
            "n_features_in": int(self.n_features_in_),
            "n_output_features": len(spec["feature_names_out"]),
            "representations": representations,
        }

    # --- Immutable lifecycle (P9.3) ---
    @property
    def lifecycle_state_(self) -> str:
        """Current lifecycle state: ``UNFITTED``, ``FITTED``, ``FROZEN``, or ``STALE``."""
        try:
            check_is_fitted(self)
        except Exception:
            return "UNFITTED"
        if getattr(self, "_frozen", False):
            return "FROZEN"
        if getattr(self, "_stale_reason", None) is not None:
            return "STALE"
        return "FITTED"

    def is_frozen(self) -> bool:
        """Return whether this preprocessor has been frozen against mutation."""
        return bool(getattr(self, "_frozen", False))

    def freeze(self) -> "Preprocessor":
        """Freeze the fitted preprocessor, blocking further ``set_params`` mutation.

        Returns ``self`` for chaining. A frozen preprocessor is intended as an
        immutable deployment artifact; use :meth:`clone_unfitted` or :meth:`refit`
        to obtain a fresh, mutable estimator.
        """
        check_is_fitted(self)
        self._frozen = True
        return self

    def mark_stale(self, reason: str) -> "Preprocessor":
        """Mark this fitted preprocessor as stale (its inputs/assumptions changed).

        Records ``reason`` and flips :attr:`lifecycle_state_` to ``STALE`` (unless
        already ``FROZEN``). Purely advisory: it does not alter the fitted state.
        Returns ``self`` for chaining.
        """
        check_is_fitted(self)
        self._stale_reason = reason
        return self

    @property
    def stale_reason_(self):
        """The reason recorded by :meth:`mark_stale`, or ``None``."""
        return getattr(self, "_stale_reason", None)

    def clone_unfitted(self) -> "Preprocessor":
        """Return a fresh, unfitted, mutable copy carrying the same constructor params."""
        return cast("Preprocessor", clone(self))

    def refit(self, X, y=None, embeddings=None) -> "Preprocessor":
        """Fit a fresh copy on new data and return it, leaving ``self`` untouched.

        Enables re-fitting a frozen or deployed preprocessor without mutating the
        original: returns a new, unfrozen, fitted :class:`Preprocessor`.
        """
        return self.clone_unfitted().fit(X, y, embeddings=embeddings)

    def set_params(self, **params):
        """Set parameters, refusing to mutate a frozen preprocessor."""
        if params and self.is_frozen():
            raise FrozenRepresentationError(
                f"Cannot set_params({', '.join(sorted(params))}) on a frozen {type(self).__name__}. "
                "Use clone_unfitted() for a mutable copy, or refit() to fit fresh data."
            )
        return super().set_params(**params)
