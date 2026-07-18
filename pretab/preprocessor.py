import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .core.logging import configure_logging, get_logger
from .pipeline import (
    get_categorical_transformer_steps,
    get_numerical_transformer_steps,
)

logger = get_logger(__name__)



class Preprocessor(TransformerMixin, BaseEstimator):
    """
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
        ``"onehot_from_ordinal"`` (integer codes then one-hot), ``"pretrained"`` (sentence-transformer
        language embeddings), and ``"custombin"`` (discretized bin codes). Pass ``None`` (resolved to
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
    task : str, default="regression"
        Supervised task (``"regression"`` or ``"classification"``) used by target-aware methods to
        place basis units / knots against ``y``. Only consulted when ``use_target`` is True.
    use_target : bool, default=True
        Whether target-aware methods (feature maps and splines) use ``y`` to place their basis
        units, e.g. decision-tree knot/center selection. Requires ``y`` to be passed to ``fit``;
        set to False for a purely unsupervised, ``y``-free fit driven by ``strategy``.
    strategy : str, default="uniform"
        Placement strategy for basis units when ``use_target`` is False: ``"uniform"`` (evenly
        spaced across the feature range) or ``"quantile"`` (spaced by the data quantiles).
    degree : int, default=3
        Polynomial / spline basis degree, used by ``"polynomial"`` and the spline methods
        (``"cubicspline"``, ``"pspline"``, ``"bspline"``, ...). Ignored by methods without a degree.
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

    Attributes
    ----------
    column_transformer_ : ColumnTransformer
        The internal scikit-learn column transformer that handles feature-wise preprocessing.
        Set when the preprocessor is fitted.
    n_features_in_ : int
        Number of input features seen during ``fit``.
    embeddings_ : bool
        Whether embedding vectors were provided at ``fit`` time and are expected in transformation.
    embedding_dimensions_ : dict
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

    >>> pre = Preprocessor(numerical_method="rbf", use_target=True, task="regression",
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
        adaptive=False,
        min_output_dim=5,
        max_output_dim=10,
        task="regression",
        use_target=True,
        strategy="uniform",
        degree=3,
        scaling="minmax",
        cat_cutoff=0.03,
        treat_all_integers_as_numerical=False,
        verbose=0,
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
        self.adaptive = adaptive
        self.min_output_dim = min_output_dim
        self.max_output_dim = max_output_dim
        self.task = task
        self.use_target = use_target
        self.strategy = strategy
        self.degree = degree
        self.scaling = scaling
        self.cat_cutoff = cat_cutoff
        self.treat_all_integers_as_numerical = treat_all_integers_as_numerical
        self.verbose = verbose

    def _detect_column_types(self, X):
        """
        Detects categorical and numerical features in the input data.

        Parameters
        ----------
        X : pandas.DataFrame, numpy.ndarray, or dict
            The input data to analyze.

        Returns
        -------
        numerical_features : list of str
            Column names detected as numerical features.
        categorical_features : list of str
            Column names detected as categorical features.
        """

        categorical_features = []
        numerical_features = []

        if isinstance(X, dict):
            X = pd.DataFrame(X)
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        for col in X.columns:
            num_unique_values = X[col].nunique()
            total_samples = len(X[col])

            if self.treat_all_integers_as_numerical and X[col].dtype.kind == "i":
                numerical_features.append(col)
            else:
                if isinstance(self.cat_cutoff, float):
                    cutoff_condition = (
                        num_unique_values / total_samples
                    ) < self.cat_cutoff
                elif isinstance(self.cat_cutoff, int):
                    cutoff_condition = num_unique_values < self.cat_cutoff
                else:
                    raise ValueError(
                        "cat_cutoff should be either a float or an integer."
                    )

                if X[col].dtype.kind not in "iufc" or (
                    X[col].dtype.kind == "i" and cutoff_condition
                ):
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)

        return numerical_features, categorical_features

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

        if isinstance(X, dict):
            X = pd.DataFrame(X)
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        numerical_method = (
            self.numerical_method.lower()
            if self.numerical_method is not None
            else "none"
        )
        categorical_method = (
            self.categorical_method.lower()
            if self.categorical_method is not None
            else "none"
        )
        feature_preprocessing = self.feature_preprocessing or {}

        self.embeddings_ = False
        self.embedding_dimensions_ = {}
        if embeddings is not None:
            self.embeddings_ = True
            if isinstance(embeddings, np.ndarray):
                self.embedding_dimensions_["embedding_1"] = embeddings.shape[1]
            elif isinstance(embeddings, list):
                for i, e in enumerate(embeddings):
                    self.embedding_dimensions_[f"embedding_{i + 1}"] = e.shape[1]

        numerical_features, categorical_features = self._detect_column_types(X)
        transformers = []

        for feature in numerical_features:
            method = feature_preprocessing.get(feature, numerical_method)
            steps = get_numerical_transformer_steps(
                method=method,
                task=self.task,
                use_decision_tree=self.use_target,
                add_imputer=True,
                imputer_strategy="mean",
                output_dim=self.output_dim,
                adaptive=self.adaptive,
                min_output_dim=self.min_output_dim if self.adaptive else None,
                max_output_dim=self.max_output_dim if self.adaptive else None,
                degree=self.degree,
                scaling=self.scaling,
                strategy=self.strategy,
            )
            transformers.append((f"num_{feature}", Pipeline(steps), [feature]))

        for feature in categorical_features:
            method = feature_preprocessing.get(feature, categorical_method)
            steps = get_categorical_transformer_steps(method)
            transformers.append((f"cat_{feature}", Pipeline(steps), [feature]))

        self.column_transformer_ = ColumnTransformer(
            transformers=transformers, remainder="passthrough"
        )
        self.column_transformer_.fit(X, y)
        self.n_features_in_ = X.shape[1]

        if verbose >= 1:
            logger.info(
                "fit complete: %d numerical (%s) + %d categorical (%s) feature(s) "
                "-> %d output columns in %.3fs",
                len(numerical_features),
                numerical_method,
                len(categorical_features),
                categorical_method,
                len(self.get_feature_names_out()),
                time.perf_counter() - start_time,
            )
        if verbose >= 2:
            info = self.get_feature_info(verbose=False)
            for line in self._feature_table_lines(*info):
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
        dict or np.ndarray
            Transformed data. A dictionary if return_array=False, else a NumPy array.
        """

        check_is_fitted(self)

        if isinstance(X, dict):
            X = pd.DataFrame(X)
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            X = X.copy()

        transformed_X = self.column_transformer_.transform(X)

        if return_array:
            return transformed_X

        transformed_dict = {}
        start = 0
        for name, transformer, columns in self.column_transformer_.transformers_:
            if transformer == "drop":
                continue
            if hasattr(transformer, "transform"):
                width = transformer.transform(X[columns]).shape[1]
            else:
                width = 1
            transformed_dict[name] = transformed_X[:, start : start + width]
            start += width

        if embeddings is not None:
            if not self.embeddings_:
                raise ValueError("Embeddings were not expected, but were provided.")
            if isinstance(embeddings, np.ndarray):
                transformed_dict["embedding_1"] = embeddings.astype(np.float32)
            elif isinstance(embeddings, list):
                for idx, e in enumerate(embeddings):
                    transformed_dict[f"embedding_{idx + 1}"] = e.astype(np.float32)

        return transformed_dict

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

        return self.fit(X, y, embeddings=embeddings).transform(
            X, embeddings, return_array
        )

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
        return self.column_transformer_.get_feature_names_out(input_features)

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

        numerical_feature_info = {}
        categorical_feature_info = {}

        embedding_feature_info = (
            {
                key: {"preprocessing": None, "dimension": dim, "categories": None}
                for key, dim in self.embedding_dimensions_.items()
            }
            if self.embeddings_
            else {}
        )

        for (
            name,
            transformer_pipeline,
            columns,
        ) in self.column_transformer_.transformers_:
            steps = [step[0] for step in transformer_pipeline.steps]

            for feature_name in columns:
                preprocessing_type = " -> ".join(steps)
                dimension = None
                categories = None

                if "discretizer" in steps or any(
                    step in steps
                    for step in [
                        "standardization",
                        "minmax",
                        "quantile",
                        "polynomial",
                        "splines",
                        "box-cox",
                    ]
                ):
                    last_step = transformer_pipeline.steps[-1][1]
                    if hasattr(last_step, "transform"):
                        dummy_input = np.zeros((1, 1)) + 1e-05
                        try:
                            transformed_feature = last_step.transform(dummy_input)
                            dimension = transformed_feature.shape[1]
                        except (ValueError, TypeError, AttributeError, IndexError) as exc:
                            logger.debug(
                                "Could not introspect output width of %r: %s",
                                feature_name,
                                exc,
                            )
                            dimension = None
                    numerical_feature_info[feature_name] = {
                        "preprocessing": preprocessing_type,
                        "dimension": dimension,
                        "categories": None,
                    }

                elif "continuous_ordinal" in steps:
                    step = transformer_pipeline.named_steps["continuous_ordinal"]
                    categories = len(step.mapping_[columns.index(feature_name)])
                    dimension = 1
                    categorical_feature_info[feature_name] = {
                        "preprocessing": preprocessing_type,
                        "dimension": dimension,
                        "categories": categories,
                    }

                elif "onehot" in steps:
                    step = transformer_pipeline.named_steps["onehot"]
                    if hasattr(step, "categories_"):
                        categories = sum(len(cat) for cat in step.categories_)
                        dimension = categories
                    categorical_feature_info[feature_name] = {
                        "preprocessing": preprocessing_type,
                        "dimension": dimension,
                        "categories": categories,
                    }

                else:
                    last_step = transformer_pipeline.steps[-1][1]
                    if hasattr(last_step, "transform"):
                        dummy_input = np.zeros((1, 1))
                        try:
                            transformed_feature = last_step.transform(dummy_input)
                            dimension = transformed_feature.shape[1]
                        except (ValueError, TypeError, AttributeError, IndexError) as exc:
                            logger.debug(
                                "Could not introspect output width of %r: %s",
                                feature_name,
                                exc,
                            )
                            dimension = None
                    if "cat" in name:
                        categorical_feature_info[feature_name] = {
                            "preprocessing": preprocessing_type,
                            "dimension": dimension,
                            "categories": None,
                        }
                    else:
                        numerical_feature_info[feature_name] = {
                            "preprocessing": preprocessing_type,
                            "dimension": dimension,
                            "categories": None,
                        }

        if verbose:
            configure_logging(1)
            for line in self._feature_table_lines(
                numerical_feature_info,
                categorical_feature_info,
                embedding_feature_info,
            ):
                logger.info(line)

        return numerical_feature_info, categorical_feature_info, embedding_feature_info

    def _feature_table_lines(self, numerical_info, categorical_info, embedding_info):
        """Build aligned, human-readable rows describing the fitted feature layout."""
        rows = []
        for feat, info in numerical_info.items():
            rows.append(
                (str(feat), "numerical", str(info["preprocessing"]), info["dimension"], info["categories"])
            )
        for feat, info in categorical_info.items():
            rows.append(
                (str(feat), "categorical", str(info["preprocessing"]), info["dimension"], info["categories"])
            )
        for feat, info in embedding_info.items():
            rows.append((str(feat), "embedding", "-", info["dimension"], info["categories"]))
        if not rows:
            return []

        feat_w = max(len("feature"), *(len(r[0]) for r in rows))
        kind_w = max(len("kind"), *(len(r[1]) for r in rows))
        pipe_w = max(len("pipeline"), *(len(r[2]) for r in rows))
        header = (
            f"{'feature':<{feat_w}}  {'kind':<{kind_w}}  "
            f"{'pipeline':<{pipe_w}}  {'dim':>4}  {'cats':>5}"
        )
        lines = [header, "-" * len(header)]
        for feat, kind, pipe, dim, cats in rows:
            dim_s = "-" if dim is None else str(dim)
            cats_s = "-" if cats is None else str(cats)
            lines.append(
                f"{feat:<{feat_w}}  {kind:<{kind_w}}  "
                f"{pipe:<{pipe_w}}  {dim_s:>4}  {cats_s:>5}"
            )
        return lines

    def _log_internal_decisions(self):
        """Log fitted internal decisions (bins / knots / centers) at DEBUG."""
        for name, transformer, _columns in self.column_transformer_.transformers_:
            last_step = (
                transformer.steps[-1][1]
                if hasattr(transformer, "steps")
                else transformer
            )
            for attr in (
                "thresholds_",
                "knots_",
                "centers_",
                "n_knots_",
                "total_output_dim_",
            ):
                if hasattr(last_step, attr):
                    logger.debug("%s.%s = %r", name, attr, getattr(last_step, attr))

