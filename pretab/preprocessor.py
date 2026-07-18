import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from .pipeline import (
    get_categorical_transformer_steps,
    get_numerical_transformer_steps,
)


class Preprocessor(TransformerMixin):
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

    Attributes
    ----------
    column_transformer : ColumnTransformer
        The internal scikit-learn column transformer that handles feature-wise preprocessing.
    fitted : bool
        Whether the preprocessor has been fitted.
    embeddings : bool
        Whether embedding vectors are expected and used in transformation.
    embedding_dimensions : dict
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
    ):
        """
        Initialize the Preprocessor with various transformation options for tabular data.

        See the :class:`Preprocessor` class docstring for the full parameter reference,
        available ``numerical_method`` / ``categorical_method`` values, and usage examples.
        """

        self.numerical_method = (
            numerical_method.lower() if numerical_method is not None else "none"
        )
        self.categorical_method = (
            categorical_method.lower() if categorical_method is not None else "none"
        )
        self.feature_preprocessing = feature_preprocessing or {}
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
        self.column_transformer = None
        self.fitted = False
        self.embeddings = False
        self.embedding_dimensions = {}

    def get_params(self, deep=True):
        """Get parameters for the preprocessor.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return parameters of subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {
            "numerical_method": self.numerical_method,
            "categorical_method": self.categorical_method,
            "feature_preprocessing": self.feature_preprocessing,
            "output_dim": self.output_dim,
            "adaptive": self.adaptive,
            "min_output_dim": self.min_output_dim,
            "max_output_dim": self.max_output_dim,
            "task": self.task,
            "use_target": self.use_target,
            "strategy": self.strategy,
            "degree": self.degree,
            "scaling": self.scaling,
            "cat_cutoff": self.cat_cutoff,
            "treat_all_integers_as_numerical": self.treat_all_integers_as_numerical,
        }
        return params

    def set_params(self, **params):
        """Set parameters for the preprocessor.

        Parameters
        ----------
        **params : dict
            Parameter names mapped to their new values.

        Returns
        -------
        self : object
            Preprocessor instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

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

        if isinstance(X, dict):
            X = pd.DataFrame(X)
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        if embeddings is not None:
            self.embeddings = True
            if isinstance(embeddings, np.ndarray):
                self.embedding_dimensions["embedding_1"] = embeddings.shape[1]
            elif isinstance(embeddings, list):
                for i, e in enumerate(embeddings):
                    self.embedding_dimensions[f"embedding_{i + 1}"] = e.shape[1]

        numerical_features, categorical_features = self._detect_column_types(X)
        transformers = []

        for feature in numerical_features:
            method = self.feature_preprocessing.get(feature, self.numerical_method)
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
            method = self.feature_preprocessing.get(feature, self.categorical_method)
            steps = get_categorical_transformer_steps(method)
            transformers.append((f"cat_{feature}", Pipeline(steps), [feature]))

        self.column_transformer = ColumnTransformer(
            transformers=transformers, remainder="passthrough"
        )
        self.column_transformer.fit(X, y)
        self.fitted = True
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

        if not self.fitted:
            raise NotFittedError(
                "Preprocessor must be fitted before calling transform."
            )

        if isinstance(X, dict):
            X = pd.DataFrame(X)
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            X = X.copy()

        transformed_X = self.column_transformer.transform(X)

        if return_array:
            return transformed_X

        transformed_dict = {}
        start = 0
        for name, transformer, columns in self.column_transformer.transformers_:
            if transformer == "drop":
                continue
            if hasattr(transformer, "transform"):
                width = transformer.transform(X[columns]).shape[1]
            else:
                width = 1
            transformed_dict[name] = transformed_X[:, start : start + width]
            start += width

        if embeddings is not None:
            if not self.embeddings:
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
            If True, prints detailed information for each feature.

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

        if not self.fitted:
            raise NotFittedError(
                "Preprocessor must be fitted before calling get_feature_info."
            )

        numerical_feature_info = {}
        categorical_feature_info = {}

        embedding_feature_info = (
            {
                key: {"preprocessing": None, "dimension": dim, "categories": None}
                for key, dim in self.embedding_dimensions.items()
            }
            if self.embeddings
            else {}
        )

        for (
            name,
            transformer_pipeline,
            columns,
        ) in self.column_transformer.transformers_:
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
                        except Exception:
                            dimension = None
                    numerical_feature_info[feature_name] = {
                        "preprocessing": preprocessing_type,
                        "dimension": dimension,
                        "categories": None,
                    }
                    if verbose:
                        print(
                            f"Numerical Feature: {feature_name}, Info: {numerical_feature_info[feature_name]}"
                        )

                elif "continuous_ordinal" in steps:
                    step = transformer_pipeline.named_steps["continuous_ordinal"]
                    categories = len(step.mapping_[columns.index(feature_name)])
                    dimension = 1
                    categorical_feature_info[feature_name] = {
                        "preprocessing": preprocessing_type,
                        "dimension": dimension,
                        "categories": categories,
                    }
                    if verbose:
                        print(
                            f"Categorical Feature (Ordinal): {feature_name}, Info: {categorical_feature_info[feature_name]}"
                        )

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
                    if verbose:
                        print(
                            f"Categorical Feature (One-Hot): {feature_name}, Info: {categorical_feature_info[feature_name]}"
                        )

                else:
                    last_step = transformer_pipeline.steps[-1][1]
                    if hasattr(last_step, "transform"):
                        dummy_input = np.zeros((1, 1))
                        try:
                            transformed_feature = last_step.transform(dummy_input)
                            dimension = transformed_feature.shape[1]
                        except Exception:
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
                        print(
                            f"Feature: {feature_name}, Info: {preprocessing_type}, Dimension: {dimension}"
                        )

                if verbose:
                    print("-" * 50)

        if verbose and self.embeddings:
            print("Embeddings:")
            for key, value in embedding_feature_info.items():
                print(f"  Feature: {key}, Dimension: {value['dimension']}")

        return numerical_feature_info, categorical_feature_info, embedding_feature_info
