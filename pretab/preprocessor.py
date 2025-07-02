import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

from .utils import (
    get_numerical_transformer_steps,
    get_categorical_transformer_steps,
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
    feature_preprocessing : dict, optional
        Dictionary mapping feature names to specific preprocessing methods. Overrides global defaults.
    n_bins : int, default=64
        Number of bins used for binning-based preprocessing (e.g., for discretizers or PLE).
    numerical_preprocessing : str, default="ple"
        Preprocessing method for numerical features (e.g., "standardization", "minmax", "ple", "rbf", etc.).
    categorical_preprocessing : str, default="int"
        Preprocessing method for categorical features (e.g., "int", "ordinal", "onehot").
    use_decision_tree_bins : bool, default=False
        Whether to use decision tree binning for numerical discretization.
    binning_strategy : str, default="uniform"
        Strategy for bin placement when not using tree-based methods. Options: "uniform", "quantile".
    task : str, default="regression"
        Problem type used to guide preprocessing (e.g., "regression" or "classification").
    cat_cutoff : float or int, default=0.03
        Threshold to determine whether integer-valued features are treated as categorical.
    treat_all_integers_as_numerical : bool, default=False
        If True, treat all integer-typed columns as numerical regardless of cardinality.
    degree : int, default=3
        Degree of polynomial or spline basis functions where applicable.
    scaling_strategy : str, default="minmax"
        Strategy for feature scaling (e.g., "standardization", "minmax", etc.).
    n_knots : int, default=64
        Number of knots used in spline-based feature expansions.
    use_decision_tree_knots : bool, default=True
        Whether to use decision tree-based knot placement for spline transformations.
    knots_strategy : str, default="uniform"
        Strategy for placing knots for splines ("uniform" or "quantile").
    spline_implementation : str, default="sklearn"
        Which spline backend implementation to use (e.g., "sklearn", "custom").
    min_unique_vals : int, default=5
        Minimum number of unique values required for a feature to be treated as numerical.

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

    Examples
    --------
    >>> from prefab import Preprocessor
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "age": [25, 32, 47],
    ...     "gender": ["M", "F", "F"]
    ... })
    >>> pre = Preprocessor()
    >>> pre.fit(df)
    >>> out = pre.transform(df)
    >>> out.keys()
    dict_keys(['num_age', 'cat_gender'])
    """

    def __init__(
        self,
        feature_preprocessing=None,
        n_bins=64,
        numerical_preprocessing="ple",
        categorical_preprocessing="int",
        use_decision_tree_bins=False,
        binning_strategy="uniform",
        task="regression",
        cat_cutoff=0.03,
        treat_all_integers_as_numerical=False,
        degree=3,
        scaling_strategy="minmax",
        n_knots=64,
        use_decision_tree_knots=True,
        knots_strategy="uniform",
        spline_implementation="sklearn",
        min_unique_vals=5,
    ):
        """
        Initialize the Preprocessor with various transformation options for tabular data.

        Parameters
        ----------
        feature_preprocessing : dict, optional
            Dictionary specifying preprocessing methods per feature. If None, global settings are used.
        n_bins : int, default=64
            Number of bins to use for binning-based transformations.
        numerical_preprocessing : str, default="ple"
            Preprocessing strategy for numerical features.
        categorical_preprocessing : str, default="int"
            Preprocessing strategy for categorical features.
        use_decision_tree_bins : bool, default=False
            Whether to use decision tree-based binning for numerical features.
        binning_strategy : str, default="uniform"
            Strategy for determining bin edges ("uniform", "quantile").
        task : str, default="regression"
            Task type for decision tree splitting ("regression", "classification").
        cat_cutoff : float or int, default=0.03
            Threshold to determine whether integer-valued columns are treated as categorical.
        treat_all_integers_as_numerical : bool, default=False
            If True, treat all integer columns as numerical.
        degree : int, default=3
            Degree of polynomial or spline basis expansion.
        scaling_strategy : str, default="minmax"
            Scaling method for numerical data ("standardization", "minmax", etc.).
        n_knots : int, default=64
            Number of knots for spline transformations.
        use_decision_tree_knots : bool, default=True
            Use decision tree-based knot placement for splines.
        knots_strategy : str, default="uniform"
            Strategy for placing spline knots.
        spline_implementation : str, default="sklearn"
            Backend implementation to use for splines.
        min_unique_vals : int, default=5
            Minimum number of unique values required for numerical processing.
        """

        self.n_bins = n_bins
        self.numerical_preprocessing = (
            numerical_preprocessing.lower()
            if numerical_preprocessing is not None
            else "none"
        )
        self.categorical_preprocessing = (
            categorical_preprocessing.lower()
            if categorical_preprocessing is not None
            else "none"
        )

        self.use_decision_tree_bins = use_decision_tree_bins
        self.feature_preprocessing = feature_preprocessing or {}
        self.column_transformer = None
        self.fitted = False
        self.binning_strategy = binning_strategy
        self.task = task
        self.cat_cutoff = cat_cutoff
        self.treat_all_integers_as_numerical = treat_all_integers_as_numerical
        self.degree = degree
        self.scaling_strategy = scaling_strategy
        self.n_knots = n_knots
        self.use_decision_tree_knots = use_decision_tree_knots
        self.knots_strategy = knots_strategy
        self.spline_implementation = spline_implementation
        self.min_unique_vals = min_unique_vals
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
            "n_bins": self.n_bins,
            "numerical_preprocessing": self.numerical_preprocessing,
            "categorical_preprocessing": self.categorical_preprocessing,
            "use_decision_tree_bins": self.use_decision_tree_bins,
            "binning_strategy": self.binning_strategy,
            "task": self.task,
            "cat_cutoff": self.cat_cutoff,
            "treat_all_integers_as_numerical": self.treat_all_integers_as_numerical,
            "degree": self.degree,
            "scaling_strategy": self.scaling_strategy,
            "n_knots": self.n_knots,
            "use_decision_tree_knots": self.use_decision_tree_knots,
            "knots_strategy": self.knots_strategy,
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
            method = self.feature_preprocessing.get(
                feature, self.numerical_preprocessing
            )
            steps = get_numerical_transformer_steps(
                method=method,
                task=self.task,
                use_decision_tree=self.use_decision_tree_knots,
                add_imputer=True,
                imputer_strategy="mean",
                bins=self.n_bins,
                degree=self.degree,
                n_knots=self.n_knots,
                scaling=self.scaling_strategy,
                strategy=self.knots_strategy,
                implementation=self.spline_implementation,
            )
            transformers.append((f"num_{feature}", Pipeline(steps), [feature]))

        for feature in categorical_features:
            method = self.feature_preprocessing.get(
                feature, self.categorical_preprocessing
            )
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
