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

    def _detect_column_types(self, X):
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
        return self.fit(X, y, embeddings=embeddings).transform(
            X, embeddings, return_array
        )

    def get_feature_info(self, verbose=True):
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
