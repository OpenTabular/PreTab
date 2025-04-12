import re
import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from .tree_to_code import tree_to_code


class PLETransformer(BaseEstimator, TransformerMixin):
    """
    Piecewise Linear Encoding (PLE) transformer for numerical features using decision tree-derived binning.

    This transformer encodes each feature by discretizing it using a decision tree and applying piecewise
    linear transformations within each bin. It is particularly useful for capturing nonlinear feature-target
    relationships while maintaining interpretability and continuity.

    Parameters
    ----------
    n_bins : int, default=20
        The maximum number of leaf nodes (bins) the decision tree should use per feature.

    task : {"regression", "classification"}, default="regression"
        Specifies the type of task for tree construction. This determines whether a `DecisionTreeRegressor`
        or `DecisionTreeClassifier` is used for identifying bin splits.

    conditions : list of str or None, default=None
        Optionally supply pre-defined conditions for the tree splits. If None, conditions are learned during fitting.

    **kwargs : dict
        Additional keyword arguments passed to the BaseEstimator.

    Attributes
    ----------
    conditions_ : list of list of str
        Each element is a list of Python expressions representing the tree-based conditions for one input feature.

    n_features_in_ : int
        The number of features seen during `fit`.

    pattern : str
        Regular expression pattern used for extracting numerical values from tree split conditions.

    Notes
    -----
    For each feature, the transformer builds a decision tree to identify split thresholds. These splits are
    then used to encode the feature into a piecewise linear vector representation with increasing cumulative
    effects across bins. Each transformed feature is represented with (n_bins - 1) + 1 features, where values
    below the first threshold are sparse and values above the last threshold are linearly scaled.
    """

    def __init__(self, n_bins=20, task="regression", conditions=None, **kwargs):
        super().__init__(**kwargs)
        self.task = task

        self.n_bins = n_bins
        self.conditions = conditions
        self.pattern = r"-?\d+\.?\d*[eE]?[+-]?\d*"

    def fit(self, X, y):
        original_dim = np.shape(X)[1] if np.ndim(X) == 2 else 1
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        if X.shape[1] < original_dim:
            warnings.warn(
                "Some input features were dropped during check_array validation.",
                UserWarning,
            )

        self.conditions_ = []

        for i in range(X.shape[1]):
            x_feat = X[:, [i]]
            if self.task == "regression":
                dt = DecisionTreeRegressor(
                    max_leaf_nodes=self.n_bins,
                )
            elif self.task == "classification":
                dt = DecisionTreeClassifier(
                    max_leaf_nodes=self.n_bins,
                )
            else:
                raise ValueError("This task is not supported")
            dt.fit(x_feat, y)
            self.conditions_.append(tree_to_code(dt, ["feature"]))

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        original_dim = np.shape(X)[1] if np.ndim(X) == 2 else 1
        X = check_array(X, dtype=np.float64, ensure_2d=True, ensure_all_finite=False)
        if X.shape[1] < original_dim:
            warnings.warn(
                "Some input features were dropped during check_array validation.",
                UserWarning,
            )

        all_transformed = []

        for col in range(X.shape[1]):
            feature = X[:, col]
            result_list = []
            for idx, cond in enumerate(self.conditions_[col]):
                result_list.append(eval(cond) * (idx + 1))

            encoded_feature = np.expand_dims(np.sum(np.stack(result_list).T, axis=1), 1)
            encoded_feature = np.array(encoded_feature - 1, dtype=np.int64)

            locations = []
            for string in self.conditions_[col]:
                matches = re.findall(self.pattern, string)
                locations.extend(matches)

            locations = [float(number) for number in locations]
            locations = list(set(locations))
            locations = np.sort(locations)

            ple_encoded_feature = np.zeros((len(feature), len(locations) + 1))
            if locations[-1] > np.max(feature):
                locations[-1] = np.max(feature)

            for idx in range(len(encoded_feature)):
                bin_idx = encoded_feature[idx][0]
                if feature[idx] >= locations[-1]:
                    ple_encoded_feature[idx][bin_idx] = feature[idx]
                    ple_encoded_feature[idx, :bin_idx] = 1
                elif feature[idx] <= locations[0]:
                    ple_encoded_feature[idx][bin_idx] = feature[idx]
                else:
                    ple_encoded_feature[idx][bin_idx] = (
                        feature[idx] - locations[bin_idx - 1]
                    ) / (locations[bin_idx] - locations[bin_idx - 1])
                    ple_encoded_feature[idx, :bin_idx] = 1

            if ple_encoded_feature.shape[1] == 1:
                ple_encoded_feature = np.zeros([len(feature), self.n_bins])

            all_transformed.append(ple_encoded_feature)

        return np.hstack(all_transformed).astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be specified")
        return input_features
