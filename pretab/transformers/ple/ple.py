import re
import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from .tree_to_code import tree_to_code


class PLETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=20, tree_params=None, task="regression", conditions=None):
        self.task = task
        self.tree_params = tree_params or {}
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
                    max_leaf_nodes=self.n_bins, **self.tree_params
                )
            elif self.task == "classification":
                dt = DecisionTreeClassifier(
                    max_leaf_nodes=self.n_bins, **self.tree_params
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
