import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .tree_to_code import tree_to_code


class PLETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=20, tree_params={}, task="regression", conditions=None, **kwargs):
        super().__init__(**kwargs)

        self.task = task
        self.tree_params = tree_params
        self.n_bins = n_bins
        self.conditions = conditions
        # This pattern matches integers and floats
        self.pattern = r"-?\d+\.?\d*[eE]?[+-]?\d*"

    def fit(self, feature, target):
        self.n_features_in_ = 1
        if self.task == "regression":
            dt = DecisionTreeRegressor(max_leaf_nodes=self.n_bins)
        elif self.task == "classification":
            dt = DecisionTreeClassifier(max_leaf_nodes=self.n_bins)
        else:
            raise ValueError("This task is not supported")

        dt.fit(feature, target)

        self.conditions = tree_to_code(dt, ["feature"])
        # self.fitted = True
        return self

    def transform(self, feature):
        if feature.shape == (feature.shape[0], 1):
            feature = np.squeeze(feature, axis=1)
        else:
            feature = feature
        result_list = []
        for idx, cond in enumerate(self.conditions):  # type: ignore
            result_list.append(eval(cond) * (idx + 1))  # type: ignore

        encoded_feature = np.expand_dims(np.sum(np.stack(result_list).T, axis=1), 1)

        encoded_feature = np.array(encoded_feature - 1, dtype=np.int64)

        # Initialize an empty list to store the extracted numbers
        locations = []
        # Iterate through the strings and extract numbers
        for string in self.conditions:  # type: ignore
            matches = re.findall(self.pattern, string)
            locations.extend(matches)

        locations = [float(number) for number in locations]
        locations = list(set(locations))
        locations = np.sort(locations)

        ple_encoded_feature = np.zeros((len(feature), locations.shape[0] + 1))
        if locations[-1] > np.max(feature):
            locations[-1] = np.max(feature)

        for idx in range(len(encoded_feature)):
            if feature[idx] >= locations[-1]:
                ple_encoded_feature[idx][encoded_feature[idx]] = feature[idx]
                ple_encoded_feature[idx, : encoded_feature[idx][0]] = 1
            elif feature[idx] <= locations[0]:
                ple_encoded_feature[idx][encoded_feature[idx]] = feature[idx]

            else:
                ple_encoded_feature[idx][encoded_feature[idx]] = (
                    feature[idx] - locations[(encoded_feature[idx] - 1)[0]]
                ) / (locations[(encoded_feature[idx])[0]] - locations[(encoded_feature[idx] - 1)[0]])

                ple_encoded_feature[idx, : encoded_feature[idx][0]] = 1

        if ple_encoded_feature.shape[1] == 1:
            return np.zeros([len(feature), self.n_bins])

        else:
            return np.array(ple_encoded_feature, dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be specified")
        return input_features
