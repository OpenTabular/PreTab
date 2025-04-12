import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from ..utils.utils import center_identification_using_decision_tree


class ReLUExpansionTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_centers=10,
        use_decision_tree=True,
        task: str = "regression",
        strategy="uniform",
    ):
        self.n_centers = n_centers
        self.use_decision_tree = use_decision_tree
        self.strategy = strategy
        self.task = task

        if self.strategy not in ["uniform", "quantile"]:
            raise ValueError("Invalid strategy. Choose 'uniform' or 'quantile'.")

        if self.task not in ["regression", "classification"]:
            raise ValueError("Invalid task. Choose 'regression' or 'classification'.")

    def fit(self, X, y=None):
        original_dim = np.shape(X)[1] if np.ndim(X) == 2 else 1
        X = check_array(
            X, dtype=np.float64, ensure_2d=True, ensure_all_finite="allow-nan"
        )

        if X.shape[1] < original_dim:
            warnings.warn(
                "Some input features were dropped during check_array validation.",
                UserWarning,
            )

        if self.use_decision_tree and y is None:
            raise ValueError(
                "Target variable 'y' must be provided when use_decision_tree=True."
            )

        if self.use_decision_tree:
            centers_list = center_identification_using_decision_tree(
                X, y, self.task, self.n_centers
            )
        else:
            if self.strategy == "quantile":
                centers_list = [
                    np.percentile(X[:, i], np.linspace(0, 100, self.n_centers))
                    for i in range(X.shape[1])
                ]
            else:
                centers_list = [
                    np.linspace(X[:, i].min(), X[:, i].max(), self.n_centers)
                    for i in range(X.shape[1])
                ]

        self.centers_ = centers_list
        return self

    def transform(self, X):
        original_dim = np.shape(X)[1] if np.ndim(X) == 2 else 1
        X = check_array(
            X, dtype=np.float64, ensure_2d=True, ensure_all_finite="allow-nan"
        )
        if X.shape[1] < original_dim:
            warnings.warn(
                "Some input features were dropped during check_array validation.",
                UserWarning,
            )

        if len(self.centers_) != X.shape[1]:
            raise ValueError("X and centers must have the same number of features.")

        relu_outputs = []
        for i in range(X.shape[1]):
            centers = np.asarray(self.centers_[i])
            relu_feats = np.maximum(0, X[:, [i]] - centers[np.newaxis, :])
            relu_outputs.append(relu_feats)

        return np.hstack(relu_outputs)
