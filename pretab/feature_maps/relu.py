import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

class ReLUExpansionTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, n_centers=10, use_decision_tree=True, task: str = "regression", strategy="uniform"
    ):
        """
        ReLU Basis Expansion.

        Parameters:
        - n_centers: Number of ReLU centers.
        - use_decision_tree: If True, use decision tree to determine center positions.
        - task: Task type ('regression' or 'classification').
        - strategy: 'uniform' or 'quantile' center placement if not using tree.
        """
        self.n_centers = n_centers
        self.use_decision_tree = use_decision_tree
        self.strategy = strategy
        self.task = task

    def fit(self, X, y=None):
        X = check_array(X)

        if self.use_decision_tree and y is None:
            raise ValueError("Target variable 'y' must be provided when use_decision_tree=True.")

        if self.use_decision_tree:
            self.centers_ = center_identification_using_decision_tree(X, y, self.task, self.n_centers)
            self.centers_ = np.vstack(self.centers_)
        else:
            if self.strategy == "quantile":
                self.centers_ = np.percentile(X, np.linspace(0, 100, self.n_centers), axis=0)
            elif self.strategy == "uniform":
                self.centers_ = np.linspace(X.min(axis=0), X.max(axis=0), self.n_centers)

        return self

    def transform(self, X):
        X = check_array(X)
        transformed = []
        self.centers_ = np.array(self.centers_)

        for center in self.centers_.T:
            relu_features = np.maximum(0, X - center)
            transformed.append(relu_features)

        return np.hstack(transformed)
