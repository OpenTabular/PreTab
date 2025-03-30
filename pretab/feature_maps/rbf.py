import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class RBFTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, n_centers=10, gamma: float = 1.0, use_decision_tree=True, task: str = "regression", strategy="uniform"
    ):
        """
        Radial Basis Function Expansion.

        Parameters:
        - n_centers: Number of RBF centers.
        - gamma: Width of the RBF kernel.
        - use_decision_tree: If True, use a decision tree to determine RBF centers.
        - task: Task type, 'regression' or 'classification'.
        - strategy: If 'uniform', centers are uniformly spaced. If 'quantile', centers are
                    determined by data quantile.
        """
        self.n_centers = n_centers
        self.gamma = gamma
        self.use_decision_tree = use_decision_tree
        self.strategy = strategy
        self.task = task

        if self.strategy not in ["uniform", "quantile"]:
            raise ValueError("Invalid strategy. Choose 'uniform' or 'quantile'.")

    def fit(self, X, y=None):
        X = check_array(X)

        if self.use_decision_tree and y is None:
            raise ValueError("Target variable 'y' must be provided when use_decision_tree=True.")

        if self.use_decision_tree:
            self.centers_ = center_identification_using_decision_tree(X, y, self.task, self.n_centers)
            self.centers_ = np.vstack(self.centers_)
        else:
            # Compute centers
            if self.strategy == "quantile":
                self.centers_ = np.percentile(X, np.linspace(0, 100, self.n_centers), axis=0)
            elif self.strategy == "uniform":
                self.centers_ = np.linspace(X.min(axis=0), X.max(axis=0), self.n_centers)

        # Compute gamma if not provided
        # if self.gamma is None:
        #     dists = pairwise_distances(self.centers_)
        #     self.gamma = 1 / (2 * np.mean(dists[dists > 0]) ** 2)  # Mean pairwise distance
        return self

    def transform(self, X):
        X = check_array(X)
        transformed = []
        self.centers_ = np.array(self.centers_)
        for center in self.centers_.T:
            rbf_features = np.exp(-self.gamma * (X - center) ** 2)  # type: ignore
            transformed.append(rbf_features)
        return np.hstack(transformed)


