import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from ..utils.utils import center_identification_using_decision_tree


class ReLUExpansionTransformer(BaseEstimator, TransformerMixin):
    """
    Applies ReLU basis expansion to input features using fixed or data-driven center placement.

    This transformer expands each feature using a set of ReLU activation functions centered at fixed positions,
    which can be either uniformly/quantile spaced or determined by a decision tree based on the target.

    Parameters
    ----------
    n_centers : int, default=10
        Number of ReLU centers per feature.

    use_decision_tree : bool, default=True
        If True, uses a decision tree to determine center locations based on the input `X` and target `y`.

    task : {"regression", "classification"}, default="regression"
        Task type used for center selection when `use_decision_tree=True`.

    strategy : {"uniform", "quantile"}, default="uniform"
        Strategy used to determine center locations when `use_decision_tree=False`.

    Attributes
    ----------
    centers_ : list of ndarray
        A list of arrays containing the center locations for each input feature.

    Notes
    -----
    For a feature `x`, and centers `c`, this transformer produces `max(0, x - c_i)` for all `c_i` in centers.
    """

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
