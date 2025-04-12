import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from ..utils.utils import center_identification_using_decision_tree
import warnings


class RBFExpansionTransformer(BaseEstimator, TransformerMixin):
    """
    Radial Basis Function (RBF) feature expansion for numerical tabular data.

    This transformer expands each feature into a set of RBF (Gaussian) basis functions
    centered at fixed points. The centers can be determined either by a decision tree
    (based on supervised splits) or based on quantiles or uniform spacing.

    Parameters
    ----------
    n_centers : int, default=10
        Number of RBF centers per feature.

    gamma : float, default=1.0
        Width parameter of the RBF kernel. Larger values make the kernel narrower.

    use_decision_tree : bool, default=True
        Whether to use a decision tree to select center locations based on `y`.

    task : {"regression", "classification"}, default="regression"
        Type of task for the decision tree used to find center locations.

    strategy : {"uniform", "quantile"}, default="uniform"
        Strategy for choosing centers when not using a decision tree.

    Attributes
    ----------
    centers_ : list of ndarray
        List of arrays containing center locations for each feature.

    Examples
    --------
    >>> from prefab.transformers import RBFExpansionTransformer
    >>> import numpy as np
    >>> X = np.array([[1.], [2.], [3.]])
    >>> transformer = RBFExpansionTransformer(n_centers=3, gamma=0.5, use_decision_tree=False)
    >>> transformer.fit(X)
    RBFExpansionTransformer(...)
    >>> transformer.transform(X).shape
    (3, 3)
    """

    def __init__(
        self,
        n_centers=10,
        gamma: float = 1.0,
        use_decision_tree=True,
        task: str = "regression",
        strategy="uniform",
    ):
        self.n_centers = n_centers
        self.gamma = gamma
        self.use_decision_tree = use_decision_tree
        self.strategy = strategy
        self.task = task

        if self.strategy not in ["uniform", "quantile"]:
            raise ValueError("Invalid strategy. Choose 'uniform' or 'quantile'.")

        if self.task not in ["regression", "classification"]:
            raise ValueError("Invalid task. Choose 'regression' or 'classification'.")

    def fit(self, X, y=None):
        X = check_array(X, dtype=np.float64)

        if not np.issubdtype(X.dtype, np.floating):
            raise ValueError("Input X must be of float type.")

        if self.use_decision_tree and y is None:
            raise ValueError(
                "Target variable 'y' must be provided when use_decision_tree=True."
            )

        self.centers_ = []

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
            else:  # uniform
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

        transformed = []
        for i in range(X.shape[1]):
            centers = np.asarray(self.centers_[i])
            # shape: (n_samples, n_centers)
            rbf_feats = np.exp(-self.gamma * (X[:, [i]] - centers[np.newaxis, :]) ** 2)
            transformed.append(rbf_feats)

        return np.hstack(transformed)
