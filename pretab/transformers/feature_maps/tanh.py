import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from ..utils.utils import center_identification_using_decision_tree


class TanhExpansionTransformer(BaseEstimator, TransformerMixin):
    """
    Applies hyperbolic tangent (tanh) basis expansion to input features using specified or learned center locations.

    This transformer expands each input feature into multiple tanh-activated features, useful for capturing
    nonlinear and saturating patterns in the data.

    Parameters
    ----------
    n_centers : int, default=10
        Number of tanh centers per feature.

    scale : float, default=1.0
        Controls the sharpness of the tanh transitions. Smaller values make the activation sharper.

    use_decision_tree : bool, default=True
        If True, uses a decision tree to determine the tanh center locations based on the input `X` and target `y`.

    task : {"regression", "classification"}, default="regression"
        Type of prediction task. Required for decision tree-based center selection.

    strategy : {"uniform", "quantile"}, default="uniform"
        Strategy to determine center placement when `use_decision_tree=False`.

    Attributes
    ----------
    centers_ : list of ndarray
        A list of center values for each input feature used in the tanh expansion.

    Notes
    -----
    Each original feature `x` is transformed into `n_centers` features of the form:
        tanh((x - c) / scale)

    where `c` is a center value and `scale` controls the spread of the activation.
    """

    def __init__(
        self,
        n_centers=10,
        scale: float = 1.0,
        use_decision_tree=True,
        task: str = "regression",
        strategy="uniform",
    ):
        self.n_centers = n_centers
        self.scale = scale
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
            tanh_feats = np.tanh((X[:, [i]] - centers[np.newaxis, :]) / self.scale)
            transformed.append(tanh_feats)

        return np.hstack(transformed)
