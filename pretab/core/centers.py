"""Decision-tree center identification shared by the feature-map expansions.

The RBF / ReLU / sigmoid / tanh transformers all place their basis centers at the
split thresholds of a shallow decision tree fitted per feature. That logic lives
here in ``core`` so every feature map consumes one implementation.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .exceptions import InvalidParamError

__all__ = ["center_identification_using_decision_tree"]


def center_identification_using_decision_tree(X, y, task="regression", n_centers=5, random_state=None):
    """Return per-feature center locations from decision-tree split thresholds.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data. A 1D array is treated as a single feature.
    y : array-like of shape (n_samples,)
        Target used to fit the per-feature decision trees.
    task : {"regression", "classification"}, default="regression"
        Which decision-tree estimator to fit.
    n_centers : int, default=5
        Target number of centers per feature (the tree uses
        ``max_leaf_nodes = n_centers + 1``).
    random_state : int or None, default=None
        Seed for the per-feature decision trees, controlling split tie-breaking
        so center placement is reproducible.

    Returns
    -------
    centers : list of ndarray
        One sorted array of split thresholds per input feature.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, np.newaxis]

    centers = []
    for i in range(X.shape[1]):
        x_feat = X[:, [i]]
        if task == "classification":
            tree = DecisionTreeClassifier(max_leaf_nodes=n_centers + 1, random_state=random_state)
        elif task == "regression":
            tree = DecisionTreeRegressor(max_leaf_nodes=n_centers + 1, random_state=random_state)
        else:
            raise InvalidParamError(
                f"Invalid task. Choose 'regression' or 'classification'. Got {task!r}."
            )
        tree.fit(x_feat, y)
        thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]
        centers.append(np.sort(thresholds))
    return centers
