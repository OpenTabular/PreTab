import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def center_identification_using_decision_tree(X, y, task="regression", n_centers=5):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, np.newaxis]

    centers = []
    for i in range(X.shape[1]):
        x_feat = X[:, [i]]
        if task == "classification":
            tree = DecisionTreeClassifier(max_leaf_nodes=n_centers + 1)
        elif task == "regression":
            tree = DecisionTreeRegressor(max_leaf_nodes=n_centers + 1)
        else:
            raise ValueError(
                "Invalid task type. Choose 'regression' or 'classification'."
            )
        tree.fit(x_feat, y)
        thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]
        centers.append(np.sort(thresholds))
    return centers
