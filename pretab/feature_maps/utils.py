import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def center_identification_using_decision_tree(X, y, task="regression", n_centers=5):
    # Use DecisionTreeClassifier for classification tasks
    centers = []
    if task == "classification":
        tree = DecisionTreeClassifier(max_leaf_nodes=n_centers + 1)
    elif task == "regression":
        tree = DecisionTreeRegressor(max_leaf_nodes=n_centers + 1)
    else:
        raise ValueError(
            "Invalid task type. Choose 'regression' or 'classification'.")
    tree.fit(X, y)
    # Extract thresholds from the decision tree
    # type: ignore
    thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]
    centers.append(np.sort(thresholds))
    return centers
