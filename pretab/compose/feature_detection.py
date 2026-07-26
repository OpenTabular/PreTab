"""Coerce inputs to DataFrames and classify columns as numerical or categorical.

Feature-type detection decides which construction path each column takes. It is
kept here, separate from orchestration, so the Preprocessor's ``fit`` reads as a
sequence of delegations rather than inlining the classification heuristic.
"""

import numpy as np
import pandas as pd

from ..exceptions import invalid_param_error

__all__ = ["detect_column_types", "to_dataframe"]


def to_dataframe(X, *, copy: bool = False) -> pd.DataFrame:
    """Return ``X`` as a DataFrame, naming array columns ``feature_0``, ``feature_1`` ....

    Dicts and NumPy arrays are wrapped in a fresh DataFrame; an existing
    DataFrame is returned as-is, or copied when ``copy`` is True.
    """
    if isinstance(X, dict):
        return pd.DataFrame(X)
    if isinstance(X, np.ndarray):
        return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return X.copy() if copy else X


def detect_column_types(X, *, cat_cutoff, treat_all_integers_as_numerical, estimator_name="Preprocessor"):
    """Classify each column of ``X`` as numerical or categorical.

    An integer column is treated as categorical when its cardinality falls below
    ``cat_cutoff`` -- interpreted as a unique-ratio cutoff when a float, or an
    absolute unique-count cutoff when an int. Non-numeric dtypes are always
    categorical; ``treat_all_integers_as_numerical`` bypasses the heuristic for
    integer columns.

    Returns
    -------
    numerical_features : list
        Column labels detected as numerical.
    categorical_features : list
        Column labels detected as categorical.
    """
    X = to_dataframe(X)

    categorical_features = []
    numerical_features = []

    for col in X.columns:
        num_unique_values = X[col].nunique()
        total_samples = len(X[col])

        if treat_all_integers_as_numerical and X[col].dtype.kind == "i":
            numerical_features.append(col)
        else:
            if isinstance(cat_cutoff, float):
                cutoff_condition = (num_unique_values / total_samples) < cat_cutoff
            elif isinstance(cat_cutoff, int):
                cutoff_condition = num_unique_values < cat_cutoff
            else:
                raise invalid_param_error(
                    estimator_name,
                    "cat_cutoff",
                    cat_cutoff,
                    "must be a float (unique-ratio cutoff) or an int (absolute unique-count cutoff)",
                )

            if X[col].dtype.kind not in "iufc" or (X[col].dtype.kind == "i" and cutoff_condition):
                categorical_features.append(col)
            else:
                numerical_features.append(col)

    return numerical_features, categorical_features
