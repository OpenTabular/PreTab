"""Coerce inputs to pandas DataFrames and classify columns as numerical or categorical.

Feature-type detection decides which construction path each column takes. It is
kept here, separate from orchestration, so the Preprocessor's ``fit`` reads as a
sequence of delegations rather than inlining the classification heuristic.
"""

import numpy as np
import pandas as pd

from ..core.validation import is_polars_frame, polars_to_pandas
from ..exceptions import PretabDataError, invalid_param_error

__all__ = ["detect_column_types", "has_column_labels", "to_dataframe"]


def has_column_labels(X) -> bool:
    """Return True if ``X`` carries its own column labels.

    DataFrames (pandas or polars) and dicts of columns are matched by label;
    anything else (a NumPy array, a list of rows, ...) is matched by position.
    """
    return isinstance(X, dict) or hasattr(X, "columns")


def to_dataframe(X, *, copy: bool = False) -> pd.DataFrame:
    """Return ``X`` as a pandas DataFrame, naming array columns ``feature_0``, ``feature_1`` ....

    Dicts are wrapped in a fresh DataFrame and a polars DataFrame is converted to
    one with the same column names, order and (as far as possible) dtypes. Input
    without column labels -- a NumPy array, a list of rows -- is read as a 2D
    array. An existing pandas DataFrame is returned as-is, or copied when
    ``copy`` is True.

    Raises
    ------
    PretabDataError
        If ``X`` is a pandas Series or array-like input that is not 2D, or if the
        resulting columns contain a duplicate label. A
        :class:`~sklearn.compose.ColumnTransformer` keys its per-column steps by
        name, so a duplicate cannot be routed unambiguously.
    """
    if isinstance(X, pd.Series):
        raise PretabDataError(
            "Expected 2D input (a DataFrame or a 2D array), got a pandas Series.\n"
            "Fix: pass X.to_frame() to use the Series as a single column."
        )
    if isinstance(X, dict):
        X = pd.DataFrame(X)
    elif is_polars_frame(X):
        X = polars_to_pandas(X)
    elif not has_column_labels(X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise PretabDataError(
                f"Expected 2D input (a DataFrame or a 2D array), got an array with {X.ndim} dimension(s).\n"
                "Fix: reshape a single feature to one column, e.g. X.reshape(-1, 1)."
            )
        X = pd.DataFrame(X, columns=pd.Index([f"feature_{i}" for i in range(X.shape[1])]))
    else:
        X = X.copy() if copy else X

    duplicated = X.columns[X.columns.duplicated()].unique().tolist()
    if duplicated:
        raise PretabDataError(
            f"Duplicate column names are not supported: {duplicated}.\nFix: rename the columns so every name is unique."
        )
    return X


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

        if treat_all_integers_as_numerical and X[col].dtype.kind in "iu":
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

            if X[col].dtype.kind not in "iufc" or (X[col].dtype.kind in "iu" and cutoff_condition):
                categorical_features.append(col)
            else:
                numerical_features.append(col)

    return numerical_features, categorical_features
