"""Coerce inputs to pandas DataFrames and classify columns as numerical or categorical.

Feature-type detection decides which construction path each column takes. It is
kept here, separate from orchestration, so the Preprocessor's ``fit`` reads as a
sequence of delegations rather than inlining the classification heuristic.
"""

from collections import Counter

import numpy as np
import pandas as pd

from ..core.validation import is_polars_frame
from ..exceptions import PretabDataError, invalid_param_error

__all__ = [
    "bool_columns_as_object",
    "detect_column_types",
    "has_column_labels",
    "to_dataframe",
    "with_string_labels",
]


def _polars_to_pandas(X) -> pd.DataFrame:
    """Convert a polars DataFrame to pandas, keeping column names, order and dtypes.

    ``polars.DataFrame.to_pandas`` requires pyarrow, which neither polars nor
    PreTab depends on, so each column goes through ``Series.to_numpy`` instead:
    integer, unsigned, float, boolean and temporal columns keep their NumPy
    dtype (an integer column with nulls becomes float with NaN, as in pandas),
    while string, categorical and enum columns -- and boolean columns with nulls
    -- become ``object`` columns. Their nulls arrive as ``None`` and are mapped
    to ``NaN``, the missing marker the imputers and missing indicators recognize.
    """
    columns = {}
    for column in X.get_columns():
        values = column.to_numpy()
        if values.dtype == object:
            values = np.where(pd.isna(values), np.nan, values)
        columns[column.name] = values
    return pd.DataFrame(columns)


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
        X = _polars_to_pandas(X)
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


def with_string_labels(X: pd.DataFrame) -> pd.DataFrame:
    """Return ``X`` with every column label converted to its ``str`` form.

    scikit-learn's :class:`~sklearn.compose.ColumnTransformer` treats an integer
    column selector as a *position*, so routing a column by an integer label (as
    in ``pd.DataFrame(array)`` or ``read_csv(header=None)`` after reordering or
    dropping a column) would select the wrong column. The Preprocessor therefore
    fits and transforms its ColumnTransformer on string labels. ``X`` itself is
    returned when every label is already a string; otherwise a shallow copy with
    relabelled columns, so the caller's frame is never modified.

    Raises
    ------
    PretabDataError
        If two labels share a string form (e.g. ``1`` and ``"1"``), which would
        make the columns indistinguishable.
    """
    if all(isinstance(label, str) for label in X.columns):
        return X
    labels = [str(label) for label in X.columns]
    collisions = sorted(label for label, count in Counter(labels).items() if count > 1)
    if collisions:
        raise PretabDataError(
            f"Column labels must stay unique when converted to strings; {collisions} would collide.\n"
            "Fix: rename the columns so their string forms are unique."
        )
    relabelled = X.copy(deep=False)
    relabelled.columns = pd.Index(labels)
    return relabelled


def bool_columns_as_object(X: pd.DataFrame) -> pd.DataFrame:
    """Return ``X`` with its boolean columns cast to ``object``.

    Boolean columns are categorical (see :func:`detect_column_types`), but the
    categorical pipeline starts with a :class:`~sklearn.impute.SimpleImputer`,
    which rejects the ``bool`` dtype. As ``object`` columns of ``True`` /
    ``False`` they are encoded like any other binary categorical column; a
    missing value of pandas' nullable ``boolean`` dtype becomes ``NaN``, the
    missing marker the imputer recognizes. ``X`` itself is returned when it has
    no boolean column, so the caller's frame is never modified.
    """
    bool_columns = [label for label, dtype in X.dtypes.items() if dtype.kind == "b"]
    if not bool_columns:
        return X
    cast = X.copy(deep=False)
    for label in bool_columns:
        column = X[label].astype(object)
        cast[label] = column.where(column.notna(), np.nan)
    return cast


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
