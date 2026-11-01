"""Single source of NaN-aware 2D input validation.

Every transformer routes ``X`` through :func:`validate_2d_allow_nan` so the
"dropped columns" warning is emitted from exactly one place -- letting Python's
warning registry de-duplicate it instead of re-firing per transformer per
``transform`` -- and so ``n_features_in_`` is recorded consistently.
"""

import sys
import warnings
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array

from ..exceptions import DataWarning, IncompatibleParamsError, PretabDataError

__all__ = ["is_polars_frame", "polars_to_pandas", "single_target", "validate_2d_allow_nan"]


def single_target(y, estimator: str, *, fix: str | None = None) -> np.ndarray:
    """Return the target that target-aware placement fits on, as a 1D array.

    Locations are placed by one decision tree or boosting model fitted on a single
    target: a column vector is flattened, while a multi-output ``y`` raises instead
    of being flattened into ``n_samples * n_outputs`` values.

    Parameters
    ----------
    y : array-like
        Target values.
    estimator : str
        Class name used in the error message.
    fix : str, optional
        How to resolve a multi-output ``y``, appended to the error message.
        Defaults to fitting on one target column or switching to unsupervised
        placement (``target_aware=False``).

    Returns
    -------
    np.ndarray of shape (n_samples,)
        The flattened single target.

    Raises
    ------
    IncompatibleParamsError
        If ``y`` has more than one output column.
    """
    y = np.asarray(y)
    if y.ndim > 1 and int(np.prod(y.shape[1:])) != 1:
        if fix is None:
            fix = (
                "fit on one target column, or use unsupervised placement (target_aware=False) "
                "for a multi-output target."
            )
        raise IncompatibleParamsError(
            f"{estimator} places locations against a single target, but y has shape {y.shape}.\nFix: {fix}"
        )
    return y.ravel()


def is_polars_frame(X) -> bool:
    """Return True if ``X`` is a polars DataFrame, without importing polars.

    A polars object can only exist once polars has been imported, so when the
    module is absent from ``sys.modules`` ``X`` cannot be a polars frame.
    """
    polars = sys.modules.get("polars")
    return polars is not None and isinstance(X, polars.DataFrame)


def polars_to_pandas(X) -> pd.DataFrame:
    """Convert a polars DataFrame to pandas, keeping column names, order and dtypes.

    ``polars.DataFrame.to_pandas`` requires pyarrow, which neither polars nor
    PreTab depends on, so each column goes through ``Series.to_numpy`` instead:
    integer, unsigned, float, boolean and temporal columns keep their NumPy
    dtype (an integer column with nulls becomes float with NaN, as in pandas),
    while string, categorical and enum columns -- and boolean columns with nulls
    -- become ``object`` columns. Their nulls arrive as ``None`` and are mapped
    to ``NaN``, the missing marker the imputers and missing indicators recognize.
    Categorical and enum columns are first cast to strings so the native NumPy
    export also works on older Polars releases.

    Convert a frame once and take row subsets of the result: converted on its
    own, a subset can get another dtype (an integer column with a null is float
    as a whole, but int in a subset without the null).
    """
    polars = sys.modules["polars"]
    columns = {}
    for column in X.get_columns():
        if column.dtype in (polars.Categorical, polars.Enum):
            column = column.cast(str)
        values = column.to_numpy()
        if column.dtype == polars.Boolean and column.null_count() == 0:
            values = values.astype(bool, copy=False)
        if values.dtype == object:
            values = np.where(pd.isna(values), np.nan, values)
        columns[column.name] = values
    return pd.DataFrame(columns)


def validate_2d_allow_nan(X, *, allow_nan: bool = True, reset: bool, estimator):
    """Coerce ``X`` to a 2D float array, optionally letting NaNs pass through.

    Parameters
    ----------
    X : array-like
        Input data.
    allow_nan : bool, default=True
        When True, missing values are preserved so a later imputer can handle
        them; when False, NaN/inf values raise as usual.
    reset : bool
        When True (during ``fit``) record ``estimator.n_features_in_``; when
        False (during ``transform``) verify the feature count matches the value
        seen at ``fit`` and raise otherwise.
    estimator : object
        The calling transformer; used for the ``n_features_in_`` side effect and
        the transform-time feature-count check.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The validated float array.
    """
    input_shape = getattr(X, "shape", None)
    original_dim = input_shape[1] if input_shape is not None and len(input_shape) == 2 else None
    ensure_all_finite: Literal["allow-nan"] | bool = "allow-nan" if allow_nan else True
    X = check_array(
        X,
        dtype=np.float64,  # type: ignore
        ensure_2d=True,
        ensure_all_finite=ensure_all_finite,  # type: ignore
    )
    if original_dim is not None and X.shape[1] < original_dim:
        warnings.warn(
            "Some input features were dropped during check_array validation.",
            DataWarning,
            stacklevel=2,
        )
    if reset:
        estimator.n_features_in_ = X.shape[1]
    else:
        n_features_in_ = getattr(estimator, "n_features_in_", None)
        if n_features_in_ is not None and X.shape[1] != n_features_in_:
            raise PretabDataError(
                f"X has {X.shape[1]} features, but {type(estimator).__name__} "
                f"is expecting {n_features_in_} features as input."
            )
    return X
