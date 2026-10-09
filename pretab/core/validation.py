"""Single source of NaN-aware 2D input validation.

Every transformer routes ``X`` through :func:`validate_2d_allow_nan` so the
"dropped columns" warning is emitted from exactly one place -- letting Python's
warning registry de-duplicate it instead of re-firing per transformer per
``transform`` -- and so ``n_features_in_`` is recorded consistently.
"""

import sys
import warnings
from typing import Literal, cast

import numpy as np
import pandas as pd
from sklearn.utils.validation import _check_feature_names, _check_feature_names_in, check_array

from ..exceptions import DataWarning, IncompatibleParamsError, PretabDataError, invalid_param_error

__all__ = ["is_polars_frame", "polars_to_pandas", "resolve_input_features", "single_target", "validate_2d_allow_nan"]


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

    Convert a frame once and take row subsets of the result: converted on its
    own, a subset can get another dtype (an integer column with a null is float
    as a whole, but int in a subset without the null).
    """
    columns = {}
    for column in X.get_columns():
        values = column.to_numpy()
        if values.dtype == object:
            values = np.where(pd.isna(values), np.nan, values)
        columns[column.name] = values
    return pd.DataFrame(columns)


def resolve_input_features(estimator, input_features) -> list:
    """Return the input feature names that a transformer's output names are built from.

    Follows scikit-learn's ``get_feature_names_out`` contract: explicit
    ``input_features`` need one entry per input feature and, when the estimator was
    fitted on named (DataFrame) columns, must equal ``feature_names_in_``; ``None``
    falls back to ``feature_names_in_``, else to ``x0, x1, ...``.
    """
    n_features_in_ = getattr(estimator, "n_features_in_", None)
    if input_features is not None and n_features_in_ is not None and len(input_features) != n_features_in_:
        raise invalid_param_error(
            type(estimator).__name__,
            "get_feature_names_out.input_features",
            len(input_features),
            f"must have exactly {n_features_in_} entries (one per input feature)",
        )
    names = cast(np.ndarray, _check_feature_names_in(estimator, input_features))
    return [str(name) for name in names]


def single_target(y, estimator: str) -> np.ndarray:
    """Return the target that target-aware placement fits on, as a 1D array.

    Locations are placed by one decision tree or boosting model fitted on a single
    target: a column vector is flattened, while a multi-output ``y`` raises instead
    of being flattened into ``n_samples * n_outputs`` values.

    Raises
    ------
    IncompatibleParamsError
        If ``y`` has more than one output column.
    """
    y = np.asarray(y)
    if y.ndim > 1 and int(np.prod(y.shape[1:])) != 1:
        raise IncompatibleParamsError(
            f"{estimator} places locations against a single target, but y has shape {y.shape}.\n"
            "Fix: fit on one target column, or use unsupervised placement (target_aware=False) "
            "for a multi-output target."
        )
    return y.ravel()


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
        When True (during ``fit``) record ``estimator.n_features_in_`` and, for a
        DataFrame, ``estimator.feature_names_in_``; when False (during
        ``transform``) verify the feature count matches the value seen at ``fit``
        and raise otherwise, and check the column names as scikit-learn does
        (renamed or reordered columns raise).
    estimator : object
        The calling transformer; used for the ``n_features_in_`` side effect and
        the transform-time feature-count check.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The validated float array.
    """
    _check_feature_names(estimator, X, reset=reset)
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
