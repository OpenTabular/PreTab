"""Single source of NaN-aware 2D input validation.

Every transformer routes ``X`` through :func:`validate_2d_allow_nan` so the
"dropped columns" warning is emitted from exactly one place -- letting Python's
warning registry de-duplicate it instead of re-firing per transformer per
``transform`` -- and so ``n_features_in_`` is recorded consistently.
"""

import warnings
from typing import Literal

import numpy as np
from sklearn.utils.validation import _check_feature_names, _check_feature_names_in, check_array

from ..exceptions import DataWarning, PretabDataError, invalid_param_error

__all__ = ["resolve_input_features", "validate_2d_allow_nan"]


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
    return [str(name) for name in _check_feature_names_in(estimator, input_features)]


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
