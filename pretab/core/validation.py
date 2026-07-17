"""Single source of NaN-aware 2D input validation.

Every transformer routes ``X`` through :func:`validate_2d_allow_nan` so the
"dropped columns" warning is emitted from exactly one place -- letting Python's
warning registry de-duplicate it instead of re-firing per transformer per
``transform`` -- and so ``n_features_in_`` is recorded consistently.
"""

import warnings
from typing import Literal

import numpy as np
from sklearn.utils.validation import check_array

from .exceptions import DataWarning

__all__ = ["validate_2d_allow_nan"]


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
        When True (during ``fit``) record ``estimator.n_features_in_``.
    estimator : object
        The calling transformer; used only for the ``n_features_in_`` side effect.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The validated float array.
    """
    original_dim = np.shape(X)[1] if np.ndim(X) == 2 else 1
    ensure_all_finite: Literal["allow-nan"] | bool = "allow-nan" if allow_nan else True
    X = check_array(
        X, dtype=np.float64, ensure_2d=True, ensure_all_finite=ensure_all_finite  # type: ignore
    )
    if X.shape[1] < original_dim:
        warnings.warn(
            "Some input features were dropped during check_array validation.",
            DataWarning,
            stacklevel=2,
        )
    if reset:
        estimator.n_features_in_ = X.shape[1]
    return X
