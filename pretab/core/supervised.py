"""Leakage-safe supervised contract: warning helper and cross-fitting wrapper.

Supervised (target-aware) representations place their basis using ``y``. Fitting
such a transformer on the full training data and then transforming that same
data leaks target information into the features. This module provides:

* :func:`warn_target_leakage` -- emits a :class:`~pretab.exceptions.LeakageWarning`
  when a supervised transformer is fit on ``(X, y)`` outside a controlled
  (Pipeline / cross-validation / cross-fitting) context.
* :class:`CrossFittedTransformer` -- wraps a supervised transformer and produces
  out-of-fold features during ``fit_transform`` so the training representation
  carries no target leakage, while ``transform`` uses a model fit on all data.
"""

import contextvars
import sys
import warnings
from dataclasses import replace

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from ..exceptions import (
    IncompatibleParamsError,
    InvalidParamError,
    LeakageWarning,
    PretabDataError,
)
from .representation import RepresentationSpecMixin

__all__ = ["CrossFittedTransformer", "in_controlled_context", "warn_target_leakage"]

# Module prefixes whose presence on the call stack marks a controlled context in
# which fitting a supervised transformer on ``(X, y)`` is expected and safe.
_CONTROLLED_MODULE_PREFIXES = (
    "sklearn.pipeline",
    "sklearn.model_selection",
    "sklearn.compose",
    "pretab.preprocessor",
    "pretab.compose",
)

# Set while :class:`CrossFittedTransformer` fits its internal clones, so their
# fits never emit a leakage warning.
_cross_fit_active: contextvars.ContextVar[bool] = contextvars.ContextVar("pretab_cross_fit_active", default=False)


def in_controlled_context() -> bool:
    """Return True when a Pipeline / CV / cross-fitting context is on the stack."""
    if _cross_fit_active.get():
        return True
    frame = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if module.startswith(_CONTROLLED_MODULE_PREFIXES):
            return True
        frame = frame.f_back
    return False


def warn_target_leakage(estimator, y) -> None:
    """Warn if a supervised ``estimator`` is fit on ``y`` outside a safe context.

    No warning is emitted when ``y`` is ``None``, when the estimator does not
    consume the target (``is_supervised`` is False), or when a controlled
    context (Pipeline, cross-validation, or :class:`CrossFittedTransformer`) is
    detected on the call stack.
    """
    if y is None:
        return
    if not getattr(estimator, "is_supervised", False):
        return
    if in_controlled_context():
        return
    warnings.warn(
        f"{type(estimator).__name__} is target-aware and was fit on (X, y) outside a "
        "Pipeline / cross-validation context, which can leak target information into "
        "the features. Fit it inside a scikit-learn Pipeline, wrap it in "
        "pretab.CrossFittedTransformer, or ignore this warning if the fitted data "
        "will not be reused to train a downstream model.",
        LeakageWarning,
        stacklevel=3,
    )


class CrossFittedTransformer(RepresentationSpecMixin, TransformerMixin, BaseEstimator):
    """Cross-fit a supervised transformer to remove target leakage on training data.

    During :meth:`fit_transform`, the wrapped transformer is fit on each
    training fold and used to transform the held-out fold, so every training row
    is encoded by a model that never saw its own target. :meth:`transform`
    (for unseen data) uses ``estimator_``, a single transformer fit on all data.

    Parameters
    ----------
    transformer : estimator
        A supervised (target-aware) PreTab transformer to cross-fit.
    n_folds : int, default=5
        Number of cross-fitting folds. Must be at least 2.
    task : {"regression", "classification"}, default="regression"
        Controls the splitter: ``KFold`` for regression, ``StratifiedKFold`` for
        classification.
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    random_state : int or None, default=None
        Seed used when ``shuffle`` is True.

    Attributes
    ----------
    estimator_ : estimator
        The transformer fit on all of ``(X, y)``, used by :meth:`transform`.
    n_features_in_ : int
        Number of input features seen during ``fit``.
    """

    _representation_supervision = "supervised"

    def __init__(self, transformer, n_folds=5, task="regression", shuffle=True, random_state=None):
        self.transformer = transformer
        self.n_folds = n_folds
        self.task = task
        self.shuffle = shuffle
        self.random_state = random_state

    def _make_splitter(self):
        """Return the cross-fitting splitter for the configured task."""
        seed = self.random_state if self.shuffle else None
        if self.task == "classification":
            return StratifiedKFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=seed)
        return KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=seed)

    def _fit_full(self, X, y):
        """Validate inputs and fit ``estimator_`` on all data; return arrays."""
        if y is None:
            raise IncompatibleParamsError("CrossFittedTransformer requires y at fit time; got y=None.")
        if not isinstance(self.n_folds, (int, np.integer)) or self.n_folds < 2:
            raise InvalidParamError(f"n_folds must be an integer >= 2; got {self.n_folds!r}.")
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y).ravel()
        if len(X_arr) != len(y_arr):
            raise PretabDataError(f"X and y must have same length. Got {len(X_arr)} and {len(y_arr)}")
        estimator = clone(self.transformer)
        token = _cross_fit_active.set(True)
        try:
            estimator.fit(X_arr, y_arr)
        finally:
            _cross_fit_active.reset(token)
        self.estimator_ = estimator
        self.n_features_in_ = X_arr.shape[1]
        return X_arr, y_arr

    def fit(self, X, y=None):
        """Fit the all-data ``estimator_`` used by :meth:`transform`."""
        self._fit_full(X, y)
        return self

    def transform(self, X):
        """Transform ``X`` using the transformer fit on all training data."""
        check_is_fitted(self, "estimator_")
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return self.estimator_.transform(X_arr)

    def fit_transform(self, X, y=None):
        """Fit and return leakage-free out-of-fold features for the training data."""
        X_arr, y_arr = self._fit_full(X, y)
        width = len(self.estimator_.get_feature_names_out())
        out = np.empty((X_arr.shape[0], width), dtype=float)
        splitter = self._make_splitter()
        token = _cross_fit_active.set(True)
        try:
            for train_idx, test_idx in splitter.split(X_arr, y_arr):
                fold = clone(self.transformer)
                fold.fit(X_arr[train_idx], y_arr[train_idx])
                fold_out = np.asarray(fold.transform(X_arr[test_idx]))
                if fold_out.shape[1] != width:
                    raise IncompatibleParamsError(
                        "Cross-fitting requires a fixed output width across folds; expected "
                        f"{width}, got {fold_out.shape[1]}. Disable adaptive sizing on the "
                        "wrapped transformer."
                    )
                out[test_idx] = fold_out
        finally:
            _cross_fit_active.reset(token)
        return out

    def get_feature_names_out(self, input_features=None):
        """Delegate output feature names to the all-data ``estimator_``."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.get_feature_names_out(input_features)

    def get_representation_spec(self, input_features=None):
        """Return the wrapped spec, flagged as cross-fitted."""
        check_is_fitted(self, "estimator_")
        if hasattr(self.estimator_, "get_representation_spec"):
            base = self.estimator_.get_representation_spec(input_features)
            return replace(base, uses_target=True, cross_fitted=True, n_folds=int(self.n_folds))
        return super().get_representation_spec(input_features)

    def _representation_cross_fitting(self):
        """Report cross-fitting metadata for the spec fallback path."""
        return True, int(self.n_folds)
