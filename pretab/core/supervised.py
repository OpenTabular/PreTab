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
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from ..exceptions import (
    IncompatibleParamsError,
    InvalidParamError,
    LeakageWarning,
    PretabDataError,
)
from ._typing import TransformerLike
from .parameters import validate_task
from .representation import RepresentationSpecMixin
from .validation import is_polars_frame, polars_to_pandas

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


def _as_2d(X):
    """Return ``X`` as 2D input for the wrapped transformer.

    A pandas DataFrame is passed through unchanged, so column names and per-column
    dtypes reach the wrapped estimator (e.g. a Preprocessor detecting numerical
    vs categorical columns, or a name-based ColumnTransformer). A polars
    DataFrame is converted to pandas once, so every fold's rows carry the dtypes
    of the whole frame. Any other input is converted to a NumPy array, with 1D
    input reshaped to a single column.
    """
    if is_polars_frame(X):
        return polars_to_pandas(X)
    if hasattr(X, "iloc") and getattr(X, "ndim", None) == 2:
        return X
    X = np.asarray(X)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def _take_rows(X, indices):
    """Select rows of a DataFrame or array by position."""
    return X.iloc[indices] if hasattr(X, "iloc") else X[indices]


def _stack_folds(blocks, test_indices, X) -> Any:
    """Stack the out-of-fold ``blocks`` and put their rows back in the order of ``X``.

    ``blocks[i]`` holds the transformed rows ``test_indices[i]``. The result keeps
    the kind of output the wrapped transformer returns: sparse blocks stay sparse
    (in the format of the first block), pandas / polars DataFrames keep their
    columns and dtypes, and dense blocks take their common dtype (``object`` for
    strings left unchanged). A pandas DataFrame keeps the index of ``X`` when the
    blocks carry it (scikit-learn's ``set_output`` copies the input index);
    otherwise it is renumbered from 0, as ``transform`` numbers the rows of a
    transformer that builds its own index (such as a Preprocessor).
    """
    rows = np.concatenate(test_indices)
    # order[i] is the position of input row i among the stacked fold rows.
    order = np.empty_like(rows)
    order[rows] = np.arange(len(rows))
    first = blocks[0]
    if all(sp.issparse(block) for block in blocks):
        return sp.vstack(blocks).tocsr()[order].asformat(first.format)
    if all(isinstance(block, pd.DataFrame) for block in blocks):
        stacked = pd.concat(blocks).iloc[order]
        index = getattr(X, "index", None)
        return stacked if index is not None and stacked.index.equals(index) else stacked.reset_index(drop=True)
    if type(first).__module__.partition(".")[0] == "polars":
        import polars as pl  # type: ignore

        return pl.concat(blocks)[order]
    dense = [block.toarray() if sp.issparse(block) else np.asarray(block) for block in blocks]
    out = np.empty((len(rows), first.shape[1]), dtype=np.result_type(*dense))
    for block, test_idx in zip(dense, test_indices, strict=True):
        out[test_idx] = block
    return out


class CrossFittedTransformer(RepresentationSpecMixin, TransformerMixin, BaseEstimator):
    """Cross-fit a supervised transformer to remove target leakage on training data.

    During :meth:`fit_transform`, the wrapped transformer is fit on each
    training fold and used to transform the held-out fold, so every training row
    is encoded by a model that never saw its own target. :meth:`transform`
    (for unseen data) uses ``estimator_``, a single transformer fit on all data.
    Both return the wrapped transformer's kind of output: a dense array of the
    same dtype, a sparse matrix in the same format, or a DataFrame.

    Parameters
    ----------
    transformer : estimator
        A supervised (target-aware) PreTab transformer to cross-fit, or an
        estimator such as a :class:`~pretab.Preprocessor` or ``Pipeline`` built from
        them. A pandas DataFrame ``X`` is passed to it unchanged (each fold is a row
        subset), so column names and dtypes are preserved. For a
        :class:`~pretab.Preprocessor`, only its target-aware blocks are refit per
        fold: column types, category codes and the blocks that do not use ``y``
        come from the all-data fit, so the out-of-fold features use exactly the
        encoding of :meth:`transform`.
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
        validate_task(self.task, type(self).__name__)
        X_arr = _as_2d(X)
        y_arr = np.asarray(y).ravel()
        if len(X_arr) != len(y_arr):
            raise PretabDataError(f"X and y must have same length. Got {len(X_arr)} and {len(y_arr)}")
        estimator = cast(TransformerLike, clone(self.transformer))
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
        return self.estimator_.transform(_as_2d(X))

    def _fit_fold(self, X, y):
        """Return the model that encodes one held-out fold, fit on the other rows.

        A wrapped estimator that implements ``_cross_fit_fold`` (the
        :class:`~pretab.Preprocessor`) derives the fold model from ``estimator_``
        and refits only its target-aware parts, so the out-of-fold features share
        the encoding ``transform`` uses. Any other estimator is cloned and refit.
        """
        cross_fit_fold = getattr(self.estimator_, "_cross_fit_fold", None)
        if cross_fit_fold is not None:
            return cast(TransformerLike, cross_fit_fold(X, y))
        fold = cast(TransformerLike, clone(self.transformer))
        fold.fit(X, y)
        return fold

    def fit_transform(self, X, y=None):
        """Fit and return leakage-free out-of-fold features for the training data.

        Returns
        -------
        array-like of shape (n_samples, n_features_out)
            The out-of-fold features, in the row order of ``X`` and in the kind of
            output :meth:`transform` returns: a sparse matrix in the same format, a
            DataFrame, or a dense array with the folds' dtype (``object`` when the
            wrapped transformer leaves strings unchanged).
        """
        X_arr, y_arr = self._fit_full(X, y)
        names = [str(name) for name in self.estimator_.get_feature_names_out()]
        width = len(names)
        blocks, test_indices = [], []
        splitter = self._make_splitter()
        token = _cross_fit_active.set(True)
        try:
            for train_idx, test_idx in splitter.split(X_arr, y_arr):
                fold = self._fit_fold(_take_rows(X_arr, train_idx), y_arr[train_idx])
                fold_out = fold.transform(_take_rows(X_arr, test_idx))
                if isinstance(fold_out, dict):
                    raise IncompatibleParamsError(
                        "Cross-fitting stacks the out-of-fold features into one array; the "
                        "wrapped transformer returned a dict of blocks. Use output_structure="
                        "'matrix' on a wrapped Preprocessor."
                    )
                if not hasattr(fold_out, "shape"):
                    fold_out = np.asarray(fold_out)
                if fold_out.shape[1] != width:
                    raise IncompatibleParamsError(
                        "Cross-fitting requires a fixed output width across folds; expected "
                        f"{width}, got {fold_out.shape[1]}. The wrapped transformer's width "
                        "depends on the rows it is fit on (e.g. adaptive sizing, or category / "
                        "indicator columns learned from the data); configure it to produce the "
                        "same columns on every fold."
                    )
                # DataFrames are stacked by column label, so a fold that kept other
                # columns at the same width (e.g. other top categories) cannot be aligned.
                columns = getattr(fold_out, "columns", None)
                if columns is not None and [str(column) for column in columns] != names:
                    differing = sorted(set(map(str, columns)).symmetric_difference(names))
                    raise IncompatibleParamsError(
                        "Cross-fitting requires the same output columns across folds; a fold's "
                        f"columns differ from the all-data fit in {differing}. The wrapped "
                        "transformer learns its columns from the rows it is fit on; configure it "
                        "to produce the same columns on every fold."
                    )
                blocks.append(fold_out)
                test_indices.append(test_idx)
        finally:
            _cross_fit_active.reset(token)
        return _stack_folds(blocks, test_indices, X_arr)

    def get_feature_names_out(self, input_features=None):
        """Delegate output feature names to the all-data ``estimator_``."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.get_feature_names_out(input_features)

    def get_representation_spec(self, input_features=None):
        """Return the wrapped spec, flagged as cross-fitted."""
        check_is_fitted(self, "estimator_")
        spec_fn = getattr(self.estimator_, "get_representation_spec", None)
        if spec_fn is not None:
            base = spec_fn(input_features)
            return replace(base, uses_target=True, cross_fitted=True, n_folds=int(self.n_folds))
        if input_features is None:
            # Name the inputs as the wrapped estimator does, so a DataFrame fit
            # passes its column names (not x0, x1, ...) to get_feature_names_out.
            input_features = getattr(self.estimator_, "feature_names_in_", None)
        return super().get_representation_spec(input_features)

    def _representation_cross_fitting(self):
        """Report cross-fitting metadata for the spec fallback path."""
        return True, int(self.n_folds)
