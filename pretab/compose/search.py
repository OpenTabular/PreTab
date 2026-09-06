"""Cross-validated search over numerical representation methods.

``RepresentationSearchCV`` is a lightweight skeleton that, for each candidate
numerical method, builds a :class:`~pretab.preprocessor.Preprocessor` feeding a
cloned downstream ``estimator``, scores it with cross-validation, and refits the
best-scoring representation on all data. It is intentionally minimal: the search
space is the ``numerical_method`` axis, and every fold uses the preprocessor's
native array output so no target leaks across the train/validation split.
"""

from collections.abc import Callable
from typing import cast

import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import BaseCrossValidator, check_cv
from sklearn.utils.validation import check_is_fitted

from ..core._typing import PredictorLike
from ..exceptions import InvalidParamError
from ..preprocessor import Preprocessor

__all__ = ["RepresentationSearchCV"]


def _row_subset(data, idx):
    """Return the rows of ``data`` at positions ``idx`` for arrays or frames."""
    if hasattr(data, "iloc"):
        return data.iloc[idx]
    return np.asarray(data)[idx]


class RepresentationSearchCV(BaseEstimator):
    """Select the best numerical representation by cross-validation.

    Parameters
    ----------
    estimator : estimator
        Downstream supervised estimator fit on the transformed features.
    methods : sequence of str
        Candidate ``numerical_method`` values to search over.
    cv : int or cross-validation generator, default=5
        Cross-validation splitting strategy passed to
        :func:`sklearn.model_selection.check_cv`.
    scoring : str or callable or None, default=None
        Scoring passed to :func:`sklearn.metrics.check_scoring`; ``None`` uses the
        estimator's ``score`` method.
    preprocessor_params : dict or None, default=None
        Extra keyword arguments forwarded to every :class:`Preprocessor`.
    random_state : int or None, default=None
        Seed forwarded to each :class:`Preprocessor`.

    Attributes
    ----------
    cv_results_ : dict
        Mapping of method name to mean cross-validation score.
    best_method_ : str
        The highest-scoring numerical method.
    best_score_ : float
        Mean cross-validation score of ``best_method_``.
    best_preprocessor_ : Preprocessor
        Preprocessor for ``best_method_`` refit on all data.
    best_estimator_ : estimator
        Estimator refit on the best representation of all data.
    """

    def __init__(self, estimator, methods, *, cv=5, scoring=None, preprocessor_params=None, random_state=None):
        self.estimator = estimator
        self.methods = methods
        self.cv = cv
        self.scoring = scoring
        self.preprocessor_params = preprocessor_params
        self.random_state = random_state

    def _make_preprocessor(self, method):
        """Build a Preprocessor for ``method`` with the shared parameters."""
        params = dict(self.preprocessor_params or {})
        params.setdefault("random_state", self.random_state)
        return Preprocessor(numerical_method=method, **params)

    def fit(self, X, y=None):
        """Search over ``methods`` and refit the best representation on all data."""
        methods = list(self.methods)
        if not methods:
            raise InvalidParamError("methods must be a non-empty sequence of numerical methods.")
        if y is None:
            raise InvalidParamError("RepresentationSearchCV requires y at fit time; got y=None.")
        y_arr = np.asarray(y).ravel()
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        cv = cast(BaseCrossValidator, check_cv(self.cv, y_arr, classifier=is_classifier(self.estimator)))
        # Reuse the same held-out rows for every candidate, including splitters
        # whose random state advances on each call to split().
        splits = list(cv.split(np.zeros(n_samples), y_arr))

        cv_results: dict[str, float] = {}
        best_score = -np.inf
        best_method = methods[0]
        for method in methods:
            fold_scores = []
            for train_idx, test_idx in splits:
                pre = self._make_preprocessor(method)
                est = cast(PredictorLike, clone(self.estimator))
                x_train = pre.fit_transform(_row_subset(X, train_idx), y_arr[train_idx], return_array=True)
                est.fit(x_train, y_arr[train_idx])
                scorer = cast("Callable[..., float]", check_scoring(est, scoring=self.scoring))
                x_test = pre.transform(_row_subset(X, test_idx), return_array=True)
                fold_scores.append(scorer(est, x_test, y_arr[test_idx]))
            mean_score = float(np.mean(fold_scores))
            cv_results[method] = mean_score
            if mean_score > best_score:
                best_score = mean_score
                best_method = method

        self.cv_results_ = cv_results
        self.best_method_ = best_method
        self.best_score_ = best_score
        self.best_preprocessor_ = self._make_preprocessor(best_method)
        x_all = self.best_preprocessor_.fit_transform(X, y_arr, return_array=True)
        self.best_estimator_ = cast(PredictorLike, clone(self.estimator)).fit(x_all, y_arr)
        return self

    def predict(self, X):
        """Predict with the best refit estimator on the best representation."""
        check_is_fitted(self, "best_estimator_")
        x = self.best_preprocessor_.transform(X, return_array=True)
        return self.best_estimator_.predict(x)

    def score(self, X, y):
        """Score the best refit estimator on ``(X, y)``."""
        check_is_fitted(self, "best_estimator_")
        x = self.best_preprocessor_.transform(X, return_array=True)
        scorer = cast("Callable[..., float]", check_scoring(self.best_estimator_, scoring=self.scoring))
        return scorer(self.best_estimator_, x, np.asarray(y).ravel())
