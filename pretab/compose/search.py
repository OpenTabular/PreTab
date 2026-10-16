"""Cross-validated search over numerical representation methods.

``RepresentationSearchCV`` is a lightweight skeleton that, for each candidate
numerical method, builds a :class:`~pretab.preprocessor.Preprocessor` feeding a
cloned downstream ``estimator``, scores it with cross-validation, and refits the
best-scoring representation on all data. It is intentionally minimal: the search
space is the ``numerical_method`` axis, and every fold uses the preprocessor's
native array output so no target leaks across the train/validation split.
"""

from collections.abc import Callable
from copy import deepcopy
from typing import cast

import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import BaseCrossValidator, check_cv
from sklearn.utils import get_tags
from sklearn.utils.metaestimators import available_if
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


def _estimator_has(attr):
    """Check whether ``attr`` can be delegated to the downstream estimator.

    As in scikit-learn's search estimators, the refit ``best_estimator_`` is
    checked after fit and the unfitted ``estimator`` before, so ``hasattr`` on an
    unfitted search already reflects the estimator it wraps.
    """

    def check(self):
        # getattr raises AttributeError when the estimator does not provide attr.
        getattr(self.best_estimator_ if hasattr(self, "best_estimator_") else self.estimator, attr)
        return True

    return check


class RepresentationSearchCV(BaseEstimator):
    """Select the best numerical representation by cross-validation.

    The search takes the estimator type of ``estimator``: wrapping a classifier
    makes it a classifier (``is_classifier``, stratified outer cross-validation),
    and ``predict_proba``, ``predict_log_proba`` and ``decision_function`` are
    available whenever ``estimator`` provides them.

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
    classes_ : ndarray of shape (n_classes,)
        Class labels of ``best_estimator_``. Only available for a classifier.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of the features seen during :meth:`fit`. Only defined when ``X`` has
        string column names.
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
        self.n_features_in_ = self.best_preprocessor_.n_features_in_
        # scikit-learn convention: feature names are recorded only for input whose
        # column labels are all strings.
        columns = getattr(X, "columns", None)
        if columns is not None and all(isinstance(label, str) for label in columns):
            self.feature_names_in_ = np.asarray(columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        return self

    @property
    def classes_(self):
        """Class labels of the refit ``best_estimator_``, for a classifier."""
        return self.best_estimator_.classes_

    def _best_representation(self, X):
        """Transform ``X`` with the refit ``best_preprocessor_``."""
        check_is_fitted(self, "best_estimator_")
        return self.best_preprocessor_.transform(X, return_array=True)

    def predict(self, X):
        """Predict with the best refit estimator on the best representation."""
        x = self._best_representation(X)
        return self.best_estimator_.predict(x)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Predict class probabilities with the best refit estimator.

        Only available when ``estimator`` implements ``predict_proba``.
        """
        x = self._best_representation(X)
        return self.best_estimator_.predict_proba(x)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Predict class log-probabilities with the best refit estimator.

        Only available when ``estimator`` implements ``predict_log_proba``.
        """
        x = self._best_representation(X)
        return self.best_estimator_.predict_log_proba(x)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Compute the decision function of the best refit estimator.

        Only available when ``estimator`` implements ``decision_function``.
        """
        x = self._best_representation(X)
        return self.best_estimator_.decision_function(x)

    def score(self, X, y):
        """Score the best refit estimator on ``(X, y)``."""
        x = self._best_representation(X)
        scorer = cast("Callable[..., float]", check_scoring(self.best_estimator_, scoring=self.scoring))
        return scorer(self.best_estimator_, x, np.asarray(y).ravel())

    def __sklearn_tags__(self):
        """Take the estimator type and classifier / regressor tags of ``estimator``."""
        tags = super().__sklearn_tags__()  # type: ignore[attr-defined]
        estimator_tags = get_tags(self.estimator)
        tags.estimator_type = estimator_tags.estimator_type
        tags.classifier_tags = deepcopy(estimator_tags.classifier_tags)
        tags.regressor_tags = deepcopy(estimator_tags.regressor_tags)
        return tags
