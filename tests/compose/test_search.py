"""Tests for :class:`~pretab.compose.search.RepresentationSearchCV`.

The search picks the best ``numerical_method`` by cross-validation, then refits
the winning representation (and the downstream estimator) on all data. The data
here is a controlled nonlinear signal (``sin``) where an expressive basis
(``bspline``) must beat a linear ``standardization`` baseline.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    StratifiedKFold,
    check_cv,
    cross_val_score,
)
from sklearn.svm import SVC

from pretab import RepresentationSearchCV
from pretab.exceptions import InvalidParamError

# Unsupervised, deterministic placement so scores are reproducible fold to fold.
_UNSUPERVISED = {"target_aware": False, "placement_strategy": "uniform", "output_dim": 10}


@pytest.fixture
def nonlinear_data():
    """Random (unsorted) x in [-3, 3] with a smooth nonlinear target."""
    rng = np.random.RandomState(0)
    x = rng.uniform(-3.0, 3.0, size=200)
    X = pd.DataFrame({"x": x})
    y = np.sin(x) + 0.05 * rng.randn(200)
    return X, y


def _search(estimator, methods, **kwargs):
    params = {"cv": 4, "preprocessor_params": _UNSUPERVISED, "random_state": 0}
    params.update(kwargs)
    return RepresentationSearchCV(estimator, methods=methods, **params)


def test_selects_expressive_method_on_nonlinear_signal(nonlinear_data):
    X, y = nonlinear_data
    search = _search(LinearRegression(), ["standardization", "bspline"]).fit(X, y)

    assert set(search.cv_results_) == {"standardization", "bspline"}
    assert search.best_method_ == "bspline"
    assert search.cv_results_["bspline"] > search.cv_results_["standardization"]
    assert search.best_score_ == pytest.approx(max(search.cv_results_.values()))


def test_refit_best_representation_and_predict(nonlinear_data):
    X, y = nonlinear_data
    search = _search(LinearRegression(), ["standardization", "bspline"]).fit(X, y)

    assert search.best_method_ == "bspline"
    # best_preprocessor_ carries the winning method and is refit on all data.
    assert search.best_preprocessor_.numerical_method == "bspline"
    preds = search.predict(X)
    assert preds.shape == (len(X),)
    # A refit bspline fits the smooth signal well.
    assert search.score(X, y) > 0.9


def test_fit_is_reproducible(nonlinear_data):
    X, y = nonlinear_data
    first = _search(LinearRegression(), ["standardization", "bspline"]).fit(X, y)
    second = _search(LinearRegression(), ["standardization", "bspline"]).fit(X, y)

    assert first.best_method_ == second.best_method_
    assert first.cv_results_ == second.cv_results_
    np.testing.assert_allclose(first.predict(X), second.predict(X))


def test_accepts_cv_splitter_object(nonlinear_data):
    X, y = nonlinear_data
    search = _search(LinearRegression(), ["bspline"], cv=KFold(n_splits=3, shuffle=True, random_state=0)).fit(X, y)

    assert search.best_method_ == "bspline"
    assert set(search.cv_results_) == {"bspline"}


def test_classification_uses_stratified_cv():
    rng = np.random.RandomState(0)
    x = rng.uniform(-3.0, 3.0, size=200)
    X = pd.DataFrame({"x": x})
    y = (np.sin(x) > 0).astype(int)
    search = _search(LogisticRegression(max_iter=1000), ["standardization", "bspline"]).fit(X, y)

    assert search.best_method_ in {"standardization", "bspline"}
    assert 0.0 <= search.score(X, y) <= 1.0


def test_candidates_share_randomized_folds(nonlinear_data):
    X, y = nonlinear_data
    # These aliases resolve to the same method. With identical folds their
    # scores must agree even when the splitter has mutable RNG state.
    cv = KFold(n_splits=3, shuffle=True, random_state=np.random.RandomState(42))
    search = _search(LinearRegression(), ["standardization", "standard"], cv=cv).fit(X, y)
    assert search.cv_results_["standardization"] == search.cv_results_["standard"]


def test_empty_methods_raises(nonlinear_data):
    X, y = nonlinear_data
    with pytest.raises(InvalidParamError):
        RepresentationSearchCV(LinearRegression(), methods=[]).fit(X, y)


def test_requires_y_at_fit(nonlinear_data):
    X, _ = nonlinear_data
    with pytest.raises(InvalidParamError):
        RepresentationSearchCV(LinearRegression(), methods=["bspline"]).fit(X, None)


def test_predict_before_fit_raises(nonlinear_data):
    X, _ = nonlinear_data
    search = RepresentationSearchCV(LinearRegression(), methods=["bspline"])
    with pytest.raises(NotFittedError):
        search.predict(X)


def test_get_params_and_clone_preserve_config():
    from sklearn.base import clone

    search = RepresentationSearchCV(LinearRegression(), methods=["bspline", "standardization"], cv=3)
    assert search.get_params()["methods"] == ["bspline", "standardization"]
    cloned = clone(search)
    assert isinstance(cloned, RepresentationSearchCV)
    assert cloned.get_params()["cv"] == 3


@pytest.fixture
def class_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.uniform(-3, 3, 300)})
    return X, np.where(np.sin(X["x"]) > 0, "pos", "neg")


def test_classifier_search_places_against_class_labels(class_data):
    """A classifier made the search keep task="regression": string labels crashed the
    target-aware placement and integer labels were treated as a regression target."""
    X, y = class_data
    search = RepresentationSearchCV(LogisticRegression(), methods=["minmax", "ple"], cv=3, random_state=0).fit(X, y)
    assert search.best_preprocessor_.task == "classification"
    assert set(search.predict(X)) <= {"pos", "neg"}


def test_classifier_search_keeps_an_explicit_task(class_data):
    X, y = class_data
    search = RepresentationSearchCV(
        LogisticRegression(), methods=["minmax"], cv=3, preprocessor_params={"task": "regression"}
    ).fit(X, (y == "pos").astype(int))
    assert search.best_preprocessor_.task == "regression"


def test_regressor_search_keeps_the_regression_task(nonlinear_data):
    X, y = nonlinear_data
    search = RepresentationSearchCV(LinearRegression(), methods=["minmax"], cv=3).fit(X, y)
    assert search.best_preprocessor_.task == "regression"


class _RowRecorder(RegressorMixin, BaseEstimator):
    """Mean regressor whose ``fit`` records the ``row_id`` fit parameter it gets."""

    def fit(self, X, y, row_id=None):
        self.row_id_ = row_id
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def _recording_scorer(fold_rows):
    """Scorer that stores the ``row_id`` of each fold's fitted estimator."""

    def score(est, x_test, y_test):
        fold_rows.append(np.asarray(est.row_id_))
        return 0.0

    return score


@pytest.fixture
def grouped_data(nonlinear_data):
    """``nonlinear_data`` with 20 groups of 10 rows (e.g. 20 patients)."""
    X, y = nonlinear_data
    return X, y, np.repeat(np.arange(20), 10)


@pytest.mark.parametrize(
    "cv",
    [GroupKFold(n_splits=4), LeaveOneGroupOut(), GroupShuffleSplit(n_splits=3, test_size=0.25, random_state=0)],
    ids=["group_kfold", "leave_one_group_out", "group_shuffle_split"],
)
def test_group_splitter_receives_groups(grouped_data, cv):
    """Group splitters always failed: fit() took no groups and split() got none."""
    X, y, groups = grouped_data
    fold_rows = []
    search = _search(_RowRecorder(), ["minmax"], cv=cv, scoring=_recording_scorer(fold_rows))
    search.fit(X, y, groups=groups, row_id=np.arange(len(X)))

    for rows, (train, test) in zip(fold_rows, cv.split(X, y, groups), strict=True):
        np.testing.assert_array_equal(rows, train)
        # No group is on both sides of a train/validation split.
        assert set(groups[rows]).isdisjoint(groups[test])


@pytest.mark.filterwarnings("ignore:The groups parameter is ignored:UserWarning")
def test_groups_leave_non_group_splits_unchanged(grouped_data):
    X, y, groups = grouped_data
    plain = _search(LinearRegression(), ["standardization", "bspline"]).fit(X, y)
    grouped = _search(LinearRegression(), ["standardization", "bspline"]).fit(X, y, groups=groups)
    assert grouped.cv_results_ == plain.cv_results_


def test_fit_params_are_split_per_fold_and_passed_to_the_refit(nonlinear_data):
    X, y = nonlinear_data
    row_id = np.arange(len(X))
    cv = KFold(n_splits=4, shuffle=True, random_state=0)
    fold_rows = []
    search = _search(_RowRecorder(), ["minmax"], cv=cv, scoring=_recording_scorer(fold_rows))
    search.fit(X, y, row_id=row_id)

    # Each fold's estimator sees only its training rows; the refit sees all rows.
    for rows, (train, _) in zip(fold_rows, cv.split(X), strict=True):
        np.testing.assert_array_equal(rows, train)
    np.testing.assert_array_equal(search.best_estimator_.row_id_, row_id)


def test_sample_weight_reaches_the_estimator(nonlinear_data):
    X, y = nonlinear_data
    weights = np.linspace(0.1, 2.0, len(X))
    search = _search(Ridge(), ["bspline"]).fit(X, y, sample_weight=weights)

    x_all = search.best_preprocessor_.transform(X, return_array=True)
    expected = Ridge().fit(x_all, y, sample_weight=weights)
    np.testing.assert_allclose(search.best_estimator_.coef_, expected.coef_)
    # The fold fits are weighted as well, so the validation scores change.
    assert search.cv_results_ != _search(Ridge(), ["bspline"]).fit(X, y).cv_results_


_DELEGATED = ("classes_", "predict_proba", "predict_log_proba", "decision_function")


def test_search_takes_the_estimator_type(class_data):
    """The search was neither a classifier nor a regressor, so nested cross-validation
    of a classifier search used KFold instead of StratifiedKFold."""
    _, y = class_data
    classifier_search = RepresentationSearchCV(LogisticRegression(), methods=["minmax"])
    regressor_search = RepresentationSearchCV(Ridge(), methods=["minmax"])

    assert is_classifier(classifier_search) and not is_regressor(classifier_search)
    assert is_regressor(regressor_search) and not is_classifier(regressor_search)
    assert isinstance(check_cv(3, y, classifier=is_classifier(classifier_search)), StratifiedKFold)


def test_classifier_search_delegates_to_the_best_estimator(class_data):
    X, y = class_data
    search = RepresentationSearchCV(LogisticRegression(), methods=["minmax", "ple"], cv=3, random_state=0).fit(X, y)
    x = search.best_preprocessor_.transform(X, return_array=True)

    np.testing.assert_array_equal(search.classes_, ["neg", "pos"])
    np.testing.assert_allclose(search.predict_proba(X), search.best_estimator_.predict_proba(x))
    np.testing.assert_allclose(search.predict_log_proba(X), search.best_estimator_.predict_log_proba(x))
    np.testing.assert_allclose(search.decision_function(X), search.best_estimator_.decision_function(x))


def test_delegated_methods_follow_the_wrapped_estimator(class_data, nonlinear_data):
    X, y = class_data
    # SVC without probability=True has a decision_function but no predict_proba.
    svc_search = RepresentationSearchCV(SVC(), methods=["minmax"], cv=3)
    assert hasattr(svc_search, "decision_function") and not hasattr(svc_search, "predict_proba")
    svc_search.fit(X, y)
    assert hasattr(svc_search, "decision_function") and not hasattr(svc_search, "predict_proba")

    X_reg, y_reg = nonlinear_data
    regressor_search = _search(Ridge(), ["minmax"]).fit(X_reg, y_reg)
    assert not any(hasattr(regressor_search, attr) for attr in _DELEGATED)


def test_unfitted_classifier_search_raises_not_fitted(class_data):
    X, _ = class_data
    search = RepresentationSearchCV(LogisticRegression(), methods=["minmax"])
    assert not hasattr(search, "classes_")
    with pytest.raises(NotFittedError):
        search.predict_proba(X)


def test_nested_cross_val_score_with_roc_auc(class_data):
    """roc_auc needs classes_ and predict_proba / decision_function: every outer fold
    scored nan."""
    X, y = class_data
    search = RepresentationSearchCV(LogisticRegression(), methods=["minmax", "ple"], cv=3, random_state=0)
    scores = cross_val_score(search, X, y, scoring="roc_auc", cv=3)
    assert np.all(np.isfinite(scores))
    assert scores.min() > 0.9


def test_search_records_the_input_features(nonlinear_data):
    X, y = nonlinear_data
    search = _search(LinearRegression(), ["bspline"]).fit(X, y)
    assert search.n_features_in_ == 1
    np.testing.assert_array_equal(search.feature_names_in_, ["x"])

    # Refitting on an array drops the stale names.
    search.fit(X.to_numpy(), y)
    assert search.n_features_in_ == 1
    assert not hasattr(search, "feature_names_in_")
