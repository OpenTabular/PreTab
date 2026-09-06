"""Tests for :class:`~pretab.compose.search.RepresentationSearchCV`.

The search picks the best ``numerical_method`` by cross-validation, then refits
the winning representation (and the downstream estimator) on all data. The data
here is a controlled nonlinear signal (``sin``) where an expressive basis
(``bspline``) must beat a linear ``standardization`` baseline.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

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
