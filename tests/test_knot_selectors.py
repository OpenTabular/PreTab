import numpy as np
import pytest

from pretab.transformers.splines import (
    BaseKnotSelector,
    CARTKnotSelector,
    LightGBMKnotSelector,
)


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, size=(300, 1))
    y = np.sin(X[:, 0]) + 0.1 * rng.randn(300)
    return X, y


def test_selectors_subclass_base():
    assert issubclass(CARTKnotSelector, BaseKnotSelector)
    assert issubclass(LightGBMKnotSelector, BaseKnotSelector)


def test_basis_to_knots_conversion():
    sel = CARTKnotSelector(degree=3)
    assert sel._basis_to_knots(10) == 10 - 3 - 1
    assert sel._basis_to_knots(2) == 0


def test_cart_returns_sorted_knots_in_range(data):
    X, y = data
    sel = CARTKnotSelector(max_basis_functions=12, degree=3)
    knots = sel.get_knot_locations(X, y, task="regression")

    assert knots.ndim == 1
    assert np.all(np.diff(knots) > 0)  # sorted and unique
    assert knots.min() > X.min()
    assert knots.max() < X.max()


def test_cart_respects_max_knots(data):
    X, y = data
    sel = CARTKnotSelector(min_basis_functions=6, max_basis_functions=8, degree=3)
    knots = sel.get_knot_locations(X, y)
    assert len(knots) <= sel.max_knots


def test_cart_requires_y(data):
    X, _ = data
    with pytest.raises(ValueError, match="requires y"):
        CARTKnotSelector().get_knot_locations(X, None)


def test_cart_reproducible(data):
    X, y = data
    a = CARTKnotSelector().get_knot_locations(X, y)
    b = CARTKnotSelector().get_knot_locations(X, y)
    np.testing.assert_array_equal(a, b)


def test_cart_small_sample_quantile_fallback():
    rng = np.random.RandomState(1)
    X = rng.rand(5, 1)
    y = rng.rand(5)
    sel = CARTKnotSelector(min_samples_split=20, min_basis_functions=5, degree=1)
    knots = sel.get_knot_locations(X, y)
    assert len(knots) == sel.min_knots


def test_cart_classification_task():
    rng = np.random.RandomState(2)
    X = rng.rand(200, 1)
    y = (X[:, 0] > 0.5).astype(int)
    knots = CARTKnotSelector().get_knot_locations(X, y, task="classification")
    assert knots.ndim == 1


def test_cart_handles_nan_rows(data):
    X, y = data
    X_missing = X.copy()
    X_missing[:5, 0] = np.nan
    knots = CARTKnotSelector(max_basis_functions=12).get_knot_locations(X_missing, y)
    assert np.isfinite(knots).all()


def test_lightgbm_selector_runs(data):
    pytest.importorskip("lightgbm")
    X, y = data
    sel = LightGBMKnotSelector(n_estimators=30, max_basis_functions=12)
    knots = sel.get_knot_locations(X, y, task="regression")

    assert knots.ndim == 1
    assert np.all(np.diff(knots) > 0)
    assert knots.min() > X.min()
    assert knots.max() < X.max()


def test_lightgbm_selector_reproducible(data):
    pytest.importorskip("lightgbm")
    X, y = data
    a = LightGBMKnotSelector(n_estimators=30).get_knot_locations(X, y)
    b = LightGBMKnotSelector(n_estimators=30).get_knot_locations(X, y)
    np.testing.assert_array_equal(a, b)
