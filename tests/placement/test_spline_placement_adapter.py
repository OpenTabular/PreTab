import numpy as np
import pytest

from pretab.exceptions import IncompatibleParamsError
from pretab.placement.adapters import SplinePlacementAdapter


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, size=(300, 1))
    y = np.sin(X[:, 0]) + 0.1 * rng.randn(300)
    return X, y


def test_basis_to_knots_conversion():
    adapter = SplinePlacementAdapter(
        placement_strategy="cart", degree=3, min_basis_functions=2, max_basis_functions=10
    )
    assert adapter.max_knots == 10 - 3 - 1
    assert adapter.min_knots == 0


def test_cart_returns_sorted_knots_in_range(data):
    X, y = data
    adapter = SplinePlacementAdapter(placement_strategy="cart", max_basis_functions=12, degree=3)
    knots = adapter.get_knot_locations(X, y, task="regression")

    assert knots.ndim == 1
    assert np.all(np.diff(knots) > 0)  # sorted and unique
    assert knots.min() > X.min()
    assert knots.max() < X.max()


def test_cart_respects_max_knots(data):
    X, y = data
    adapter = SplinePlacementAdapter(
        placement_strategy="cart", min_basis_functions=6, max_basis_functions=8, degree=3
    )
    knots = adapter.get_knot_locations(X, y)
    assert len(knots) <= adapter.max_knots


def test_cart_requires_y(data):
    X, _ = data
    with pytest.raises(IncompatibleParamsError, match="requires y"):
        SplinePlacementAdapter(placement_strategy="cart", degree=3).get_knot_locations(X, None)


def test_cart_reproducible(data):
    X, y = data
    a = SplinePlacementAdapter(placement_strategy="cart", degree=3).get_knot_locations(X, y)
    b = SplinePlacementAdapter(placement_strategy="cart", degree=3).get_knot_locations(X, y)
    np.testing.assert_array_equal(a, b)


def test_cart_small_sample_quantile_fallback():
    rng = np.random.RandomState(1)
    X = rng.rand(5, 1)
    y = rng.rand(5)
    adapter = SplinePlacementAdapter(placement_strategy="cart", min_basis_functions=5, degree=1)
    knots = adapter.get_knot_locations(X, y)
    assert len(knots) == adapter.min_knots


def test_cart_classification_task():
    rng = np.random.RandomState(2)
    X = rng.rand(200, 1)
    y = (X[:, 0] > 0.5).astype(int)
    knots = SplinePlacementAdapter(placement_strategy="cart", degree=3).get_knot_locations(
        X, y, task="classification"
    )
    assert knots.ndim == 1


def test_cart_handles_nan_rows(data):
    X, y = data
    X_missing = X.copy()
    X_missing[:5, 0] = np.nan
    adapter = SplinePlacementAdapter(placement_strategy="cart", max_basis_functions=12, degree=3)
    knots = adapter.get_knot_locations(X_missing, y)
    assert np.isfinite(knots).all()


def test_rejects_unsupervised_strategy():
    with pytest.raises(Exception):  # noqa: B017 - invalid_param_error -> InvalidParamError
        SplinePlacementAdapter(placement_strategy="quantile", degree=3)


def test_lightgbm_adapter_runs(data):
    pytest.importorskip("lightgbm")
    X, y = data
    adapter = SplinePlacementAdapter(placement_strategy="lightgbm", max_basis_functions=12, degree=3)
    knots = adapter.get_knot_locations(X, y, task="regression")

    assert knots.ndim == 1
    assert np.all(np.diff(knots) > 0)
    assert knots.min() > X.min()
    assert knots.max() < X.max()


def test_lightgbm_adapter_reproducible(data):
    pytest.importorskip("lightgbm")
    X, y = data
    a = SplinePlacementAdapter(placement_strategy="lightgbm", degree=3).get_knot_locations(X, y)
    b = SplinePlacementAdapter(placement_strategy="lightgbm", degree=3).get_knot_locations(X, y)
    np.testing.assert_array_equal(a, b)
