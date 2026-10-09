import numpy as np
import pytest

from pretab.core.selectors import (
    BaseLocationSelector,
    CARTLocationSelector,
    LightGBMLocationSelector,
)
from pretab.exceptions import IncompatibleParamsError
from pretab.placement.adapters import SplinePlacementAdapter


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, size=(300, 1))
    y = np.sin(X[:, 0]) + 0.1 * rng.randn(300)
    return X, y


def test_selectors_subclass_base():
    assert issubclass(CARTLocationSelector, BaseLocationSelector)
    assert issubclass(LightGBMLocationSelector, BaseLocationSelector)


def test_cart_select_sorted_in_range(data):
    X, y = data
    locations = CARTLocationSelector().select(X, y, task="regression", min_count=2, max_count=10)

    assert locations.ndim == 1
    assert np.all(np.diff(locations) > 0)  # sorted and unique
    assert locations.min() > X.min()
    assert locations.max() < X.max()


def test_cart_respects_max_count(data):
    X, y = data
    locations = CARTLocationSelector().select(X, y, task="regression", min_count=2, max_count=5)
    assert len(locations) <= 5


def test_cart_requires_y(data):
    X, _ = data
    with pytest.raises(IncompatibleParamsError, match="requires y"):
        CARTLocationSelector().select(X, None, min_count=2, max_count=5)


def test_cart_reproducible(data):
    X, y = data
    a = CARTLocationSelector().select(X, y, min_count=3, max_count=10)
    b = CARTLocationSelector().select(X, y, min_count=3, max_count=10)
    np.testing.assert_array_equal(a, b)


def test_cart_small_sample_quantile_fallback():
    rng = np.random.RandomState(1)
    X = rng.rand(5, 1)
    y = rng.rand(5)
    locations = CARTLocationSelector(min_samples_split=20).select(X, y, min_count=5, max_count=8)
    assert len(locations) == 5


def test_cart_classification_task():
    rng = np.random.RandomState(2)
    X = rng.rand(200, 1)
    y = (X[:, 0] > 0.5).astype(int)
    locations = CARTLocationSelector().select(X, y, task="classification", min_count=2, max_count=10)
    assert locations.ndim == 1


def test_cart_handles_nan_rows(data):
    X, y = data
    X_missing = X.copy()
    X_missing[:5, 0] = np.nan
    locations = CARTLocationSelector().select(X_missing, y, min_count=2, max_count=12)
    assert np.isfinite(locations).all()


def test_cart_matches_knot_adapter(data):
    X, y = data
    adapter = SplinePlacementAdapter(placement_strategy="cart", max_basis_functions=12, degree=3)
    from_adapter = adapter.get_knot_locations(X, y, task="regression")
    from_selector = CARTLocationSelector().select(
        X, y, task="regression", min_count=adapter.min_knots, max_count=adapter.max_knots
    )
    np.testing.assert_array_equal(from_adapter, from_selector)


def test_lightgbm_select_runs(data):
    pytest.importorskip("lightgbm")
    X, y = data
    locations = LightGBMLocationSelector(n_estimators=30).select(X, y, task="regression", min_count=2, max_count=10)

    assert locations.ndim == 1
    assert np.all(np.diff(locations) > 0)
    assert locations.min() > X.min()
    assert locations.max() < X.max()


def test_lightgbm_reproducible(data):
    pytest.importorskip("lightgbm")
    X, y = data
    a = LightGBMLocationSelector(n_estimators=30).select(X, y, min_count=3, max_count=10)
    b = LightGBMLocationSelector(n_estimators=30).select(X, y, min_count=3, max_count=10)
    np.testing.assert_array_equal(a, b)


def test_lightgbm_matches_knot_adapter(data):
    pytest.importorskip("lightgbm")
    X, y = data
    adapter = SplinePlacementAdapter(placement_strategy="lightgbm", max_basis_functions=12, degree=3)
    from_adapter = adapter.get_knot_locations(X, y, task="regression")
    from_selector = LightGBMLocationSelector().select(
        X, y, task="regression", min_count=adapter.min_knots, max_count=adapter.max_knots
    )
    np.testing.assert_array_equal(from_adapter, from_selector)


def test_lightgbm_locations_span_feature_range():
    """Regression guard for issue #10: lightgbm placement must not cluster into one subrange."""
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, size=2000)
    y = np.sin(x) + 0.05 * rng.normal(size=2000)
    X = x.reshape(-1, 1)

    lgb_locs = LightGBMLocationSelector().select(X, y, task="regression", min_count=3, max_count=8)
    cart_locs = CARTLocationSelector().select(X, y, task="regression", min_count=3, max_count=8)

    # lightgbm locations must span at least half the range that CART covers
    lgb_span = float(lgb_locs[-1] - lgb_locs[0])
    cart_span = float(cart_locs[-1] - cart_locs[0])
    assert lgb_span >= 0.5 * cart_span, (
        f"lightgbm span {lgb_span:.2f} is less than half of CART span {cart_span:.2f}; "
        "locations are clustering into a subrange (issue #10)"
    )


def test_supplement_keeps_existing_locations():
    """Regression guard for issue #18: supplementing must not drop selector-found locations."""
    sel = CARTLocationSelector()
    x = np.linspace(0, 10, 500).reshape(-1, 1)

    result = sel._supplement([9.5, 9.7], x, 5)

    assert 9.5 in result
    assert 9.7 in result
    assert len(result) == 5


def test_supplement_preserves_high_end_split_end_to_end():
    """Regression guard for issue #18: an end-to-end topped-up selection must keep the

    highest tree-found split instead of discarding it in favour of low quantiles.
    """
    rng = np.random.default_rng(0)
    xs = rng.choice([0.0, 0.05, 9.5, 9.8], size=300)
    ys = (xs > 5).astype(float) + 0.01 * rng.normal(size=300)

    locations = CARTLocationSelector().select(xs.reshape(-1, 1), ys, task="regression", min_count=6, max_count=6)

    assert locations.max() > 5.0, f"expected a high-end location to survive supplementing, got {locations}"


def _single_step(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, 1000)
    step = rng.uniform(2, 8)
    y = np.where(x > step, 1.0, 0.0) + 0.5 * rng.normal(size=1000)
    return x.reshape(-1, 1), y


def test_cart_spacing_keeps_the_dominant_split_over_a_weak_neighbour():
    """Regression guard for issue #70: candidates were spaced in ascending location
    order, so a weak split just below the root split evicted it."""
    from sklearn.tree import DecisionTreeRegressor

    X, y = _single_step(34)
    root = DecisionTreeRegressor(max_depth=1).fit(X, y).tree_.threshold[0]
    locations = CARTLocationSelector().select(X, y, task="regression", min_count=6, max_count=6)
    assert np.isclose(locations, root).any(), f"root split {root:.3f} missing from {locations}"


@pytest.mark.parametrize("seed", range(10))
def test_cart_keeps_the_root_split_of_a_single_step_target(seed):
    from sklearn.tree import DecisionTreeRegressor

    X, y = _single_step(seed)
    root = DecisionTreeRegressor(max_depth=1).fit(X, y).tree_.threshold[0]
    locations = CARTLocationSelector().select(X, y, task="regression", min_count=6, max_count=6)
    assert np.isclose(locations, root).any()


def test_cart_candidates_are_ordered_by_impurity_decrease(data):
    X, y = data
    selector = CARTLocationSelector()
    candidates, importance = selector._ordered_candidates(X, y, "regression")
    gains = [importance[c] for c in candidates]
    assert gains == sorted(gains, reverse=True)


# --- tied / discrete features (issue #57) ------------------------------------


def _discrete(levels, n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, levels, size=n).astype(float)
    return x.reshape(-1, 1), (x >= 2) + rng.normal(0, 0.1, size=n)


def _assert_distinct_interior(locations, X, count):
    assert len(locations) == count
    assert len(np.unique(locations)) == count
    assert locations.min() > X.min() and locations.max() < X.max()


@pytest.mark.parametrize("selector_cls", [CARTLocationSelector, LightGBMLocationSelector])
@pytest.mark.parametrize("levels", [2, 3, 5])
def test_selector_tops_up_tied_features_with_distinct_interior_locations(selector_cls, levels):
    """Regression guard for issue #57: the quantile top-up collapsed onto the tied
    values and the range boundary, returning too few / duplicate locations."""
    if selector_cls is LightGBMLocationSelector:
        pytest.importorskip("lightgbm")
    X, y = _discrete(levels)
    locations = selector_cls().select(X, y, task="regression", min_count=8, max_count=8)
    _assert_distinct_interior(locations, X, 8)


def test_supplement_on_tied_data_respects_spacing_and_count():
    x = np.repeat([0.0, 1.0, 2.0], 50)
    selector = CARTLocationSelector()
    result = selector._supplement([0.5], x, 6)
    assert len(result) == 6
    assert 0.5 in result
    assert min(np.diff(result)) >= selector.min_location_spacing * 2.0
    assert min(result) > 0.0 and max(result) < 2.0


def test_supplement_fills_dense_requests_beyond_the_spacing():
    x = np.linspace(0.0, 1.0, 1000)
    result = CARTLocationSelector()._supplement([], x, 150)
    assert len(result) == 150
    assert len(np.unique(result)) == 150


def test_small_sample_fallback_on_tied_data_is_distinct_and_interior():
    X = np.array([[0.0], [0.0], [0.0], [1.0], [1.0]])
    locations = CARTLocationSelector().select(X, np.arange(5.0), min_count=3, max_count=3)
    _assert_distinct_interior(locations, X, 3)


def test_constant_feature_fallback_keeps_the_requested_count():
    X = np.full((40, 1), 3.0)
    locations = CARTLocationSelector().select(X, np.arange(40.0), min_count=4, max_count=4)
    np.testing.assert_array_equal(locations, np.full(4, 3.0))


def test_lightgbm_zero_split_sentinel_maps_to_the_bin_midpoint():
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, size=300).astype(float)
    y = (x >= 1) + rng.normal(0, 0.1, size=300)
    locations = LightGBMLocationSelector().select(x.reshape(-1, 1), y, min_count=4, max_count=4)
    # The 0-vs-positive split is reported by LightGBM as threshold 1e-35.
    assert 0.5 in locations
    assert (np.abs(locations) > 1e-30).all()
