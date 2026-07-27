import numpy as np
import pytest

from pretab.core.exceptions import IncompatibleParamsError
from pretab.core.selectors import (
    BaseLocationSelector,
    CARTLocationSelector,
    LightGBMLocationSelector,
)
from pretab.transformers.splines.knot_selectors import (
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
    adapter = CARTKnotSelector(max_basis_functions=12, degree=3)
    from_adapter = adapter.get_knot_locations(X, y, task="regression")
    from_selector = CARTLocationSelector().select(
        X, y, task="regression", min_count=adapter.min_knots, max_count=adapter.max_knots
    )
    np.testing.assert_array_equal(from_adapter, from_selector)


def test_lightgbm_select_runs(data):
    pytest.importorskip("lightgbm")
    X, y = data
    locations = LightGBMLocationSelector(n_estimators=30).select(
        X, y, task="regression", min_count=2, max_count=10
    )

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
    adapter = LightGBMKnotSelector(n_estimators=30, max_basis_functions=12)
    from_adapter = adapter.get_knot_locations(X, y, task="regression")
    from_selector = LightGBMLocationSelector(n_estimators=30).select(
        X, y, task="regression", min_count=adapter.min_knots, max_count=adapter.max_knots
    )
    np.testing.assert_array_equal(from_adapter, from_selector)


# --------------------------------------------------------------------------- #
# Spacing must not depend on the order candidates arrive in.
#
# ``_ordered_candidates`` returns location order for a single tree but
# gain-descending order for a boosted ensemble, so an order-sensitive spacing
# filter discarded every candidate sitting below the previously kept one.
# --------------------------------------------------------------------------- #
def test_enforce_spacing_is_order_independent():
    x = np.linspace(0, 10, 500).reshape(-1, 1)
    selector = CARTLocationSelector(min_location_spacing=0.01)
    ascending = [1.0, 2.0, 3.0, 8.0, 9.0]

    from_ascending = selector._enforce_spacing(list(ascending), x)
    from_shuffled = selector._enforce_spacing([9.0, 1.0, 8.0, 3.0, 2.0], x)

    assert sorted(from_shuffled) == sorted(from_ascending) == ascending


def test_enforce_spacing_still_drops_close_neighbours():
    x = np.linspace(0, 10, 500).reshape(-1, 1)  # min_distance = 0.1
    selector = CARTLocationSelector(min_location_spacing=0.01)

    assert selector._enforce_spacing([1.0, 1.01, 5.0], x) == [1.0, 5.0]


def test_enforce_spacing_matches_legacy_result_on_ascending_input():
    # The single-tree path always supplies ascending candidates; comparing
    # against every kept location must be equivalent there.
    rng = np.random.RandomState(0)
    x = rng.uniform(0, 10, size=(500, 1))
    points = sorted(rng.uniform(0, 10, size=40).tolist())
    selector = CARTLocationSelector(min_location_spacing=0.05)

    min_distance = 0.05 * float(x.max() - x.min())
    legacy = [points[0]]
    for point in points[1:]:
        if point - legacy[-1] >= min_distance:
            legacy.append(point)

    assert selector._enforce_spacing(points, x) == legacy


def test_lightgbm_locations_cover_the_feature_range(data):
    pytest.importorskip("lightgbm")
    X, y = data
    x_min, x_max = float(X.min()), float(X.max())

    lgbm = LightGBMLocationSelector(n_estimators=30).select(
        X, y, task="regression", min_count=3, max_count=8
    )
    cart = CARTLocationSelector().select(X, y, task="regression", min_count=3, max_count=8)

    def span(locations):
        return (locations.max() - locations.min()) / (x_max - x_min)

    # Previously the gain-ordered candidates collapsed into a narrow cluster;
    # coverage should now be comparable to the single-tree selector.
    assert span(lgbm) > 0.5 * span(cart)


def test_lightgbm_keeps_the_highest_gain_locations(data):
    pytest.importorskip("lightgbm")
    X, y = data
    selector = LightGBMLocationSelector(n_estimators=30)

    candidates, _ = selector._ordered_candidates(X, y, "regression")
    spaced = selector._enforce_spacing(candidates, X)
    # Gain-descending order survives the spacing filter, so the trim keeps the
    # top-ranked entries rather than whichever happen to be numerically lowest.
    assert spaced == [c for c in candidates if c in set(spaced)]
    assert selector._trim_over_max(spaced, None, 5) == spaced[:5]
