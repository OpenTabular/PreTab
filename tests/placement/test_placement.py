"""Contract tests for the :mod:`pretab.placement` subsystem (Phase 2, P2.8).

These lock the placement strategy contract independently of any transformer:
sorted, in-range, dedup-free locations; the requested-vs-effective unit counts;
reproducibility; target-required behaviour for the supervised strategies; both
classification and regression; the factory's combo validation; and the fixed
resolution policy.
"""

import numpy as np
import pytest

from pretab.core.knots import quantile_knots, spanning_knots, uniform_knots
from pretab.exceptions import IncompatibleParamsError, InvalidParamError
from pretab.placement import (
    BasePlacementStrategy,
    CARTPlacement,
    FixedResolution,
    PlacementResult,
    QuantilePlacement,
    UniformPlacement,
    create_placement_strategy,
)
from pretab.placement.adapters import RBFPlacementAdapter, SplinePlacementAdapter


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    x = rng.uniform(-3, 3, size=300)
    y = np.sin(x) + 0.1 * rng.randn(300)
    return x, y


@pytest.fixture
def clf_data():
    rng = np.random.RandomState(1)
    x = rng.uniform(-3, 3, size=300)
    y = (x > 0).astype(int)
    return x, y


# --------------------------------------------------------------------------- #
# Unsupervised strategies
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [UniformPlacement, QuantilePlacement])
def test_unsupervised_sorted_and_counted(cls, data):
    x, _ = data
    result = cls(6).place(x)
    assert isinstance(result, PlacementResult)
    assert result.locations.ndim == 1
    assert np.all(np.diff(result.locations) > 0)  # sorted, no duplicates
    assert result.requested_units == 6
    assert result.effective_units == len(result.locations) == 6
    assert result.target_aware is False


@pytest.mark.parametrize("cls", [UniformPlacement, QuantilePlacement])
def test_unsupervised_interior_in_range(cls, data):
    x, _ = data
    locs = cls(6).place(x).locations
    assert locs.min() > x.min()
    assert locs.max() < x.max()


def test_unsupervised_matches_primitives(data):
    x, _ = data
    assert np.allclose(UniformPlacement(6).place(x).locations, uniform_knots(x, 6))
    assert np.allclose(QuantilePlacement(6).place(x).locations, quantile_knots(x, 6))
    assert np.allclose(
        UniformPlacement(6, include_endpoints=True).place(x).locations,
        spanning_knots(x, 6, "uniform"),
    )
    assert np.allclose(
        QuantilePlacement(6, include_endpoints=True).place(x).locations,
        spanning_knots(x, 6, "quantile"),
    )


def test_unsupervised_endpoints_span_range(data):
    x, _ = data
    locs = UniformPlacement(6, include_endpoints=True).place(x).locations
    assert locs[0] == pytest.approx(x.min())
    assert locs[-1] == pytest.approx(x.max())


def test_unsupervised_ignores_nan(data):
    x, _ = data
    x = x.copy()
    x[:10] = np.nan
    locs = UniformPlacement(6).place(x).locations
    assert np.all(np.isfinite(locs))


# --------------------------------------------------------------------------- #
# Supervised strategies
# --------------------------------------------------------------------------- #
def test_cart_sorted_in_range_and_counts(data):
    x, y = data
    result = CARTPlacement(min_count=2, max_count=5, task="regression").place(x, y)
    assert np.all(np.diff(result.locations) > 0)
    assert result.locations.min() > x.min()
    assert result.locations.max() < x.max()
    assert result.requested_units == 5
    assert result.effective_units == len(result.locations) <= 5
    assert result.target_aware is True


def test_cart_requires_y(data):
    x, _ = data
    with pytest.raises(IncompatibleParamsError, match="requires y"):
        CARTPlacement(min_count=2, max_count=5).place(x, None)


def test_cart_reproducible(data):
    x, y = data
    a = CARTPlacement(min_count=3, max_count=10, random_state=51).place(x, y).locations
    b = CARTPlacement(min_count=3, max_count=10, random_state=51).place(x, y).locations
    np.testing.assert_array_equal(a, b)


def test_cart_classification(clf_data):
    x, y = clf_data
    locs = CARTPlacement(min_count=1, max_count=5, task="classification").place(x, y).locations
    assert np.all(np.diff(locs) > 0)
    assert locs.min() > x.min()
    assert locs.max() < x.max()


# --------------------------------------------------------------------------- #
# Factory + combo validation (D4)
# --------------------------------------------------------------------------- #
def test_factory_builds_each_strategy():
    assert isinstance(
        create_placement_strategy(target_aware=True, placement_strategy="cart", min_count=1, max_count=5),
        CARTPlacement,
    )
    assert isinstance(
        create_placement_strategy(target_aware=False, placement_strategy="uniform", min_count=6, max_count=6),
        UniformPlacement,
    )
    assert isinstance(
        create_placement_strategy(target_aware=False, placement_strategy="quantile", min_count=6, max_count=6),
        QuantilePlacement,
    )


@pytest.mark.parametrize(
    ("target_aware", "strategy"),
    [(True, "uniform"), (True, "quantile"), (False, "cart"), (False, "lightgbm")],
)
def test_factory_rejects_invalid_combo(target_aware, strategy):
    with pytest.raises(InvalidParamError):
        create_placement_strategy(target_aware=target_aware, placement_strategy=strategy, min_count=1, max_count=5)


def test_strategies_are_base_instances():
    strat = create_placement_strategy(target_aware=True, placement_strategy="cart", min_count=1, max_count=5)
    assert isinstance(strat, BasePlacementStrategy)


# --------------------------------------------------------------------------- #
# Resolution policy
# --------------------------------------------------------------------------- #
def test_fixed_resolution_non_adaptive_pins_output_dim():
    assert FixedResolution(adaptive=False).resolve(6, None, None, floor=1) == (6, 6)


def test_fixed_resolution_adaptive_window():
    assert FixedResolution(adaptive=True).resolve(6, 2, 10, floor=1) == (2, 10)


def test_fixed_resolution_rejects_below_floor():
    with pytest.raises(InvalidParamError):
        FixedResolution(adaptive=True).resolve(6, 0, 10, floor=1)


def test_fixed_resolution_non_adaptive_conflict():
    with pytest.raises(IncompatibleParamsError):
        FixedResolution(adaptive=False).resolve(3, 5, None, floor=1)


def test_not_yet_implemented_resolution_stubs_are_not_public():
    # Regression guard: CardinalityAwareResolution / DataSizeAwareResolution
    # always raise NotImplementedError, so they must not be part of the public
    # pretab.placement API (no user-facing "usable" class should always fail).
    import pretab.placement as placement_pkg

    assert "CardinalityAwareResolution" not in placement_pkg.__all__
    assert "DataSizeAwareResolution" not in placement_pkg.__all__
    assert not hasattr(placement_pkg, "CardinalityAwareResolution")
    assert not hasattr(placement_pkg, "DataSizeAwareResolution")


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
def test_spline_adapter_returns_interior_knots(data):
    x, y = data
    adapter = SplinePlacementAdapter(degree=3, placement_strategy="cart")
    knots = adapter.get_knot_locations(x.reshape(-1, 1), y, task="regression")
    assert np.all(np.diff(knots) > 0)
    assert knots.min() > x.min()
    assert knots.max() < x.max()


def test_spline_adapter_rejects_unsupervised_strategy():
    with pytest.raises(InvalidParamError):
        SplinePlacementAdapter(degree=3, placement_strategy="uniform")


def test_rbf_adapter_unsupervised_matches_inline(data):
    x, _ = data
    centers = RBFPlacementAdapter(target_aware=False, placement_strategy="quantile").get_centers(x, None, 6, 6)
    assert np.allclose(centers, np.percentile(x, np.linspace(0, 100, 6)))
    centers_u = RBFPlacementAdapter(target_aware=False, placement_strategy="uniform").get_centers(x, None, 6, 6)
    assert np.allclose(centers_u, np.linspace(x.min(), x.max(), 6))


# --- quantile placement on tied data ------------------------------------------------


def _tied_features():
    rng = np.random.default_rng(0)
    return {
        "zero-inflated": np.where(rng.random(1000) < 0.6, 0.0, rng.exponential(5.0, 1000)),
        "discrete": rng.integers(0, 5, 1000).astype(float),
        "top-coded": np.minimum(rng.uniform(0, 160, 1000), 100.0),
    }


@pytest.mark.parametrize("name", ["zero-inflated", "discrete", "top-coded"])
@pytest.mark.parametrize("include_endpoints", [True, False])
def test_quantile_placement_returns_distinct_locations_on_tied_data(name, include_endpoints):
    """Coinciding quantiles repeated feature-map centers, duplicating output columns."""
    x = _tied_features()[name]
    locations = QuantilePlacement(8, include_endpoints=include_endpoints).fit(x).locations_
    assert len(locations) == 8
    assert len(np.unique(locations)) == 8
    if include_endpoints:
        assert locations[0] == x.min() and locations[-1] == x.max()
    else:
        assert locations.min() > x.min() and locations.max() < x.max()


def test_quantile_placement_is_unchanged_on_untied_data():
    x = np.random.default_rng(1).normal(size=500)
    np.testing.assert_array_equal(
        QuantilePlacement(7, include_endpoints=True).fit(x).locations_, np.quantile(x, np.linspace(0, 1, 7))
    )


def test_quantile_feature_map_has_no_duplicate_columns_on_tied_data():
    from pretab.transformers import RBFExpansionTransformer

    X = _tied_features()["zero-inflated"].reshape(-1, 1)
    out = RBFExpansionTransformer(output_dim=8, placement_strategy="quantile").fit(X).transform(X)
    assert len(np.unique(out.round(10), axis=1).T) == 8
