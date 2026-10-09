"""Knot placement on tied data (top-coded, zero-inflated, low-cardinality features).

Regression guards for issue #56: quantile knots on tied data coincide with each
other and with the range boundary, which collapsed basis functions to a point
(dead columns) and, for B/M-splines, left every row at ``x_max`` all zero.
"""

import numpy as np
import pytest

from pretab.core.knots import supplement_interior_knots
from pretab.transformers import (
    BSplineTransformer,
    CubicRegressionSplineTransformer,
    ISplineTransformer,
    MSplineTransformer,
    NaturalCubicSplineTransformer,
    TensorProductSplineTransformer,
)


@pytest.fixture
def top_coded():
    # ~40% of the rows sit exactly on the cap of 100.
    return np.minimum(np.random.default_rng(0).uniform(0, 160, (1000, 1)), 100.0)


@pytest.fixture
def zero_inflated():
    rng = np.random.default_rng(1)
    return np.where(rng.random((1000, 1)) < 0.6, 0.0, rng.exponential(5.0, (1000, 1)))


def _interior(knots, degree):
    return knots[degree + 1 : len(knots) - degree - 1]


def _dense_grid(X, n=2001):
    return np.linspace(X.min(), X.max(), n).reshape(-1, 1)


@pytest.mark.parametrize("cls", [BSplineTransformer, MSplineTransformer, ISplineTransformer])
@pytest.mark.parametrize("data", ["top_coded", "zero_inflated"])
def test_bmi_interior_knots_are_unique_and_strictly_inside(cls, data, request):
    X = request.getfixturevalue(data)
    transformer = cls(output_dim=6).fit(X)
    knots = transformer.knots_[0]
    interior = _interior(knots, transformer.degree)

    assert len(interior) == 6 - transformer.degree - 1
    assert len(np.unique(interior)) == len(interior)
    assert (interior > X.min()).all() and (interior < X.max()).all()
    # Boundary multiplicity stays exactly degree + 1.
    assert (knots == X.min()).sum() == transformer.degree + 1
    assert (knots == X.max()).sum() == transformer.degree + 1
    assert (np.abs(transformer.transform(_dense_grid(X))).max(axis=0) > 0).all()


def test_bspline_is_a_partition_of_unity_at_the_cap(top_coded):
    transformer = BSplineTransformer(output_dim=6).fit(top_coded)
    basis = transformer.transform(top_coded)
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-9)
    # Values on and beyond the cap (clipped onto x_max) keep a full basis row.
    np.testing.assert_allclose(transformer.transform(np.array([[100.0], [150.0]])).sum(axis=1), 1.0, atol=1e-9)


def test_ispline_equals_one_at_the_cap(top_coded):
    transformer = ISplineTransformer(output_dim=6).fit(top_coded)
    np.testing.assert_allclose(transformer.transform(np.array([[100.0]])), 1.0, atol=1e-12)


@pytest.mark.parametrize("cls", [NaturalCubicSplineTransformer, CubicRegressionSplineTransformer])
def test_cubic_families_place_unique_knots_on_zero_inflated_data(cls, zero_inflated):
    transformer = cls(output_dim=6, placement_strategy="quantile").fit(zero_inflated)
    knots = transformer.knots_[0]
    assert len(np.unique(knots)) == len(knots)
    basis = transformer.transform(_dense_grid(zero_inflated))
    assert np.linalg.matrix_rank(basis) == basis.shape[1]
    assert (np.abs(basis).max(axis=0) > 0).all()


def test_natural_spline_keeps_both_endpoints_once(zero_inflated):
    knots = NaturalCubicSplineTransformer(output_dim=6, placement_strategy="quantile").fit(zero_inflated).knots_[0]
    assert len(knots) == 7
    assert knots[0] == zero_inflated.min() and knots[-1] == zero_inflated.max()
    assert (knots[1:-1] > knots[0]).all() and (knots[1:-1] < knots[-1]).all()


def test_tensor_product_has_no_degenerate_columns_on_zero_inflated_marginal(zero_inflated):
    X = np.hstack([zero_inflated, np.random.default_rng(2).normal(size=(len(zero_inflated), 1))])
    transformer = TensorProductSplineTransformer(output_dim=8, placement_strategy="quantile").fit(X)
    g0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
    g1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 200)
    grid = np.array(np.meshgrid(g0, g1)).reshape(2, -1).T
    assert (np.abs(transformer.transform(grid)).max(axis=0) > 0).all()


@pytest.mark.parametrize("levels", [[0.0, 1.0], [0.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0]])
def test_low_cardinality_feature_keeps_a_full_rank_bspline_basis(levels):
    X = np.random.default_rng(3).choice(levels, size=(300, 1))
    transformer = BSplineTransformer(output_dim=6).fit(X)
    basis = transformer.transform(_dense_grid(X))
    assert np.linalg.matrix_rank(basis) == 6


def test_supplement_interior_knots_drops_boundary_and_duplicate_knots():
    x = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 10.0, 10.0])
    knots = supplement_interior_knots(x, np.array([0.0, 2.0, 2.0, 10.0, 12.0]), 3)
    assert len(knots) == 3
    assert 2.0 in knots  # an existing interior knot is always kept
    assert (knots > 0.0).all() and (knots < 10.0).all()
    assert len(np.unique(knots)) == 3


def test_supplement_interior_knots_never_down_samples():
    x = np.linspace(0, 1, 50)
    knots = np.array([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_array_equal(supplement_interior_knots(x, knots, 2), knots)
