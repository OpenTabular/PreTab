"""Phase 9 regression tests: shared spline API (strategy / selector / task / include_bias).

These lock in that the new options default to the historical behaviour (so legacy
output is preserved) and that they are actually wired through ``fit``.
"""

import numpy as np
import pytest

from pretab.core.exceptions import IncompatibleParamsError
from pretab.transformers import (
    CubicSplineTransformer,
    NaturalCubicSplineTransformer,
    PSplineTransformer,
    TensorProductSplineTransformer,
    ThinPlateSplineTransformer,
)
from pretab.transformers.splines.knot_selectors import CARTKnotSelector

# (class, n_basis) for the four knot-based legacy splines that gained strategy/selector.
KNOT_SPLINES = [
    (CubicSplineTransformer, 8),
    (NaturalCubicSplineTransformer, 6),
    (PSplineTransformer, 8),
    (TensorProductSplineTransformer, 4),
]


@pytest.fixture
def X_uniform():
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, size=(200, 2))


@pytest.fixture
def X_skewed():
    rng = np.random.default_rng(1)
    return rng.exponential(1.0, size=(200, 2))


@pytest.fixture
def y_reg():
    rng = np.random.default_rng(2)
    return rng.normal(size=200)


@pytest.mark.parametrize(("cls", "n_basis"), KNOT_SPLINES)
def test_default_strategy_matches_explicit_uniform(cls, n_basis, X_uniform):
    """Not passing strategy must equal strategy='uniform' (legacy placement)."""
    default = cls(n_basis=n_basis).fit_transform(X_uniform)
    explicit = cls(n_basis=n_basis, strategy="uniform").fit_transform(X_uniform)
    assert default.shape == explicit.shape
    np.testing.assert_allclose(default, explicit, rtol=1e-10)


@pytest.mark.parametrize(("cls", "n_basis"), KNOT_SPLINES)
def test_quantile_strategy_runs_and_differs(cls, n_basis, X_skewed):
    """strategy='quantile' produces a finite basis of the same width as uniform."""
    uniform = cls(n_basis=n_basis, strategy="uniform").fit_transform(X_skewed)
    quantile = cls(n_basis=n_basis, strategy="quantile").fit_transform(X_skewed)
    assert quantile.shape == uniform.shape
    assert np.isfinite(quantile).all()
    # On skewed data, quantile placement should differ from uniform placement.
    assert not np.allclose(quantile, uniform)


@pytest.mark.parametrize(("cls", "n_basis"), KNOT_SPLINES)
def test_selector_places_target_aware_knots(cls, n_basis, X_uniform, y_reg):
    """A selector yields a finite, fit-consistent basis when y is supplied."""
    selector = CARTKnotSelector(spline_type="bspline", degree=3)
    transformer = cls(n_basis=n_basis, selector=selector)
    Xt = transformer.fit_transform(X_uniform, y_reg)
    assert np.isfinite(Xt).all()
    np.testing.assert_allclose(Xt, transformer.transform(X_uniform), rtol=1e-10)


@pytest.mark.parametrize(("cls", "n_basis"), KNOT_SPLINES)
def test_selector_requires_y(cls, n_basis, X_uniform):
    """Using a selector without y raises a typed error."""
    selector = CARTKnotSelector(spline_type="bspline", degree=3)
    with pytest.raises(IncompatibleParamsError):
        cls(n_basis=n_basis, selector=selector).fit(X_uniform)


def test_pspline_include_bias_adds_one_column_per_feature(X_uniform):
    no_bias = PSplineTransformer(n_basis=8).fit_transform(X_uniform)
    with_bias = PSplineTransformer(n_basis=8, include_bias=True).fit_transform(X_uniform)
    assert with_bias.shape[1] == no_bias.shape[1] + X_uniform.shape[1]
    assert np.allclose(with_bias[:, 0], 1.0)


def test_tensor_include_bias_widens_interaction(X_uniform):
    no_bias = TensorProductSplineTransformer(n_basis=4).fit_transform(X_uniform)
    with_bias = TensorProductSplineTransformer(n_basis=4, include_bias=True).fit_transform(X_uniform)
    assert with_bias.shape[1] > no_bias.shape[1]
    assert np.isfinite(with_bias).all()


def test_thinplate_include_bias_adds_one_column():
    X = np.linspace(0, 1, 40).reshape(-1, 1)
    no_bias = ThinPlateSplineTransformer(n_basis=6).fit_transform(X)
    with_bias = ThinPlateSplineTransformer(n_basis=6, include_bias=True).fit_transform(X)
    assert with_bias.shape[1] == no_bias.shape[1] + 1
    assert np.allclose(with_bias[:, 0], 1.0)


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (CubicSplineTransformer, {"strategy", "selector", "task", "include_bias"}),
        (NaturalCubicSplineTransformer, {"degree", "strategy", "selector", "task"}),
        (PSplineTransformer, {"strategy", "selector", "task", "include_bias"}),
        (TensorProductSplineTransformer, {"strategy", "selector", "task", "include_bias"}),
        (ThinPlateSplineTransformer, {"include_bias"}),
    ],
)
def test_new_params_exposed_in_get_params(cls, expected):
    params = set(cls().get_params())
    assert expected <= params


def test_tensor_penalty_matrix_signature_parity():
    X = np.random.default_rng(3).uniform(size=(30, 2))
    transformer = TensorProductSplineTransformer(n_basis=4).fit(X)
    P = transformer.get_penalty_matrix(feature_index=1)
    assert P.shape[0] == P.shape[1]
    assert np.allclose(P, P.T)


def test_thinplate_penalty_matrix_accepts_feature_index():
    X = np.linspace(0, 1, 40).reshape(-1, 1)
    transformer = ThinPlateSplineTransformer(n_basis=6, include_bias=True).fit(X)
    P = transformer.get_penalty_matrix(feature_index=0)
    assert P.shape == (7, 7)
    assert np.allclose(P[0, :], 0.0) and np.allclose(P[:, 0], 0.0)
