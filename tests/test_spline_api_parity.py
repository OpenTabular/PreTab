"""Shared spline API tests (placement_strategy / target_aware / task / include_bias).

These lock in that the new options default to the historical behaviour (so legacy
output is preserved) and that they are actually wired through ``fit``.
"""

import numpy as np
import pytest

from pretab.exceptions import IncompatibleParamsError
from pretab.transformers import (
    CubicRegressionSplineTransformer,
    NaturalCubicSplineTransformer,
    PSplineTransformer,
    TensorProductSplineTransformer,
    ThinPlateSplineTransformer,
)

# (class, output_dim) for the four knot-based splines that share the placement API.
KNOT_SPLINES = [
    (CubicRegressionSplineTransformer, 8),
    (NaturalCubicSplineTransformer, 6),
    (PSplineTransformer, 8),
    (TensorProductSplineTransformer, 5),
]

# Splines that also accept quantile placement. P-splines are uniform-only
# (equally-spaced knots for the difference penalty), so they are excluded here.
QUANTILE_SPLINES = [
    (CubicRegressionSplineTransformer, 8),
    (NaturalCubicSplineTransformer, 6),
    (TensorProductSplineTransformer, 5),
]

# The knot-based splines that also support the target-aware placement path.
# (The penalized ``pspline`` / ``tensorspline`` are unsupervised-only.)
TARGET_AWARE_SPLINES = [
    (CubicRegressionSplineTransformer, 8),
    (NaturalCubicSplineTransformer, 6),
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


@pytest.mark.parametrize(("cls", "output_dim"), KNOT_SPLINES)
def test_default_strategy_matches_explicit_uniform(cls, output_dim, X_uniform):
    """Not passing a strategy must equal placement_strategy='uniform' (legacy placement)."""
    default = cls(output_dim=output_dim).fit_transform(X_uniform)
    explicit = cls(output_dim=output_dim, placement_strategy="uniform").fit_transform(X_uniform)
    assert default.shape == explicit.shape
    np.testing.assert_allclose(default, explicit, rtol=1e-10)


@pytest.mark.parametrize(("cls", "output_dim"), QUANTILE_SPLINES)
def test_quantile_strategy_runs_and_differs(cls, output_dim, X_skewed):
    """placement_strategy='quantile' produces a finite basis of the same width as uniform."""
    uniform = cls(output_dim=output_dim, placement_strategy="uniform").fit_transform(X_skewed)
    quantile = cls(output_dim=output_dim, placement_strategy="quantile").fit_transform(X_skewed)
    assert quantile.shape == uniform.shape
    assert np.isfinite(quantile).all()
    # On skewed data, quantile placement should differ from uniform placement.
    assert not np.allclose(quantile, uniform)


@pytest.mark.parametrize(("cls", "output_dim"), TARGET_AWARE_SPLINES)
def test_selector_places_target_aware_knots(cls, output_dim, X_uniform, y_reg):
    """Target-aware placement yields a finite, fit-consistent basis when y is supplied."""
    transformer = cls(output_dim=output_dim, target_aware=True, placement_strategy="cart")
    Xt = transformer.fit_transform(X_uniform, y_reg)
    assert np.isfinite(Xt).all()
    np.testing.assert_allclose(Xt, transformer.transform(X_uniform), rtol=1e-10)


@pytest.mark.parametrize(("cls", "output_dim"), TARGET_AWARE_SPLINES)
def test_selector_requires_y(cls, output_dim, X_uniform):
    """Target-aware placement without y raises a typed error."""
    with pytest.raises(IncompatibleParamsError):
        cls(output_dim=output_dim, target_aware=True, placement_strategy="cart").fit(X_uniform)


def test_pspline_include_bias_adds_one_column_per_feature(X_uniform):
    no_bias = PSplineTransformer(output_dim=8).fit_transform(X_uniform)
    with_bias = PSplineTransformer(output_dim=8, include_bias=True).fit_transform(X_uniform)
    assert with_bias.shape[1] == no_bias.shape[1] + X_uniform.shape[1]
    assert np.allclose(with_bias[:, 0], 1.0)


def test_tensor_include_bias_widens_interaction(X_uniform):
    no_bias = TensorProductSplineTransformer(output_dim=4).fit_transform(X_uniform)
    with_bias = TensorProductSplineTransformer(output_dim=4, include_bias=True).fit_transform(X_uniform)
    assert with_bias.shape[1] > no_bias.shape[1]
    assert np.isfinite(with_bias).all()


def test_thinplate_include_bias_adds_one_column():
    X = np.linspace(0, 1, 40).reshape(-1, 1)
    no_bias = ThinPlateSplineTransformer(n_components=6, random_state=0).fit_transform(X)
    with_bias = ThinPlateSplineTransformer(n_components=6, include_bias=True, random_state=0).fit_transform(X)
    assert with_bias.shape[1] == no_bias.shape[1] + 1
    assert np.allclose(with_bias[:, 0], 1.0)


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (CubicRegressionSplineTransformer, {"target_aware", "placement_strategy", "task", "include_bias"}),
        (NaturalCubicSplineTransformer, {"degree", "target_aware", "placement_strategy", "task"}),
        (PSplineTransformer, {"placement_strategy", "include_bias"}),
        (TensorProductSplineTransformer, {"placement_strategy", "include_bias"}),
        (ThinPlateSplineTransformer, {"n_components", "landmark_strategy", "rank_strategy", "include_bias"}),
    ],
)
def test_new_params_exposed_in_get_params(cls, expected):
    params = set(cls().get_params())
    assert expected <= params


def test_tensor_penalty_matrix_signature_parity():
    X = np.random.default_rng(3).uniform(size=(30, 2))
    transformer = TensorProductSplineTransformer(output_dim=4).fit(X)
    P = transformer.get_penalty_matrix(feature_index=1)
    assert P.shape[0] == P.shape[1]
    assert np.allclose(P, P.T)


def test_thinplate_penalty_matrix_accepts_feature_index():
    X = np.linspace(0, 1, 40).reshape(-1, 1)
    transformer = ThinPlateSplineTransformer(n_components=6, include_bias=True, random_state=0).fit(X)
    P = transformer.get_penalty_matrix(feature_index=0)
    assert P.shape == (7, 7)
    assert np.allclose(P[0, :], 0.0) and np.allclose(P[:, 0], 0.0)
