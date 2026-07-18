"""Phase 15: ``output_dim`` is the single, exact width knob.

For every fixed-basis family the transformed width is a deterministic function of
``output_dim`` and the number of input features, and the fitted
``total_output_dim_`` attribute always equals the real number of output columns.
Two families are data-dependent by design and are asserted separately: PLE
(``output_dim`` is a per-feature bin *cap*, not an exact count) and the custom
binner (always a single ordinal column).
"""

import numpy as np
import pytest

from pretab.transformers import (
    BSplineTransformer,
    CubicSplineTransformer,
    CustomBinTransformer,
    ISplineTransformer,
    MSplineTransformer,
    NaturalCubicSplineTransformer,
    PLETransformer,
    PSplineTransformer,
    RBFExpansionTransformer,
    ReLUExpansionTransformer,
    SigmoidExpansionTransformer,
    TanhExpansionTransformer,
    TensorProductSplineTransformer,
    ThinPlateSplineTransformer,
)

OUTPUT_DIM = 6


@pytest.fixture
def X():
    rng = np.random.RandomState(0)
    return rng.uniform(-3, 3, size=(120, 3))


@pytest.fixture
def Xy():
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 1, size=(200, 3))
    y = np.sin(X[:, 0]) + 0.1 * rng.randn(200)
    return X, y


# Splines whose transformed width is exactly ``n_features * output_dim``.
PER_FEATURE_SPLINES = [
    CubicSplineTransformer,
    NaturalCubicSplineTransformer,
    PSplineTransformer,
    MSplineTransformer,
    ISplineTransformer,
]


@pytest.mark.parametrize("cls", PER_FEATURE_SPLINES)
def test_spline_width_is_n_features_times_output_dim(cls, X):
    transformer = cls(output_dim=OUTPUT_DIM).fit(X)
    Xt = transformer.transform(X)
    assert Xt.shape[1] == X.shape[1] * OUTPUT_DIM
    assert transformer.total_output_dim_ == Xt.shape[1]


# Feature maps expand each column into ``output_dim`` basis columns on the
# non-target (uniform) placement path.
FEATURE_MAP_CLASSES = [
    RBFExpansionTransformer,
    ReLUExpansionTransformer,
    SigmoidExpansionTransformer,
    TanhExpansionTransformer,
]


@pytest.mark.parametrize("cls", FEATURE_MAP_CLASSES)
def test_feature_map_width_is_n_features_times_output_dim(cls, X):
    transformer = cls(output_dim=OUTPUT_DIM, use_target=False).fit(X)
    Xt = transformer.transform(X)
    assert Xt.shape[1] == X.shape[1] * OUTPUT_DIM
    assert transformer.total_output_dim_ == Xt.shape[1]


def test_thinplate_width_is_output_dim():
    # Thin-plate regression splines are univariate.
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, size=(120, 1))
    transformer = ThinPlateSplineTransformer(output_dim=OUTPUT_DIM).fit(X)
    Xt = transformer.transform(X)
    assert Xt.shape[1] == OUTPUT_DIM
    assert transformer.total_output_dim_ == Xt.shape[1]


@pytest.mark.parametrize("include_bias", [False, True])
def test_bspline_width_tracks_include_bias(X, include_bias):
    transformer = BSplineTransformer(output_dim=OUTPUT_DIM, include_bias=include_bias).fit(X)
    Xt = transformer.transform(X)
    expected_per_feature = OUTPUT_DIM + (1 if include_bias else 0)
    assert Xt.shape[1] == X.shape[1] * expected_per_feature
    assert transformer.total_output_dim_ == Xt.shape[1]


def test_tensor_width_is_output_dim_to_the_n_dims():
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, size=(120, 2))
    transformer = TensorProductSplineTransformer(output_dim=4).fit(X)
    Xt = transformer.transform(X)
    assert Xt.shape[1] == 4 ** X.shape[1]
    assert transformer.total_output_dim_ == Xt.shape[1]


def test_ple_output_dim_is_a_per_feature_cap(Xy):
    X, y = Xy
    transformer = PLETransformer(output_dim=OUTPUT_DIM).fit(X, y)
    Xt = transformer.transform(X)
    # Data-dependent: never more than the cap, at least one bin per feature.
    assert X.shape[1] <= Xt.shape[1] <= X.shape[1] * OUTPUT_DIM
    assert transformer.total_output_dim_ == Xt.shape[1]


def test_custombin_is_always_a_single_ordinal_column():
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    transformer = CustomBinTransformer(output_dim=OUTPUT_DIM).fit(X)
    Xt = transformer.transform(X)
    assert Xt.shape[1] == 1
    assert transformer.total_output_dim_ == 1
