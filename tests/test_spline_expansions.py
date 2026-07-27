import numpy as np
import pytest

from pretab.transformers import (
    BSplineTransformer,
    ISplineTransformer,
    MSplineTransformer,
)


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, size=(200, 1))
    y = np.sin(X[:, 0]) + 0.1 * rng.randn(200)
    return X, y


def test_bspline_shape_with_bias(data):
    X, _ = data
    transformer = BSplineTransformer(output_dim=8, include_bias=True)
    Xt = transformer.fit_transform(X)
    assert Xt.shape == (200, 9)  # 8 basis + 1 bias
    assert np.isfinite(Xt).all()


def test_bspline_shape_without_bias(data):
    X, _ = data
    transformer = BSplineTransformer(output_dim=8, include_bias=False)
    Xt = transformer.fit_transform(X)
    assert Xt.shape == (200, 8)
    assert transformer.get_n_features_out() == 8


def test_bspline_multi_feature_shape():
    rng = np.random.RandomState(1)
    X = rng.uniform(0, 1, size=(120, 3))
    transformer = BSplineTransformer(output_dim=6, include_bias=False)
    Xt = transformer.fit_transform(X)
    assert Xt.shape == (120, 18)  # 6 basis per feature, 3 features
    assert transformer.get_n_features_out() == 18


def test_bspline_reproducible(data):
    X, _ = data
    a = BSplineTransformer(output_dim=7).fit_transform(X)
    b = BSplineTransformer(output_dim=7).fit_transform(X)
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_bspline_feature_names_out(data):
    X, _ = data
    transformer = BSplineTransformer(output_dim=8, include_bias=True).fit(X)
    names = transformer.get_feature_names_out(["age"])
    assert len(names) == transformer.get_n_features_out()
    assert names[0] == "age_bs0"


def test_bspline_removed_count_names_raise():
    with pytest.raises(TypeError):
        BSplineTransformer(n_basis_functions=8)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        BSplineTransformer(n_knots=8)  # type: ignore[call-arg]


def test_bspline_rejects_small_basis():
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    with pytest.raises(ValueError, match="output_dim must be >= degree"):
        BSplineTransformer(output_dim=3).fit(X)


def test_bspline_rejects_large_basis():
    X = np.linspace(0, 1, 60).reshape(-1, 1)
    with pytest.raises(ValueError, match="<= 50"):
        BSplineTransformer(output_dim=60).fit(X)


def test_mspline_non_negative(data):
    X, _ = data
    transformer = MSplineTransformer(output_dim=8)
    Xt = transformer.fit_transform(X)
    assert Xt.shape == (200, 8)
    assert np.all(Xt >= -1e-9)


def test_mspline_handles_nan():
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    X[5] = np.nan
    transformer = MSplineTransformer(output_dim=6)
    Xt = transformer.fit_transform(X)
    assert Xt.shape == (50, 6)
    assert np.isfinite(Xt).all()


def test_ispline_monotonic_increasing():
    X = np.linspace(0, 10, 200).reshape(-1, 1)
    transformer = ISplineTransformer(output_dim=8, include_bias=False)
    Xt = transformer.fit_transform(X)
    # Each basis column is monotonically non-decreasing in x
    for j in range(Xt.shape[1]):
        assert np.all(np.diff(Xt[:, j]) >= -1e-9)


def test_ispline_bounded_unit_interval():
    X = np.linspace(0, 10, 200).reshape(-1, 1)
    Xt = ISplineTransformer(output_dim=8).fit_transform(X)
    assert np.all(Xt >= -1e-9)
    assert np.all(Xt <= 1.0 + 1e-9)


def test_ispline_shape_multi_feature():
    rng = np.random.RandomState(2)
    X = rng.uniform(0, 5, size=(100, 2))
    transformer = ISplineTransformer(output_dim=7, include_bias=False)
    Xt = transformer.fit_transform(X)
    assert Xt.shape == (100, 14)


def test_spline_with_cart_knot_selector(data):
    X, y = data
    transformer = BSplineTransformer(
        output_dim=8, include_bias=False, target_aware=True, placement_strategy="cart"
    )
    Xt = transformer.fit_transform(X, y)
    assert Xt.shape == (200, 8)
    assert np.isfinite(Xt).all()


def test_spline_penalty_matrix_symmetric(data):
    X, _ = data
    transformer = BSplineTransformer(output_dim=8, include_bias=True).fit(X)
    P = transformer.get_penalty_matrix()
    assert P.shape[0] == P.shape[1]
    assert np.allclose(P, P.T, atol=1e-9)


# --------------------------------------------------------------------------- #
# A partition-of-unity basis must not carry a redundant intercept column.
#
# ``BSplineTransformer`` used to default to ``include_bias=True``. A B-spline
# basis over a clamped knot vector sums to 1 on every row, so the prepended
# column of ones is an exact linear combination of the rest and the design
# matrix is singular (condition number ~1e15).
# --------------------------------------------------------------------------- #
def test_bspline_default_design_is_full_rank():
    X = np.linspace(0, 1, 200).reshape(-1, 1)

    design = BSplineTransformer(output_dim=8).fit_transform(X)

    assert design.shape[1] == 8
    assert np.linalg.matrix_rank(design) == design.shape[1]
    assert np.linalg.cond(design) < 1e6


def test_bspline_basis_is_a_partition_of_unity():
    # This is *why* the bias column is redundant; pin it so the reasoning holds.
    X = np.linspace(0, 1, 200).reshape(-1, 1)

    design = BSplineTransformer(output_dim=8).fit_transform(X)

    np.testing.assert_allclose(design.sum(axis=1), 1.0)


def test_bspline_include_bias_still_available_opt_in():
    X = np.linspace(0, 1, 200).reshape(-1, 1)

    design = BSplineTransformer(output_dim=8, include_bias=True).fit_transform(X)

    assert design.shape[1] == 9
    np.testing.assert_allclose(design[:, 0], 1.0)


@pytest.mark.parametrize("cls", [BSplineTransformer, MSplineTransformer, ISplineTransformer])
def test_bmi_splines_default_to_exactly_output_dim(cls):
    X = np.linspace(0, 1, 200).reshape(-1, 1)

    assert cls(output_dim=7).fit_transform(X).shape[1] == 7
