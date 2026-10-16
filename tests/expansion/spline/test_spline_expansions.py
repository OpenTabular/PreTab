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


def test_bspline_default_design_is_full_rank(data):
    """Regression guard for issue #33: the default basis must not be rank-deficient."""
    X, _ = data
    Xt = BSplineTransformer(output_dim=8).fit_transform(X)
    assert np.linalg.matrix_rank(Xt) == Xt.shape[1]
    assert np.linalg.cond(Xt) < 1e6


def test_bspline_basis_is_a_partition_of_unity(data):
    """Every row of the (bias-free) basis sums to 1, so a prepended bias column

    would be an exact linear combination of the rest -- the root cause of #33.
    """
    X, _ = data
    Xt = BSplineTransformer(output_dim=8, include_bias=False).fit_transform(X)
    np.testing.assert_allclose(Xt.sum(axis=1), 1.0, atol=1e-9)


def test_bspline_include_bias_still_available_opt_in(data):
    X, _ = data
    Xt = BSplineTransformer(output_dim=8, include_bias=True).fit_transform(X)
    assert Xt.shape == (200, 9)


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
    transformer = BSplineTransformer(output_dim=8, include_bias=False, target_aware=True, placement_strategy="cart")
    Xt = transformer.fit_transform(X, y)
    assert Xt.shape == (200, 8)
    assert np.isfinite(Xt).all()


def test_spline_penalty_matrix_symmetric(data):
    X, _ = data
    transformer = BSplineTransformer(output_dim=8, include_bias=True).fit(X)
    P = transformer.get_penalty_matrix()
    assert P.shape[0] == P.shape[1]
    assert np.allclose(P, P.T, atol=1e-9)


# --- explicit knot_locations ----------------------------------------------------------


@pytest.mark.parametrize("cls", [BSplineTransformer, MSplineTransformer, ISplineTransformer])
@pytest.mark.parametrize("knots", [[5.0], [2.0, 4.0, 6.0, 8.0], list(np.linspace(1, 9, 12))])
def test_knot_locations_set_the_width_regardless_of_output_dim(cls, knots):
    """knot_locations raised unless its length matched the default output_dim, although
    the docstring says output_dim is ignored when knots are given."""
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    transformer = cls(knot_locations=knots).fit(X)
    np.testing.assert_allclose(transformer.knots_[0][4:-4], knots)
    assert transformer.transform(X).shape == (50, len(knots) + 4)


@pytest.mark.parametrize(
    ("knots", "min_output_dim", "max_output_dim"),
    [([2.0, 4.0, 6.0, 8.0], 5, 6), ([5.0], 7, 8)],
    ids=["not-trimmed", "not-padded"],
)
def test_knot_locations_are_kept_in_adaptive_mode(knots, min_output_dim, max_output_dim):
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    transformer = BSplineTransformer(
        knot_locations=np.array(knots), adaptive=True, min_output_dim=min_output_dim, max_output_dim=max_output_dim
    )
    np.testing.assert_array_equal(transformer.fit(X).knots_[0][4:-4], knots)


def test_knot_locations_outside_the_range_are_dropped_with_a_warning():
    from pretab.exceptions import DataWarning

    X = np.linspace(0, 10, 50).reshape(-1, 1)
    with pytest.warns(DataWarning, match=r"\[-5\.0, 15\.0\] lie outside"):
        transformer = BSplineTransformer(knot_locations=np.array([-5.0, 3.0, 3.0, 15.0])).fit(X)
    np.testing.assert_array_equal(transformer.knots_[0][4:-4], [3.0])


def test_knot_locations_on_the_boundary_are_dropped_instead_of_giving_a_dead_column():
    """A knot on the boundary was kept and repeated the boundary knot, so one basis
    function had zero-width support and was zero on all of the training data."""
    from pretab.exceptions import DataWarning

    X = np.linspace(0, 10, 50).reshape(-1, 1)
    with pytest.warns(DataWarning, match=r"\[0\.0\] lie outside"):
        transformer = BSplineTransformer(knot_locations=np.array([0.0, 5.0])).fit(X)
    np.testing.assert_array_equal(transformer.knots_[0][4:-4], [5.0])
    assert (transformer.transform(X).max(axis=0) > 0).all()


@pytest.mark.parametrize("knots", [[[1.0, 2.0]], ["a"], [np.nan]])
def test_invalid_knot_locations_raise_a_typed_error(knots):
    from pretab.exceptions import InvalidParamError

    with pytest.raises(InvalidParamError, match="knot_locations"):
        BSplineTransformer(knot_locations=knots).fit(np.linspace(0, 10, 50).reshape(-1, 1))
