import warnings

import numpy as np
import pytest

from pretab.transformers import (
    BSplineTransformer,
    CubicRegressionSplineTransformer,
    ISplineTransformer,
    MSplineTransformer,
    NaturalCubicSplineTransformer,
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
    """A missing value is left out of knot placement and expands to a NaN row."""
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    X[5] = np.nan
    transformer = MSplineTransformer(output_dim=6)
    Xt = transformer.fit_transform(X)
    assert Xt.shape == (50, 6)
    assert np.isnan(Xt[5]).all()
    assert np.isfinite(np.delete(Xt, 5, axis=0)).all()


# --- missing values expand to NaN basis rows -------------------------------------------


@pytest.mark.parametrize("cls", [BSplineTransformer, MSplineTransformer, ISplineTransformer])
@pytest.mark.parametrize("include_bias", [False, True])
@pytest.mark.parametrize("out_of_range", ["clip", "extrapolate", "warn", "error"])
def test_bmi_spline_missing_value_expands_to_a_nan_row(cls, include_bias, out_of_range):
    """A missing value became a finite all-zero basis row: impossible data for the
    B/M-spline and, for the I-spline, the exact encoding of the training minimum."""
    X = np.linspace(0.0, 10.0, 50).reshape(-1, 1)
    transformer = cls(output_dim=5, include_bias=include_bias, policy={"out_of_range": out_of_range}).fit(X)
    observed = np.array([[0.0], [2.5], [10.0]])
    if out_of_range in ("clip", "extrapolate"):
        observed = np.vstack([observed, [[-3.0], [14.0]]])
    expected = transformer.transform(observed)

    probe = np.insert(observed, [0, 2], np.nan, axis=0)
    with warnings.catch_warnings():
        # A missing value is not out of range: it must not trigger "warn" / "error".
        warnings.simplefilter("error")
        out = transformer.transform(probe)

    missing = np.isnan(probe[:, 0])
    basis = slice(1, None) if include_bias else slice(None)
    assert np.isnan(out[missing, basis]).all()
    if include_bias:
        np.testing.assert_array_equal(out[:, 0], 1.0)
    np.testing.assert_array_equal(out[~missing], expected)


@pytest.mark.parametrize("cls", [BSplineTransformer, MSplineTransformer, ISplineTransformer])
def test_bmi_spline_missing_value_only_affects_its_own_feature_block(cls):
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 10.0, size=(60, 2))
    X[4, 0] = np.nan
    X[9, 1] = np.nan
    transformer = cls(output_dim=6).fit(X)
    out = transformer.transform(X)
    np.testing.assert_array_equal(np.isnan(out[:, :6]).any(axis=1), np.isnan(X[:, 0]))
    np.testing.assert_array_equal(np.isnan(out[:, 6:]).any(axis=1), np.isnan(X[:, 1]))
    assert np.isnan(out[4, :6]).all() and np.isnan(out[9, 6:]).all()


@pytest.mark.parametrize("scale", [1.0, 1e-2, 1e-3, 1e-7])
def test_mspline_basis_integrates_to_one_at_any_feature_scale(scale):
    """Regression guard for issue #63: absolute clipping thresholds flattened the
    basis on small-scale features and zeroed it on tiny-range ones."""
    from scipy.integrate import trapezoid

    X = np.random.default_rng(0).uniform(0, 1, (2000, 1)) * scale
    transformer = MSplineTransformer(output_dim=8, placement_strategy="uniform").fit(X)
    grid = np.linspace(X.min(), X.max(), 200_001)
    integrals = trapezoid(transformer.transform(grid.reshape(-1, 1)), grid, axis=0)
    np.testing.assert_allclose(integrals, 1.0, atol=1e-3)


def test_mspline_basis_is_scale_equivariant():
    X = np.random.default_rng(0).lognormal(0, 2, (2000, 1))
    unit = MSplineTransformer(output_dim=8).fit(X).transform(X)
    scaled = MSplineTransformer(output_dim=8).fit(X * 1e-4).transform(X * 1e-4)
    # M-spline values scale like 1 / span, so rescaling x by c rescales M by 1 / c.
    np.testing.assert_allclose(scaled * 1e-4, unit, rtol=1e-8)


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


def _exact_ispline(x, knots, degree):
    """Reference I-spline: the normalized antiderivative of each B-spline basis function."""
    from scipy.interpolate import BSpline

    n_coef = len(knots) - degree - 1
    columns = []
    for i in range(n_coef):
        antiderivative = BSpline(knots, np.eye(n_coef)[i], degree).antiderivative()
        lower, upper = antiderivative(knots[0]), antiderivative(knots[-1])
        columns.append((antiderivative(x) - lower) / (upper - lower))
    return np.column_stack(columns)


@pytest.mark.parametrize("output_dim", [6, 10, 30])
def test_ispline_matches_exact_integral_on_tight_knots(output_dim):
    """Regression guard for issue #53: knot spans narrower than a fixed quadrature
    grid step produced all-zero and misplaced columns."""
    X = np.random.default_rng(0).lognormal(0, 2, (2000, 1))
    transformer = ISplineTransformer(output_dim=output_dim).fit(X)
    Xt = transformer.transform(X)

    exact = _exact_ispline(X[:, 0], transformer.knots_[0], transformer.degree)
    np.testing.assert_allclose(Xt, exact, atol=1e-10)
    assert (Xt.max(axis=0) > 0).all()
    # Every I-spline row is ordered I_0 >= I_1 >= ... >= I_{K-1}.
    assert (np.diff(Xt, axis=1) <= 1e-12).all()


def test_ispline_reaches_zero_and_one_at_the_range_boundaries():
    X = np.random.default_rng(1).exponential(size=(500, 1))
    transformer = ISplineTransformer(output_dim=8).fit(X)
    boundary = transformer.transform(np.array([[X.min()], [X.max()]]))
    np.testing.assert_allclose(boundary[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(boundary[1], 1.0, atol=1e-12)


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


# --- target-aware splines search their own knot window (issue #69) ------------


def _step_data(seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, 1000)
    y = np.where(x > 5.0, 1.0, 0.0) + 0.3 * rng.normal(size=1000)
    return x.reshape(-1, 1), y


@pytest.mark.parametrize("cls", [BSplineTransformer, MSplineTransformer, ISplineTransformer])
@pytest.mark.parametrize("output_dim", [5, 6, 7])
def test_target_aware_bmi_spline_keeps_the_dominant_split(cls, output_dim):
    """Regression guard for issue #69: selector knots were trimmed by position, so
    the split where the target changes was routinely discarded."""
    X, y = _step_data()
    transformer = cls(output_dim=output_dim, target_aware=True, placement_strategy="cart").fit(X, y)
    interior = transformer.knots_[0][transformer.degree + 1 : -(transformer.degree + 1)]
    assert len(interior) == output_dim - transformer.degree - 1
    assert np.abs(interior - 5.0).min() < 0.1, interior


@pytest.mark.parametrize("cls", [CubicRegressionSplineTransformer, NaturalCubicSplineTransformer])
def test_target_aware_cubic_families_keep_the_dominant_split(cls):
    X, y = _step_data()
    knots = cls(output_dim=4, target_aware=True, placement_strategy="cart").fit(X, y).knots_[0]
    assert np.abs(knots - 5.0).min() < 0.1, knots


@pytest.mark.parametrize("cls", [BSplineTransformer, CubicRegressionSplineTransformer, NaturalCubicSplineTransformer])
def test_adaptive_spline_width_is_not_capped_at_the_legacy_window(cls):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 3000)
    y = np.sin(2 * x) * 3 + rng.normal(0, 0.3, 3000)
    transformer = cls(
        adaptive=True, min_output_dim=5, max_output_dim=40, target_aware=True, placement_strategy="cart"
    ).fit(x.reshape(-1, 1), y)
    assert 15 < transformer.total_output_dim_ <= 40


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


def test_knot_locations_are_kept_in_adaptive_mode():
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    knots = np.array([2.0, 4.0, 6.0, 8.0])
    transformer = BSplineTransformer(knot_locations=knots, adaptive=True, min_output_dim=5, max_output_dim=6)
    np.testing.assert_array_equal(transformer.fit(X).knots_[0][4:-4], [2, 4, 6, 8])


def test_knot_locations_outside_the_range_are_dropped_with_a_warning():
    from pretab.exceptions import DataWarning

    X = np.linspace(0, 10, 50).reshape(-1, 1)
    with pytest.warns(DataWarning, match=r"\[-5\.0, 15\.0\] lie outside"):
        transformer = BSplineTransformer(knot_locations=np.array([-5.0, 3.0, 3.0, 15.0])).fit(X)
    np.testing.assert_array_equal(transformer.knots_[0][4:-4], [3.0])


@pytest.mark.parametrize("knots", [[[1.0, 2.0]], ["a"], [np.nan]])
def test_invalid_knot_locations_raise_a_typed_error(knots):
    from pretab.exceptions import InvalidParamError

    with pytest.raises(InvalidParamError, match="knot_locations"):
        BSplineTransformer(knot_locations=knots).fit(np.linspace(0, 10, 50).reshape(-1, 1))
