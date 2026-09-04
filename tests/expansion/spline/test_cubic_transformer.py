import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pretab.transformers import CubicRegressionSplineTransformer


def test_cubic_spline_single_feature_shape():
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    transformer = CubicRegressionSplineTransformer(output_dim=8)
    Xt = transformer.fit_transform(X)

    # output_dim non-bias columns (m = 3 + K interior knots) per feature
    assert Xt.shape == (20, 8)
    assert transformer.n_knots_ == [5]
    assert transformer.total_output_dim_ == 8
    assert np.isfinite(Xt).all()


def test_cubic_spline_multi_feature_shape():
    X = np.random.rand(15, 3)
    transformer = CubicRegressionSplineTransformer(output_dim=9, include_bias=True)
    Xt = transformer.fit_transform(X)

    expected_dim = (1 + 9) * 3  # bias + output_dim columns, per feature
    assert Xt.shape == (15, expected_dim)
    assert transformer.total_output_dim_ == expected_dim
    assert np.isfinite(Xt).all()


def test_cubic_spline_output_consistency():
    X = np.random.rand(10, 2)
    transformer = CubicRegressionSplineTransformer(output_dim=7)
    transformer.fit(X)
    Xt1 = transformer.transform(X)
    Xt2 = transformer.fit_transform(X)

    assert Xt1.shape == Xt2.shape
    np.testing.assert_allclose(Xt1, Xt2, rtol=1e-5)


def test_cubic_spline_penalty_matrix_shape():
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    transformer = CubicRegressionSplineTransformer(output_dim=10)
    transformer.fit(X)
    P = transformer.get_penalty_matrix()

    assert P.shape == (10, 10)
    assert np.allclose(P, P.T, atol=1e-6)


def test_cubic_spline_penalty_matrix_last_knot_value():
    # Hand-derived closed form: 36 * integral of (x - knot) ** 2 dx from knot to
    # x_max. With interior knots at [2.5, 5, 7.5] and x_max = 10, the last-knot
    # entry is 187.5.
    X = np.linspace(0, 10, 200).reshape(-1, 1)
    transformer = CubicRegressionSplineTransformer(output_dim=6, placement_strategy="uniform")
    transformer.fit(X)

    assert transformer.knots_[0] == pytest.approx([2.5, 5.0, 7.5])
    P = transformer.get_penalty_matrix()
    assert P[-1, -1] == pytest.approx(187.5, rel=1e-6)


def test_cubic_spline_penalty_matrix_penalizes_polynomial_columns():
    # x**2 and x**3 have real curvature (f'' = 2 and f'' = 6x) and must not be
    # silently unpenalized; only the linear x column has f'' = 0 everywhere.
    X = np.linspace(0, 10, 200).reshape(-1, 1)
    transformer = CubicRegressionSplineTransformer(output_dim=6).fit(X)
    P = transformer.get_penalty_matrix()

    assert not np.allclose(P[1, :], 0.0)  # x**2 row
    assert not np.allclose(P[2, :], 0.0)  # x**3 row
    assert np.allclose(P[0, :], 0.0)  # x row: identically zero second derivative


def test_cubic_feature_names_out():
    X = np.random.rand(20, 2)
    transformer = CubicRegressionSplineTransformer(output_dim=8)
    Xt = transformer.fit_transform(X)

    names = transformer.get_feature_names_out(["a", "b"])
    assert len(names) == Xt.shape[1]
    assert names[0] == "a_cs0"
    assert all(name.startswith(("a_cs", "b_cs")) for name in names)


def test_cubic_feature_names_out_default_input():
    X = np.random.rand(15, 2)
    transformer = CubicRegressionSplineTransformer(output_dim=7).fit(X)

    names = transformer.get_feature_names_out()
    assert len(names) == sum(transformer.n_basis_)
    assert names[0].startswith("x0_cs")


def test_cubic_allow_nan_tag():
    tags = CubicRegressionSplineTransformer().__sklearn_tags__()
    assert tags.input_tags.allow_nan is True


def test_cubic_transform_requires_fit():
    transformer = CubicRegressionSplineTransformer()
    with pytest.raises(NotFittedError):
        transformer.transform(np.random.rand(5, 1))
    with pytest.raises(NotFittedError):
        transformer.get_penalty_matrix()


def test_cubic_spline_does_not_retain_training_design_matrix():
    """Regression guard for issue #19: fitted size must not scale with n_samples."""
    import pickle

    small = CubicRegressionSplineTransformer(output_dim=7).fit(np.random.rand(50, 1))
    large = CubicRegressionSplineTransformer(output_dim=7).fit(np.random.rand(20_000, 1))

    assert not hasattr(small, "designs_")
    assert len(pickle.dumps(large)) < 2 * len(pickle.dumps(small))
