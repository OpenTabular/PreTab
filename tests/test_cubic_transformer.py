import numpy as np
import pytest
import warnings
from pretab.transformers import CubicSplineTransformer


def test_cubic_spline_single_feature_shape():
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    transformer = CubicSplineTransformer(n_knots=5)
    Xt = transformer.fit_transform(X)

    expected_dim = 3 + 5  # x, x², x³, and 5 truncated cubics
    assert Xt.shape == (20, expected_dim)
    assert np.isfinite(Xt).all()


def test_cubic_spline_multi_feature_shape():
    X = np.random.rand(15, 3)
    transformer = CubicSplineTransformer(n_knots=6, include_bias=True)
    Xt = transformer.fit_transform(X)

    expected_dim = (1 + 3 + 6) * 3  # bias + x,x²,x³ + 6 truncated cubics, per feature
    assert Xt.shape == (15, expected_dim)
    assert np.isfinite(Xt).all()


def test_cubic_spline_output_consistency():
    X = np.random.rand(10, 2)
    transformer = CubicSplineTransformer(n_knots=4)
    transformer.fit(X)
    Xt1 = transformer.transform(X)
    Xt2 = transformer.fit_transform(X)

    assert Xt1.shape == Xt2.shape
    np.testing.assert_allclose(Xt1, Xt2, rtol=1e-5)


def test_cubic_spline_penalty_matrix_shape():
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    transformer = CubicSplineTransformer(n_knots=7)
    transformer.fit(X)
    P = transformer.get_penalty_matrix()

    expected_dim = 3 + 7
    assert P.shape == (expected_dim, expected_dim)
    assert np.allclose(P, P.T, atol=1e-6)
