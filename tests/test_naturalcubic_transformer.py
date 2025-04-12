import numpy as np
import pytest
import warnings
from pretab.transformers import NaturalCubicSplineTransformer


def test_natural_spline_single_feature_shape():
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    transformer = NaturalCubicSplineTransformer(n_knots=5)
    Xt = transformer.fit_transform(X)

    n_basis = transformer.designs_[0].shape[1]
    assert Xt.shape == (20, n_basis)
    assert np.isfinite(Xt).all()


def test_natural_spline_multi_feature_shape():
    X = np.random.rand(25, 2)
    transformer = NaturalCubicSplineTransformer(n_knots=6, include_bias=True)
    Xt = transformer.fit_transform(X)

    n_features = X.shape[1]
    n_basis_per_feature = transformer.designs_[0].shape[1]
    assert Xt.shape == (25, n_features * n_basis_per_feature)
    assert np.isfinite(Xt).all()


def test_natural_spline_output_consistency():
    X = np.random.rand(30, 2)
    transformer = NaturalCubicSplineTransformer(n_knots=4)
    transformer.fit(X)
    Xt1 = transformer.transform(X)
    Xt2 = transformer.fit_transform(X)

    np.testing.assert_allclose(Xt1, Xt2, rtol=1e-5)


def test_natural_spline_penalty_matrix_symmetry():
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    transformer = NaturalCubicSplineTransformer(n_knots=5)
    transformer.fit(X)
    P = transformer.get_penalty_matrix()

    assert P.shape[0] == P.shape[1]
    assert np.allclose(P, P.T, atol=1e-6)
