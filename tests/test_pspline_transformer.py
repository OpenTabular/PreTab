import numpy as np
import pytest
import warnings
from pretab.transformers import PSplineTransformer


def test_pspline_single_feature_shape():
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    transformer = PSplineTransformer(n_knots=6)
    Xt = transformer.fit_transform(X)

    n_basis = transformer.n_basis_[0]
    assert Xt.shape == (30, n_basis)
    assert np.isfinite(Xt).all()


def test_pspline_multi_feature_shape():
    X = np.random.rand(25, 2)
    transformer = PSplineTransformer(n_knots=5)
    Xt = transformer.fit_transform(X)

    total_basis = sum(transformer.n_basis_)
    assert Xt.shape == (25, total_basis)
    assert np.isfinite(Xt).all()


def test_pspline_output_consistency():
    X = np.random.rand(20, 2)
    transformer = PSplineTransformer(n_knots=4)
    transformer.fit(X)
    Xt1 = transformer.transform(X)
    Xt2 = transformer.fit_transform(X)

    np.testing.assert_allclose(Xt1, Xt2, rtol=1e-5)


def test_pspline_penalty_matrix_shape_and_symmetry():
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    transformer = PSplineTransformer(n_knots=6)
    transformer.fit(X)
    P = transformer.get_penalty_matrix()

    assert P.shape[0] == P.shape[1]
    assert np.allclose(P, P.T, atol=1e-6)
