import numpy as np
import pytest
import warnings
from pretab.transformers import TensorProductSplineTransformer


def test_tensorproduct_spline_output_shape():
    X = np.random.rand(20, 2)
    transformer = TensorProductSplineTransformer(n_knots=4)
    Xt = transformer.fit_transform(X)

    n_basis_0 = transformer.bases_[0].shape[1]
    n_basis_1 = transformer.bases_[1].shape[1]
    assert Xt.shape == (20, n_basis_0 * n_basis_1)
    assert np.isfinite(Xt).all()


def test_tensorproduct_spline_output_consistency():
    X = np.random.rand(30, 2)
    transformer = TensorProductSplineTransformer(n_knots=5)
    transformer.fit(X)
    Xt1 = transformer.transform(X)
    Xt2 = transformer.fit_transform(X)

    np.testing.assert_allclose(Xt1, Xt2, rtol=1e-5)


def test_tensorproduct_spline_penalty_matrices():
    X = np.random.rand(25, 2)
    transformer = TensorProductSplineTransformer(n_knots=4)
    transformer.fit(X)
    penalties = transformer.get_penalty_matrices()

    assert len(penalties) == 2
    for P in penalties:
        assert P.shape[0] == P.shape[1]
        assert np.allclose(P, P.T, atol=1e-6)
