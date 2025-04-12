import numpy as np
import pytest
import warnings
from pretab.transformers import ThinPlateSplineTransformer


def test_tprs_output_shape_and_values():
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    transformer = ThinPlateSplineTransformer(n_basis=6)
    Xt = transformer.fit_transform(X)

    assert Xt.shape == (30, 6)
    assert np.isfinite(Xt).all()


def test_tprs_output_consistency():
    X = np.random.rand(20, 1)
    transformer = ThinPlateSplineTransformer(n_basis=5)
    transformer.fit(X)
    Xt1 = transformer.transform(X)
    Xt2 = transformer.fit_transform(X)

    np.testing.assert_allclose(Xt1, Xt2, rtol=1e-5)


def test_tprs_penalty_shape_and_symmetry():
    X = np.random.rand(25, 1)
    transformer = ThinPlateSplineTransformer(n_basis=7)
    transformer.fit(X)
    P = transformer.get_penalty_matrix()

    assert P.shape[0] == P.shape[1]
    assert np.allclose(P, P.T, atol=1e-6)


def test_tprs_multivariate_error():
    X = np.random.rand(10, 2)
    transformer = ThinPlateSplineTransformer(n_basis=4)
    with pytest.raises(ValueError, match="univariate"):
        transformer.fit(X)

    transformer = ThinPlateSplineTransformer(n_basis=4)
    transformer.fit(np.random.rand(10, 1))
    with pytest.raises(ValueError, match="univariate"):
        transformer.transform(X)
