import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pretab.transformers import ThinPlateSplineTransformer


def test_tprs_output_shape_and_values():
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    transformer = ThinPlateSplineTransformer(output_dim=6)
    Xt = transformer.fit_transform(X)

    assert Xt.shape == (30, 6)
    assert transformer.total_output_dim_ == 6
    assert np.isfinite(Xt).all()


def test_tprs_output_consistency():
    X = np.random.rand(20, 1)
    transformer = ThinPlateSplineTransformer(output_dim=5)
    transformer.fit(X)
    Xt1 = transformer.transform(X)
    Xt2 = transformer.fit_transform(X)

    np.testing.assert_allclose(Xt1, Xt2, rtol=1e-5)


def test_tprs_penalty_shape_and_symmetry():
    X = np.random.rand(25, 1)
    transformer = ThinPlateSplineTransformer(output_dim=7)
    transformer.fit(X)
    P = transformer.get_penalty_matrix()

    assert P.shape[0] == P.shape[1]
    assert np.allclose(P, P.T, atol=1e-6)


def test_tprs_multivariate_error():
    X = np.random.rand(10, 2)
    transformer = ThinPlateSplineTransformer(output_dim=4)
    with pytest.raises(ValueError, match="univariate"):
        transformer.fit(X)

    transformer = ThinPlateSplineTransformer(output_dim=4)
    transformer.fit(np.random.rand(10, 1))
    with pytest.raises(ValueError, match="univariate"):
        transformer.transform(X)


def test_tprs_feature_names_out():
    X = np.random.rand(20, 1)
    transformer = ThinPlateSplineTransformer(output_dim=6)
    Xt = transformer.fit_transform(X)

    names = transformer.get_feature_names_out(["a"])
    assert len(names) == Xt.shape[1]
    assert names[0] == "a_tps0"
    assert all(name.startswith("a_tps") for name in names)


def test_tprs_feature_names_out_default_input():
    X = np.random.rand(15, 1)
    transformer = ThinPlateSplineTransformer(output_dim=5).fit(X)

    names = transformer.get_feature_names_out()
    assert len(names) == transformer.n_basis_[0]
    assert names[0].startswith("x0_tps")


def test_tprs_allow_nan_tag():
    tags = ThinPlateSplineTransformer().__sklearn_tags__()
    assert tags.input_tags.allow_nan is True


def test_tprs_transform_requires_fit():
    transformer = ThinPlateSplineTransformer()
    with pytest.raises(NotFittedError):
        transformer.transform(np.random.rand(5, 1))
    with pytest.raises(NotFittedError):
        transformer.get_penalty_matrix()
