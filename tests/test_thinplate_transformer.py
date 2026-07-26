import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pretab.exceptions import InsufficientSamplesError, InvalidParamError
from pretab.transformers import ThinPlateSplineTransformer


def test_tprs_output_shape_and_values():
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    transformer = ThinPlateSplineTransformer(n_components=6, random_state=0)
    Xt = transformer.fit_transform(X)

    assert Xt.shape == (30, 6)
    assert transformer.total_output_dim_ == 6
    assert np.isfinite(Xt).all()


def test_tprs_output_consistency():
    X = np.random.rand(20, 1)
    transformer = ThinPlateSplineTransformer(n_components=5, random_state=0)
    transformer.fit(X)
    Xt1 = transformer.transform(X)
    Xt2 = transformer.fit_transform(X)

    np.testing.assert_allclose(Xt1, Xt2, rtol=1e-5)


def test_tprs_penalty_shape_and_symmetry():
    X = np.random.rand(25, 1)
    transformer = ThinPlateSplineTransformer(n_components=7, random_state=0)
    transformer.fit(X)
    P = transformer.get_penalty_matrix()

    assert P.shape[0] == P.shape[1]
    assert np.allclose(P, P.T, atol=1e-6)


def test_tprs_multivariate_is_supported():
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(60, 3))
    transformer = ThinPlateSplineTransformer(n_components=5, random_state=0)
    Xt = transformer.fit_transform(X)

    assert Xt.shape == (60, 5)
    assert transformer.n_features_in_ == 3
    assert np.isfinite(Xt).all()


def test_tprs_feature_count_mismatch_raises():
    rng = np.random.RandomState(0)
    transformer = ThinPlateSplineTransformer(n_components=4, random_state=0)
    transformer.fit(rng.uniform(size=(40, 1)))
    with pytest.raises(ValueError, match="is expecting 1 features"):
        transformer.transform(rng.uniform(size=(10, 2)))


def test_tprs_insufficient_samples_raises():
    X = np.random.rand(5, 2)
    transformer = ThinPlateSplineTransformer(n_components=10, random_state=0)
    with pytest.raises(InsufficientSamplesError, match="needs at least"):
        transformer.fit(X)


def test_tprs_rejects_invalid_strategies():
    X = np.random.rand(40, 1)
    with pytest.raises(InvalidParamError, match="landmark_strategy"):
        ThinPlateSplineTransformer(n_components=4, landmark_strategy="bogus").fit(X)
    with pytest.raises(InvalidParamError, match="rank_strategy"):
        ThinPlateSplineTransformer(n_components=4, rank_strategy="bogus").fit(X)


def test_tprs_nystroem_rank_strategy():
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(50, 2))
    transformer = ThinPlateSplineTransformer(n_components=6, rank_strategy="nystroem", random_state=0)
    Xt = transformer.fit_transform(X)

    assert Xt.shape == (50, 6)
    assert np.isfinite(Xt).all()


def test_tprs_feature_names_out():
    X = np.random.rand(20, 1)
    transformer = ThinPlateSplineTransformer(n_components=6, random_state=0)
    Xt = transformer.fit_transform(X)

    names = transformer.get_feature_names_out(["a"])
    assert len(names) == Xt.shape[1]
    assert names[0] == "a_tps0"
    assert all(name.startswith("a_tps") for name in names)


def test_tprs_feature_names_out_default_input():
    X = np.random.rand(15, 1)
    transformer = ThinPlateSplineTransformer(n_components=5, random_state=0).fit(X)

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
