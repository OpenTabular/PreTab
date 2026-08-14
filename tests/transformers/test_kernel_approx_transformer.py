import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pretab.exceptions import InvalidParamError
from pretab.transformers import (
    NystroemFeaturesTransformer,
    RandomFourierFeaturesTransformer,
)


@pytest.fixture
def X():
    return np.random.default_rng(0).uniform(size=(60, 3))


# --------------------------------------------------------------------------- #
# Random Fourier features (RBFSampler wrapper).
# --------------------------------------------------------------------------- #
def test_rff_output_shape_and_total_dim(X):
    transformer = RandomFourierFeaturesTransformer(n_components=20, random_state=0)
    Xt = transformer.fit_transform(X)

    assert Xt.shape == (60, 20)
    assert transformer.total_output_dim_ == 20
    assert transformer.n_features_in_ == 3
    assert np.isfinite(Xt).all()


def test_rff_is_deterministic_with_random_state(X):
    transformer = RandomFourierFeaturesTransformer(n_components=16, random_state=0)
    Xt1 = transformer.fit(X).transform(X)
    Xt2 = RandomFourierFeaturesTransformer(n_components=16, random_state=0).fit_transform(X)

    np.testing.assert_allclose(Xt1, Xt2)


def test_rff_rejects_invalid_n_components(X):
    with pytest.raises(InvalidParamError, match="n_components"):
        RandomFourierFeaturesTransformer(n_components=0).fit(X)


def test_rff_feature_names_out(X):
    transformer = RandomFourierFeaturesTransformer(n_components=5, random_state=0).fit(X)
    names = transformer.get_feature_names_out()

    assert len(names) == 5
    assert names[0].startswith("x0_rff")


def test_rff_transform_requires_fit(X):
    with pytest.raises(NotFittedError):
        RandomFourierFeaturesTransformer().transform(X)


# --------------------------------------------------------------------------- #
# Nystroem features (Nystroem wrapper).
# --------------------------------------------------------------------------- #
def test_nystroem_output_shape_and_total_dim(X):
    transformer = NystroemFeaturesTransformer(n_components=15, random_state=0)
    Xt = transformer.fit_transform(X)

    assert Xt.shape == (60, 15)
    assert transformer.total_output_dim_ == 15
    assert np.isfinite(Xt).all()


def test_nystroem_is_deterministic_with_random_state(X):
    transformer = NystroemFeaturesTransformer(n_components=12, random_state=0)
    Xt1 = transformer.fit(X).transform(X)
    Xt2 = NystroemFeaturesTransformer(n_components=12, random_state=0).fit_transform(X)

    np.testing.assert_allclose(Xt1, Xt2)


def test_nystroem_supports_non_default_kernel(X):
    transformer = NystroemFeaturesTransformer(n_components=10, kernel="laplacian", random_state=0)
    Xt = transformer.fit_transform(X)

    assert Xt.shape == (60, 10)
    assert np.isfinite(Xt).all()


def test_nystroem_rejects_invalid_params(X):
    with pytest.raises(InvalidParamError, match="n_components"):
        NystroemFeaturesTransformer(n_components=0).fit(X)
    with pytest.raises(InvalidParamError, match="kernel"):
        NystroemFeaturesTransformer(kernel="bogus").fit(X)


def test_nystroem_feature_names_out(X):
    transformer = NystroemFeaturesTransformer(n_components=8, random_state=0).fit(X)
    names = transformer.get_feature_names_out()

    assert len(names) == 8
    assert names[0].startswith("x0_nystroem")


def test_nystroem_transform_requires_fit(X):
    with pytest.raises(NotFittedError):
        NystroemFeaturesTransformer().transform(X)
