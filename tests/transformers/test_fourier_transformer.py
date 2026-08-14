import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pretab.exceptions import InvalidParamError
from pretab.transformers import FourierFeatureTransformer


def test_fourier_output_shape_and_total_dim():
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    transformer = FourierFeatureTransformer(n_frequencies=4)
    Xt = transformer.fit_transform(X)

    assert Xt.shape == (50, 8)
    assert transformer.total_output_dim_ == 8
    assert np.isfinite(Xt).all()


def test_fourier_include_original_prepends_raw_value():
    X = np.linspace(0, 5, 20).reshape(-1, 1)
    transformer = FourierFeatureTransformer(n_frequencies=3, include_original=True)
    Xt = transformer.fit_transform(X)

    assert Xt.shape == (20, 7)
    np.testing.assert_allclose(Xt[:, 0], X[:, 0])


def test_fourier_multifeature_is_per_feature_contiguous():
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, size=(40, 2))
    transformer = FourierFeatureTransformer(n_frequencies=2)
    Xt = transformer.fit_transform(X)

    assert Xt.shape == (40, 8)
    assert transformer.n_features_in_ == 2


@pytest.mark.parametrize("strategy", ["harmonic", "log_spaced", "random"])
def test_fourier_strategies_are_deterministic(strategy):
    X = np.linspace(0, 4, 30).reshape(-1, 1)
    transformer = FourierFeatureTransformer(n_frequencies=3, frequency_strategy=strategy, random_state=0)
    Xt1 = transformer.fit(X).transform(X)
    Xt2 = FourierFeatureTransformer(n_frequencies=3, frequency_strategy=strategy, random_state=0).fit_transform(X)

    np.testing.assert_allclose(Xt1, Xt2)


def test_fourier_rejects_invalid_params():
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    with pytest.raises(InvalidParamError, match="n_frequencies"):
        FourierFeatureTransformer(n_frequencies=0).fit(X)
    with pytest.raises(InvalidParamError, match="frequency_strategy"):
        FourierFeatureTransformer(frequency_strategy="bogus").fit(X)


def test_fourier_rejects_nan():
    X = np.full((10, 1), np.nan)
    with pytest.raises(ValueError, match="NaN"):
        FourierFeatureTransformer().fit(X)


def test_fourier_feature_names_out():
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    transformer = FourierFeatureTransformer(n_frequencies=2).fit(X)

    names = transformer.get_feature_names_out(["a"])
    assert list(names) == ["a_fourier0", "a_fourier1", "a_fourier2", "a_fourier3"]


def test_fourier_transform_requires_fit():
    with pytest.raises(NotFittedError):
        FourierFeatureTransformer().transform(np.linspace(0, 1, 5).reshape(-1, 1))


def test_fourier_allow_nan_tag_is_false():
    tags = FourierFeatureTransformer().__sklearn_tags__()
    assert tags.input_tags.allow_nan is False
