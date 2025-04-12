import numpy as np
import pytest
import warnings
from pretab.transformers import PLETransformer


@pytest.fixture
def X_single_feature():
    return np.linspace(0, 1, 10).reshape(-1, 1)


@pytest.fixture
def X_multi_feature():
    return np.random.rand(12, 2)


@pytest.fixture
def y_regression():
    return np.random.rand(12)


def test_ple_transformer_single_feature_shape(X_single_feature):
    y = np.linspace(0, 1, 10)
    n_bins = 4
    transformer = PLETransformer(n_bins=n_bins)
    transformer.fit(X_single_feature, y)
    Xt = transformer.transform(X_single_feature)

    # Each feature → output should have n_bins - 1 + 1 = n_bins columns
    assert Xt.shape == (X_single_feature.shape[0], n_bins)
    assert np.isfinite(Xt).all()
    assert (Xt >= 0).all()


def test_ple_transformer_multi_feature_shape(X_multi_feature, y_regression):
    n_bins = 5
    transformer = PLETransformer(n_bins=n_bins)
    transformer.fit(X_multi_feature, y_regression)
    Xt = transformer.transform(X_multi_feature)

    # Each feature → n_bins columns, 2 features → 2 * n_bins
    assert Xt.shape == (X_multi_feature.shape[0], 2 * n_bins)
    assert np.isfinite(Xt).all()
    assert (Xt >= 0).all()


def test_ple_invalid_task_raises(X_single_feature):
    with pytest.raises(ValueError, match="not supported"):
        transformer = PLETransformer(task="unsupported")
        transformer.fit(X_single_feature, np.linspace(0, 1, 10))


def test_ple_exact_bin_dimension_single_feature():
    X = np.random.randn(20, 1)
    y = np.random.randn(20, 1)
    n_bins = 6
    transformer = PLETransformer(n_bins=n_bins)
    transformer.fit(X, y)
    Xt = transformer.transform(X)

    assert Xt.shape[1] == n_bins


def test_ple_exact_bin_dimension_multi_feature():
    # 20 samples, 2 features
    rng = np.random.RandomState(42)
    X = rng.rand(20, 2)
    y = rng.randint(0, 2, size=20)
    n_bins = 6

    transformer = PLETransformer(n_bins=n_bins)
    transformer.fit(X, y)
    Xt = transformer.transform(X)

    assert Xt.shape == (20, 2 * n_bins)
