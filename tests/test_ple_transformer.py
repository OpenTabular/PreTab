import warnings

import numpy as np
import pytest

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
    with pytest.raises(ValueError, match="Unsupported task"):
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


def test_ple_thresholds_extracted_from_tree():
    rng = np.random.RandomState(0)
    X = rng.rand(50, 1)
    y = rng.rand(50)

    transformer = PLETransformer(n_bins=5)
    transformer.fit(X, y)

    # Thresholds are read straight from the fitted tree and are sorted/unique.
    thresholds = transformer.thresholds_[0]
    assert thresholds.ndim == 1
    assert np.all(np.diff(thresholds) > 0)
    assert transformer.n_bins_per_feature_[0] == len(thresholds) + 1


def test_ple_is_reproducible():
    rng = np.random.RandomState(7)
    X = rng.rand(40, 2)
    y = rng.rand(40)

    a = PLETransformer(n_bins=5).fit_transform(X, y)
    b = PLETransformer(n_bins=5).fit_transform(X, y)

    np.testing.assert_array_equal(a, b)


def test_ple_handles_nan_with_median():
    rng = np.random.RandomState(1)
    X = rng.rand(30, 1)
    y = rng.rand(30)

    transformer = PLETransformer(n_bins=5, handle_missing="median")
    transformer.fit(X, y)

    X_missing = X.copy()
    X_missing[0, 0] = np.nan
    Xt = transformer.transform(X_missing)

    assert np.isfinite(Xt).all()


def test_ple_raises_on_nan_when_configured():
    rng = np.random.RandomState(2)
    X = rng.rand(30, 1)
    y = rng.rand(30)

    transformer = PLETransformer(n_bins=5, handle_missing="error")
    transformer.fit(X, y)

    X_missing = X.copy()
    X_missing[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        transformer.transform(X_missing)


def test_ple_feature_names_out():
    rng = np.random.RandomState(3)
    X = rng.rand(30, 2)
    y = rng.rand(30)

    transformer = PLETransformer(n_bins=4)
    transformer.fit(X, y)

    names = transformer.get_feature_names_out(["age", "income"])
    assert len(names) == transformer.get_n_features_out()
    assert all("_ple_piece" in name for name in names)
    assert names[0].startswith("age")
