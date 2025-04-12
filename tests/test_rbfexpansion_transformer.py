import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted
from pretab.transformers import RBFExpansionTransformer


@pytest.fixture
def X_single_feature():
    return np.linspace(0, 1, 10).reshape(-1, 1)


@pytest.fixture
def X_multi_feature():
    return np.random.rand(15, 3)


@pytest.fixture
def y_regression():
    return np.random.rand(15)


def test_rbf_uniform_single_feature(X_single_feature):
    transformer = RBFExpansionTransformer(
        n_centers=5, strategy="uniform", use_decision_tree=False
    )
    transformer.fit(X_single_feature)
    Xt = transformer.transform(X_single_feature)

    assert Xt.shape == (X_single_feature.shape[0], 5)
    assert Xt.dtype == np.float64
    assert np.all(Xt >= 0) and np.all(Xt <= 1)


def test_rbf_quantile_multi_feature(X_multi_feature):
    transformer = RBFExpansionTransformer(
        n_centers=4, strategy="quantile", use_decision_tree=False
    )
    transformer.fit(X_multi_feature)
    Xt = transformer.transform(X_multi_feature)

    assert Xt.shape == (X_multi_feature.shape[0], 4 * X_multi_feature.shape[1])
    assert Xt.dtype == np.float64
    assert np.all(Xt >= 0) and np.all(Xt <= 1)


def test_rbf_decision_tree_centers(X_multi_feature, y_regression):
    transformer = RBFExpansionTransformer(n_centers=3, use_decision_tree=True)
    transformer.fit(X_multi_feature, y_regression)
    check_is_fitted(transformer)
    Xt = transformer.transform(X_multi_feature)

    assert Xt.shape == (X_multi_feature.shape[0], 3 * X_multi_feature.shape[1])


def test_rbf_invalid_strategy():
    with pytest.raises(ValueError, match="Invalid strategy"):
        RBFExpansionTransformer(strategy="invalid").fit(np.random.rand(5, 1))


def test_rbf_invalid_task():
    with pytest.raises(ValueError, match="Invalid task"):
        RBFExpansionTransformer(task="invalid").fit(np.random.rand(5, 1))


def test_rbf_missing_target_with_tree(X_single_feature):
    transformer = RBFExpansionTransformer(use_decision_tree=True)
    with pytest.raises(ValueError, match="Target variable.*must be provided"):
        transformer.fit(X_single_feature)


def test_rbf_transform_before_fit_raises():
    transformer = RBFExpansionTransformer(n_centers=3)
    X = np.random.rand(5, 2)
    with pytest.raises(AttributeError):
        transformer.transform(X)


def test_rbf_transform_feature_mismatch(X_multi_feature, y_regression):
    transformer = RBFExpansionTransformer(n_centers=3, use_decision_tree=True)
    transformer.fit(X_multi_feature[:, :2], y_regression)
    with pytest.raises(ValueError, match="same number of features"):
        transformer.transform(X_multi_feature)
