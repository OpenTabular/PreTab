import numpy as np
import pytest
import warnings
from sklearn.utils.validation import check_is_fitted
from pretab.transformers import ReLUExpansionTransformer


@pytest.fixture
def X_single_feature():
    return np.linspace(0, 1, 10).reshape(-1, 1)


@pytest.fixture
def X_multi_feature():
    return np.random.rand(15, 3)


@pytest.fixture
def y_regression():
    return np.random.rand(15)


def test_relu_uniform_single_feature(X_single_feature):
    transformer = ReLUExpansionTransformer(
        n_centers=4, use_decision_tree=False, strategy="uniform"
    )
    transformer.fit(X_single_feature)
    Xt = transformer.transform(X_single_feature)

    assert Xt.shape == (10, 4)
    assert (Xt >= 0).all()


def test_relu_quantile_multi_feature(X_multi_feature):
    transformer = ReLUExpansionTransformer(
        n_centers=5, use_decision_tree=False, strategy="quantile"
    )
    transformer.fit(X_multi_feature)
    Xt = transformer.transform(X_multi_feature)

    assert Xt.shape == (15, 5 * 3)
    assert (Xt >= 0).all()


def test_relu_tree_centering(X_multi_feature, y_regression):
    transformer = ReLUExpansionTransformer(n_centers=3, use_decision_tree=True)
    transformer.fit(X_multi_feature, y_regression)
    Xt = transformer.transform(X_multi_feature)

    assert Xt.shape == (15, 3 * 3)
    assert (Xt >= 0).all()


def test_relu_invalid_strategy():
    with pytest.raises(ValueError, match="Invalid strategy"):
        ReLUExpansionTransformer(strategy="nonsense")


def test_relu_invalid_task():
    with pytest.raises(ValueError, match="Invalid task"):
        ReLUExpansionTransformer(task="nonsense")


def test_relu_missing_y_tree(X_single_feature):
    transformer = ReLUExpansionTransformer(use_decision_tree=True)
    with pytest.raises(ValueError, match="Target variable.*must be provided"):
        transformer.fit(X_single_feature)


def test_relu_feature_mismatch(X_multi_feature, y_regression):
    transformer = ReLUExpansionTransformer(n_centers=3, use_decision_tree=True)
    transformer.fit(X_multi_feature[:, :2], y_regression)
    with pytest.raises(ValueError, match="same number of features"):
        transformer.transform(X_multi_feature)
