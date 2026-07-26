import warnings

import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted

from pretab.transformers import SigmoidExpansionTransformer


@pytest.fixture
def X_single_feature():
    return np.linspace(-1, 1, 10).reshape(-1, 1)


@pytest.fixture
def X_multi_feature():
    return np.random.randn(20, 3)


@pytest.fixture
def y_regression():
    return np.random.randn(20)


def test_sigmoid_uniform_single_feature(X_single_feature):
    transformer = SigmoidExpansionTransformer(output_dim=4, target_aware=False, placement_strategy="uniform", scale=0.5)
    transformer.fit(X_single_feature)
    Xt = transformer.transform(X_single_feature)

    assert Xt.shape == (10, 4)
    assert (Xt >= 0).all()
    assert (Xt <= 1).all()


def test_sigmoid_quantile_multi_feature(X_multi_feature):
    transformer = SigmoidExpansionTransformer(output_dim=5, target_aware=False, placement_strategy="quantile")
    transformer.fit(X_multi_feature)
    Xt = transformer.transform(X_multi_feature)

    assert Xt.shape == (20, 5 * 3)
    assert (Xt >= 0).all()
    assert (Xt <= 1).all()


def test_sigmoid_tree_centering(X_multi_feature, y_regression):
    transformer = SigmoidExpansionTransformer(output_dim=3, target_aware=True)
    transformer.fit(X_multi_feature, y_regression)
    Xt = transformer.transform(X_multi_feature)

    assert Xt.shape == (20, 3 * 3)
    assert (Xt >= 0).all()
    assert (Xt <= 1).all()


def test_sigmoid_invalid_strategy():
    with pytest.raises(ValueError, match="placement_strategy must be 'uniform' or 'quantile'"):
        SigmoidExpansionTransformer(target_aware=False, placement_strategy="nonsense").fit(np.random.rand(5, 1))


def test_sigmoid_invalid_task():
    with pytest.raises(ValueError, match="Invalid task"):
        SigmoidExpansionTransformer(task="nonsense").fit(np.random.rand(5, 1))


def test_sigmoid_missing_y_tree(X_single_feature):
    transformer = SigmoidExpansionTransformer(target_aware=True)
    with pytest.raises(ValueError, match=r"Target variable.*must be provided"):
        transformer.fit(X_single_feature)


def test_sigmoid_feature_mismatch(X_multi_feature, y_regression):
    transformer = SigmoidExpansionTransformer(output_dim=2, target_aware=True)
    transformer.fit(X_multi_feature[:, :2], y_regression)
    with pytest.raises(ValueError, match="is expecting"):
        transformer.transform(X_multi_feature)


def test_sigmoid_no_overflow_on_large_inputs():
    # Large-magnitude values used to trigger "overflow encountered in exp".
    transformer = SigmoidExpansionTransformer(output_dim=4, target_aware=False, placement_strategy="uniform")
    transformer.fit(np.linspace(-1, 1, 10).reshape(-1, 1))
    X_extreme = np.array([[-1000.0], [1000.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning becomes an error
        Xt = transformer.transform(X_extreme)
    assert np.isfinite(Xt).all()
    assert (Xt >= 0).all()
    assert (Xt <= 1).all()
