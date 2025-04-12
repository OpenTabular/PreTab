import numpy as np
import pytest
import warnings
from pretab.transformers import TanhExpansionTransformer


@pytest.fixture
def X_float_1d():
    return np.linspace(0, 1, 20).reshape(-1, 1)


@pytest.fixture
def X_float_2d():
    return np.random.rand(30, 3)


@pytest.fixture
def y_dummy():
    return np.random.rand(30)


def test_tanh_transformer_output_shape(X_float_2d, y_dummy):
    transformer = TanhExpansionTransformer(n_centers=5)
    Xt = transformer.fit_transform(X_float_2d, y_dummy)
    assert Xt.shape == (X_float_2d.shape[0], X_float_2d.shape[1] * 5)
    assert np.isfinite(Xt).all()
    assert (Xt >= -1).all() and (Xt <= 1).all()


def test_tanh_transformer_single_feature(X_float_1d):
    transformer = TanhExpansionTransformer(n_centers=6, use_decision_tree=False)
    Xt = transformer.fit_transform(X_float_1d)
    assert Xt.shape == (X_float_1d.shape[0], 6)
    assert np.isfinite(Xt).all()


def test_tanh_transformer_output_consistency(X_float_2d, y_dummy):
    transformer = TanhExpansionTransformer(n_centers=4)
    transformer.fit(X_float_2d, y_dummy)
    Xt1 = transformer.transform(X_float_2d)
    Xt2 = transformer.fit_transform(X_float_2d, y_dummy)
    np.testing.assert_allclose(Xt1, Xt2, rtol=1e-5)


def test_tanh_invalid_strategy_raises():
    with pytest.raises(ValueError, match="Invalid strategy"):
        TanhExpansionTransformer(strategy="invalid")


def test_tanh_invalid_task_raises():
    with pytest.raises(ValueError, match="Invalid task"):
        TanhExpansionTransformer(task="invalid")
