"""Phase 3 · feature maps place centers with the shared location selectors.

The RBF / ReLU / sigmoid / tanh transformers now put their target-aware centers
at the split points chosen by a :class:`~pretab.core.selectors.CARTLocationSelector`
(default) or :class:`~pretab.core.selectors.LightGBMLocationSelector`, instead of
the old ``center_identification_using_decision_tree`` helper. These tests lock
that wiring: exact ``output_dim`` widths in fixed mode, clamping in adaptive mode,
the LightGBM option, selector validation, and equivalence with the underlying
count-based selector.
"""

from typing import cast

import numpy as np
import pytest
from sklearn.base import clone

from pretab.core.exceptions import InvalidParamError
from pretab.core.selectors import CARTLocationSelector
from pretab.transformers import (
    RBFExpansionTransformer,
    ReLUExpansionTransformer,
    SigmoidExpansionTransformer,
    TanhExpansionTransformer,
)

FEATURE_MAPS = [
    RBFExpansionTransformer,
    ReLUExpansionTransformer,
    SigmoidExpansionTransformer,
    TanhExpansionTransformer,
]


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x = np.linspace(0.1, 5.0, 300)
    X = x.reshape(-1, 1)
    y = (x > 2.5).astype(float) + rng.normal(0, 0.05, x.size)
    return X, y


@pytest.mark.parametrize("Cls", FEATURE_MAPS)
def test_non_adaptive_target_path_gives_exact_output_dim(Cls, data):
    X, y = data
    t = Cls(output_dim=6, use_target=True, task="regression").fit(X, y)
    assert all(len(c) == 6 for c in t.centers_)


@pytest.mark.parametrize("Cls", FEATURE_MAPS)
def test_adaptive_target_path_clamps_within_window(Cls, data):
    X, y = data
    t = Cls(output_dim=8, use_target=True, adaptive=True, min_output_dim=3, max_output_dim=6).fit(X, y)
    assert all(3 <= len(c) <= 6 for c in t.centers_)


@pytest.mark.parametrize("Cls", FEATURE_MAPS)
def test_centers_match_cart_location_selector(Cls, data):
    X, y = data
    t = Cls(output_dim=5, use_target=True, task="regression").fit(X, y)
    expected = CARTLocationSelector().select(
        X[:, 0], y, task="regression", min_count=5, max_count=5
    )
    np.testing.assert_array_equal(t.centers_[0], expected)


def test_default_selector_center_snapshot(data):
    """Lock the exact CART center placement for a fixed dataset."""
    X, y = data
    t = RBFExpansionTransformer(output_dim=5, use_target=True, task="regression").fit(X, y)
    np.testing.assert_allclose(
        np.round(t.centers_[0], 5),
        [0.76371, 0.92759, 2.50084, 2.71388, 3.82826],
    )


def test_invalid_selector_raises(data):
    X, y = data
    with pytest.raises(InvalidParamError, match="Invalid selector"):
        RBFExpansionTransformer(use_target=True, selector="bogus").fit(X, y)


def test_quantile_path_ignores_selector(data):
    """The non-target quantile path is untouched: exactly ``output_dim`` centers."""
    X, _ = data
    t = RBFExpansionTransformer(
        output_dim=7, use_target=False, strategy="quantile", selector="lightgbm"
    ).fit(X)
    assert all(len(c) == 7 for c in t.centers_)


def test_selector_is_a_clonable_hyperparameter():
    t = RBFExpansionTransformer(selector="lightgbm")
    assert t.get_params()["selector"] == "lightgbm"
    cloned = cast(RBFExpansionTransformer, clone(t))
    assert cloned.selector == "lightgbm"


@pytest.mark.parametrize("Cls", FEATURE_MAPS)
def test_lightgbm_selector_places_centers(Cls, data):
    pytest.importorskip("lightgbm")
    X, y = data
    t = Cls(output_dim=5, use_target=True, task="regression", selector="lightgbm").fit(X, y)
    assert all(len(c) == 5 for c in t.centers_)
    assert all(np.all(np.diff(c) > 0) for c in t.centers_)
