"""Phase 4 · PLE places bin thresholds with the shared location selectors.

``PLETransformer`` now reads its bin edges from the split points chosen by a
:class:`~pretab.core.selectors.CARTLocationSelector` (default) or
:class:`~pretab.core.selectors.LightGBMLocationSelector`, instead of its old
inline decision tree. These tests lock that wiring: the bin-count contract
(exactly ``output_dim`` bins in fixed mode, clamped to ``[min, max]`` when
adaptive), the LightGBM option, selector validation, and equivalence with the
underlying count-based selector.
"""

from typing import cast

import numpy as np
import pytest
from sklearn.base import clone

from pretab.core.exceptions import InvalidParamError
from pretab.core.selectors import CARTLocationSelector
from pretab.transformers import PLETransformer


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x = np.linspace(0.1, 5.0, 300)
    X = x.reshape(-1, 1)
    y = (x > 2.5).astype(float) + rng.normal(0, 0.05, x.size)
    return X, y


def test_non_adaptive_target_path_gives_exact_output_dim(data):
    """Fixed mode pins the count to exactly ``output_dim`` bins per feature."""
    X, y = data
    t = PLETransformer(output_dim=6, task="regression").fit(X, y)
    assert t.n_bins_per_feature_ == [6]
    assert len(t.thresholds_[0]) == 5


def test_adaptive_target_path_clamps_within_window(data):
    X, y = data
    t = PLETransformer(output_dim=10, adaptive=True, min_output_dim=3, max_output_dim=6).fit(X, y)
    assert all(3 <= n <= 6 for n in t.n_bins_per_feature_)
    assert all(2 <= len(th) <= 5 for th in t.thresholds_)


def test_thresholds_match_cart_location_selector(data):
    X, y = data
    t = PLETransformer(output_dim=5, task="regression").fit(X, y)
    expected = CARTLocationSelector().select(X[:, 0], y, task="regression", min_count=4, max_count=4)
    np.testing.assert_array_equal(t.thresholds_[0], expected)


def test_default_selector_threshold_snapshot(data):
    """Lock the exact CART threshold placement for a fixed dataset."""
    X, y = data
    t = PLETransformer(output_dim=5, task="regression").fit(X, y)
    np.testing.assert_allclose(
        np.round(t.thresholds_[0], 5),
        [0.76371, 2.50084, 2.71388, 3.82826],
    )


def test_thresholds_are_sorted_and_unique(data):
    X, y = data
    t = PLETransformer(output_dim=6).fit(X, y)
    for th in t.thresholds_:
        assert np.all(np.diff(th) > 0)


def test_invalid_selector_raises(data):
    X, y = data
    with pytest.raises(InvalidParamError, match="Invalid placement_strategy"):
        PLETransformer(placement_strategy="bogus").fit(X, y)


def test_selector_is_a_clonable_hyperparameter():
    t = PLETransformer(placement_strategy="lightgbm")
    assert t.get_params()["placement_strategy"] == "lightgbm"
    cloned = cast(PLETransformer, clone(t))
    assert cloned.placement_strategy == "lightgbm"


def test_classification_task_places_thresholds(data):
    X, _ = data
    y = (X[:, 0] > 2.5).astype(int)
    t = PLETransformer(output_dim=5, task="classification").fit(X, y)
    assert t.n_bins_per_feature_ == [5]
    assert np.all(np.diff(t.thresholds_[0]) > 0)


def test_lightgbm_selector_places_thresholds(data):
    pytest.importorskip("lightgbm")
    X, y = data
    t = PLETransformer(output_dim=5, task="regression", placement_strategy="lightgbm").fit(X, y)
    assert t.n_bins_per_feature_ == [5]
    assert np.all(np.diff(t.thresholds_[0]) > 0)
