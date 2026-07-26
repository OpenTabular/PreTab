"""Tests for the leakage-safe supervised contract (Phase 7, P7.1 + P7.2).

Covers the transformer contract properties (``requires_y`` / ``is_supervised`` /
``uses_target_``) and the :class:`~pretab.exceptions.LeakageWarning` emitted when
a supervised transformer is fit on ``(X, y)`` outside a controlled context.
"""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from pretab import LeakageWarning, Preprocessor
from pretab.compose.registry import get_spec
from pretab.core.supervised import in_controlled_context, warn_target_leakage
from pretab.transformers import (
    BSplineTransformer,
    PLETransformer,
    RBFExpansionTransformer,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 1))
    y = (X[:, 0] > 0).astype(float) + rng.normal(scale=0.1, size=200)
    return X, y


# --- P7.1: contract properties ---------------------------------------------


def test_ple_is_always_supervised(data):
    X, y = data
    ple = PLETransformer()
    assert ple.requires_y is True
    assert ple.is_supervised is True
    ple.fit(X, y)
    assert ple.uses_target_ is True


def test_unsupervised_spline_reports_not_supervised(data):
    X, _ = data
    spline = BSplineTransformer(target_aware=False)
    assert spline.requires_y is False
    assert spline.is_supervised is False
    spline.fit(X)
    assert spline.uses_target_ is False


def test_optional_transformer_flips_with_target_aware(data):
    X, y = data
    rbf_off = RBFExpansionTransformer(target_aware=False)
    assert rbf_off.requires_y is False
    assert rbf_off.is_supervised is False

    rbf_on = RBFExpansionTransformer(target_aware=True)
    assert rbf_on.requires_y is False
    assert rbf_on.is_supervised is True
    rbf_on.fit(X, y)
    assert rbf_on.uses_target_ is True


def test_registry_supervised_flags():
    assert get_spec("ple").requires_y is True
    assert get_spec("ple").is_supervised is True
    assert get_spec("rbf").requires_y is False
    assert get_spec("rbf").is_supervised is True
    assert get_spec("standardization").is_supervised is False


# --- P7.2: leakage warning --------------------------------------------------


def test_direct_supervised_fit_warns(data):
    X, y = data
    with pytest.warns(LeakageWarning):
        PLETransformer().fit(X, y)


def test_target_aware_spline_direct_fit_warns(data):
    X, y = data
    with pytest.warns(LeakageWarning):
        BSplineTransformer(target_aware=True, placement_strategy="cart").fit(X, y)


def test_unsupervised_fit_does_not_warn(data):
    X, _ = data
    with warnings.catch_warnings():
        warnings.simplefilter("error", LeakageWarning)
        BSplineTransformer(target_aware=False).fit(X)


def test_no_warning_without_target(data):
    rbf = RBFExpansionTransformer(target_aware=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LeakageWarning)
        warn_target_leakage(rbf, None)


def test_no_warning_inside_pipeline(data):
    X, y = data
    pipe = Pipeline([("ple", PLETransformer()), ("ridge", Ridge())])
    with warnings.catch_warnings():
        warnings.simplefilter("error", LeakageWarning)
        pipe.fit(X, y)


def test_no_warning_inside_preprocessor(data):
    X, y = data
    pre = Preprocessor(numerical_method="ple", target_aware=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LeakageWarning)
        pre.fit_transform(X, y)


def test_in_controlled_context_default_false():
    assert in_controlled_context() is False


def test_warn_helper_ignores_unsupervised(data):
    _, y = data
    est = BSplineTransformer(target_aware=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LeakageWarning)
        warn_target_leakage(est, y)
