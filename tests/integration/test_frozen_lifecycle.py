"""Tests for the immutable lifecycle: freeze / stale / clone / refit (P9.3).

Covers the ``lifecycle_state_`` transitions, ``freeze`` / ``is_frozen``,
``set_params`` / ``set_output`` rejection on frozen instances, output no-ops
and global configuration, ``clone_unfitted`` and ``refit`` returning fresh
objects, and ``mark_stale``.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn import config_context
from sklearn.pipeline import Pipeline

from pretab import FrozenRepresentationError, Preprocessor


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.random(40), "c": rng.choice(["x", "y"], 40)})


@pytest.fixture
def target():
    return np.random.default_rng(1).random(40)


def _make():
    return Preprocessor(numerical_method="rbf", target_aware=False, placement_strategy="quantile")


def test_unfitted_state(frame, target):
    p = _make()
    assert p.lifecycle_state_ == "UNFITTED"
    assert p.is_frozen() is False


def test_fitted_state(frame, target):
    p = _make().fit(frame, target)
    assert p.lifecycle_state_ == "FITTED"


def test_freeze_transitions_and_blocks_set_params(frame, target):
    p = _make().fit(frame, target)
    returned = p.freeze()
    assert returned is p
    assert p.is_frozen() is True
    assert p.lifecycle_state_ == "FROZEN"

    with pytest.raises(FrozenRepresentationError, match="frozen"):
        p.set_params(output_dim=9)


def test_set_params_allowed_before_freeze(frame, target):
    p = _make()
    p.set_params(output_dim=9)
    assert p.output_dim == 9


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_frozen_fit_rejected_without_mutating_state(frame, target, method):
    p = _make().fit(frame, target).freeze()
    fingerprint = p.fingerprint_
    original = p.transform(frame, return_array=True)
    changed = frame.assign(a=frame["a"] * 100)
    with pytest.raises(FrozenRepresentationError, match="frozen"):
        getattr(p, method)(changed, target)
    assert p.fingerprint_ == fingerprint
    np.testing.assert_array_equal(p.transform(frame, return_array=True), original)


@pytest.mark.parametrize(
    ("configured", "transform"),
    [(None, "default"), (None, "pandas"), (None, "polars"), ("default", "pandas"), ("pandas", "default")],
)
@pytest.mark.parametrize("via_pipeline", [False, True])
def test_frozen_set_output_rejects_changes(frame, target, configured, transform, via_pipeline):
    processor = _make()
    processor.set_output(transform=configured)
    processor.fit(frame, target).freeze()
    fingerprint = processor.fingerprint_
    original = processor.transform(frame, return_array=True)
    output_config = getattr(processor, "_sklearn_output_config", None)
    caller = Pipeline([("pre", processor)]) if via_pipeline else processor

    with pytest.raises(FrozenRepresentationError, match=r"clone_unfitted\(\).*refit\(\)"):
        caller.set_output(transform=transform)

    assert getattr(processor, "_sklearn_output_config", None) == output_config
    assert processor.fingerprint_ == fingerprint
    np.testing.assert_array_equal(processor.transform(frame, return_array=True), original)


@pytest.mark.parametrize("configured", [None, "default", "pandas"])
@pytest.mark.parametrize("via_pipeline", [False, True])
def test_frozen_set_output_noops_preserve_state(frame, target, configured, via_pipeline):
    processor = _make()
    processor.set_output(transform=configured)
    processor.fit(frame, target).freeze()
    fingerprint = processor.fingerprint_
    original = processor.transform(frame, return_array=True)
    caller = Pipeline([("pre", processor)]) if via_pipeline else processor

    assert caller.set_output() is caller
    assert caller.set_output(transform=configured) is caller

    assert processor.fingerprint_ == fingerprint
    np.testing.assert_array_equal(processor.transform(frame, return_array=True), original)


@pytest.mark.parametrize("already_fitted", [False, True])
def test_unfrozen_set_output_keeps_sklearn_behavior(frame, target, already_fitted):
    processor = _make()
    if already_fitted:
        processor.fit(frame, target)
    assert processor.set_output(transform="pandas") is processor
    if not already_fitted:
        processor.fit(frame, target)

    wrapped = processor.transform(frame, return_array=True)
    assert isinstance(wrapped, pd.DataFrame)
    assert list(wrapped.columns) == list(processor.get_feature_names_out())
    assert processor.set_output(transform="default") is processor
    unwrapped = processor.transform(frame, return_array=True)
    assert isinstance(unwrapped, np.ndarray)
    np.testing.assert_array_equal(unwrapped, wrapped.to_numpy())


@pytest.mark.parametrize("pin_output", [False, True])
def test_frozen_global_output_config_preserves_state(frame, target, pin_output):
    processor = _make()
    if pin_output:
        processor.set_output(transform="default")
    processor.fit(frame, target).freeze()
    fingerprint = processor.fingerprint_
    original = processor.transform(frame, return_array=True)

    with config_context(transform_output="pandas"):
        transformed = processor.transform(frame, return_array=True)
        assert isinstance(transformed, np.ndarray if pin_output else pd.DataFrame)
        np.testing.assert_array_equal(transformed, original)
        assert processor.fingerprint_ == fingerprint

    assert isinstance(processor.transform(frame, return_array=True), np.ndarray)
    assert processor.fingerprint_ == fingerprint


def test_clone_unfitted_returns_fresh_unfrozen(frame, target):
    p = _make().fit(frame, target).freeze()
    clone = p.clone_unfitted()
    assert clone is not p
    assert clone.lifecycle_state_ == "UNFITTED"
    assert clone.is_frozen() is False
    # Params carry over; the clone is mutable.
    clone.set_params(output_dim=5)
    assert clone.output_dim == 5


def test_refit_returns_new_object_and_leaves_original(frame, target):
    p = _make().fit(frame, target).freeze()
    refit = p.refit(frame, target)
    assert refit is not p
    assert refit.lifecycle_state_ == "FITTED"
    assert refit.is_frozen() is False
    # Original stays frozen and untouched.
    assert p.is_frozen() is True
    out_frozen = p.transform(frame, return_array=True)
    out_refit = refit.transform(frame, return_array=True)
    assert isinstance(out_frozen, np.ndarray)
    assert isinstance(out_refit, np.ndarray)
    assert np.array_equal(out_frozen, out_refit, equal_nan=True)


def test_mark_stale(frame, target):
    p = _make().fit(frame, target)
    returned = p.mark_stale("input schema drifted")
    assert returned is p
    assert p.lifecycle_state_ == "STALE"
    assert p.stale_reason_ == "input schema drifted"


def test_frozen_takes_precedence_over_stale(frame, target):
    p = _make().fit(frame, target)
    p.mark_stale("drift")
    p.freeze()
    assert p.lifecycle_state_ == "FROZEN"


def test_freeze_requires_fitted():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        _make().freeze()


def test_clone_preserves_unfrozen_via_sklearn_clone(frame, target):
    from sklearn.base import clone

    p = _make().fit(frame, target).freeze()
    fresh = clone(p)
    assert isinstance(fresh, Preprocessor)
    assert fresh.is_frozen() is False
    fresh.set_params(output_dim=7)
    assert fresh.output_dim == 7
