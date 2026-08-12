"""Tests for the immutable lifecycle: freeze / stale / clone / refit (P9.3).

Covers the ``lifecycle_state_`` transitions, ``freeze`` / ``is_frozen``,
``set_params`` rejection on frozen instances, ``clone_unfitted`` and ``refit``
returning fresh objects, and ``mark_stale``.
"""

import numpy as np
import pandas as pd
import pytest

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
    assert np.array_equal(
        p.transform(frame, return_array=True), refit.transform(frame, return_array=True), equal_nan=True
    )


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
