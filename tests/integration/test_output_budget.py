"""Output-budget controls on the Preprocessor (roadmap Phase 8, P8.3).

The budget parameters are opt-in: with all of them ``None`` (the default) the
Preprocessor behaves exactly as before. When a budget is set, exceeding it is
handled by ``overflow_policy`` -- raise, warn, or ignore.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from pretab import OutputBudgetError, Preprocessor
from pretab.exceptions import ConfigWarning, InvalidParamError


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})


@pytest.fixture
def y():
    return np.random.default_rng(1).normal(size=50)


def _bspline(**kwargs):
    return Preprocessor(
        numerical_method="bspline",
        output_dim=8,
        target_aware=False,
        placement_strategy="quantile",
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Estimation helpers
# --------------------------------------------------------------------------- #
def test_estimate_output_shape_matches_transform(frame, y):
    pre = _bspline().fit(frame, y)
    n_rows, n_cols = pre.estimate_output_shape(frame)
    assert n_rows == frame.shape[0]
    assert n_cols == pre.total_output_dim_
    out = pre.transform(frame, return_array=True)
    assert isinstance(out, np.ndarray)
    assert out.shape == (n_rows, n_cols)


def test_estimate_memory_is_rows_times_cols_times_itemsize(frame, y):
    pre = _bspline().fit(frame, y)
    n_rows, n_cols = pre.estimate_output_shape(frame)
    assert pre.estimate_memory(frame) == n_rows * n_cols * np.dtype(np.float64).itemsize


def test_estimate_shape_scales_with_new_rows(frame, y):
    pre = _bspline().fit(frame, y)
    bigger = pd.concat([frame] * 3, ignore_index=True)
    assert pre.estimate_output_shape(bigger)[0] == frame.shape[0] * 3


# --------------------------------------------------------------------------- #
# No budget set -> no enforcement (non-regressive default)
# --------------------------------------------------------------------------- #
def test_default_has_no_budget_enforcement(frame, y):
    # Fits fine even though the output is wider than any of the (unset) budgets.
    pre = _bspline().fit(frame, y)
    assert pre.total_output_dim_ > 0


# --------------------------------------------------------------------------- #
# max_output_features
# --------------------------------------------------------------------------- #
def test_max_output_features_error(frame, y):
    with pytest.raises(OutputBudgetError, match="max_output_features"):
        _bspline(max_output_features=10).fit(frame, y)


def test_max_output_features_within_budget_is_fine(frame, y):
    pre = _bspline().fit(frame, y)
    _bspline(max_output_features=pre.total_output_dim_).fit(frame, y)


# --------------------------------------------------------------------------- #
# max_features_per_input
# --------------------------------------------------------------------------- #
def test_max_features_per_input_error(frame, y):
    with pytest.raises(OutputBudgetError, match="max_features_per_input"):
        _bspline(max_features_per_input=5).fit(frame, y)


# --------------------------------------------------------------------------- #
# max_dense_memory
# --------------------------------------------------------------------------- #
def test_max_dense_memory_error(frame, y):
    with pytest.raises(OutputBudgetError, match="max_dense_memory"):
        _bspline(max_dense_memory=100).fit(frame, y)


def test_max_dense_memory_generous_budget_is_fine(frame, y):
    _bspline(max_dense_memory=10**9).fit(frame, y)


# --------------------------------------------------------------------------- #
# overflow_policy
# --------------------------------------------------------------------------- #
def test_overflow_policy_warn(frame, y):
    with pytest.warns(ConfigWarning, match="Output budget exceeded"):
        _bspline(max_output_features=10, overflow_policy="warn").fit(frame, y)


def test_overflow_policy_ignore(frame, y):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigWarning)
        # No warning and no error even though the budget is exceeded.
        _bspline(max_output_features=1, overflow_policy="ignore").fit(frame, y)


def test_overflow_policy_invalid(frame, y):
    with pytest.raises(InvalidParamError):
        _bspline(max_output_features=1, overflow_policy="bogus").fit(frame, y)


def test_multiple_budgets_reported_together(frame, y):
    with pytest.raises(OutputBudgetError) as excinfo:
        _bspline(max_output_features=1, max_features_per_input=1).fit(frame, y)
    message = str(excinfo.value)
    assert "max_output_features" in message
    assert "max_features_per_input" in message
