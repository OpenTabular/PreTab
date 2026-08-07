"""Tests for the high-level ``missing_policy`` orchestration knob (P8.5)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from pretab import Preprocessor
from pretab.exceptions import InvalidParamError, PretabDataError


@pytest.fixture
def frame_with_nan():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "b": [0.1, 0.2, 0.3, np.nan, 0.5, 0.6],
        }
    )


@pytest.fixture
def clean_frame():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )


@pytest.fixture
def y():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def _bspline(**kwargs):
    return Preprocessor(
        numerical_method="bspline",
        output_dim=8,
        target_aware=False,
        placement_strategy="quantile",
        **kwargs,
    )


# --- default / roundtrip -------------------------------------------------------


def test_default_missing_policy_is_none():
    assert Preprocessor().missing_policy is None


def test_missing_policy_survives_clone():
    p = _bspline(missing_policy="separate_state")
    assert clone(p).missing_policy == "separate_state"


def test_default_still_imputes(frame_with_nan, y):
    # missing_policy=None keeps numerical_imputation="median" authoritative.
    p = Preprocessor(numerical_method="minmax").fit(frame_with_nan, y)
    out = p.transform(frame_with_nan, return_array=True)
    assert not np.isnan(out).any()


# --- error ---------------------------------------------------------------------


def test_error_policy_raises_at_fit(frame_with_nan, y):
    with pytest.raises(PretabDataError, match="missing_policy='error'"):
        _bspline(missing_policy="error").fit(frame_with_nan, y)


def test_error_policy_raises_at_transform(clean_frame, frame_with_nan, y):
    p = _bspline(missing_policy="error").fit(clean_frame, y)
    with pytest.raises(PretabDataError, match="missing_policy='error'"):
        p.transform(frame_with_nan)


def test_error_policy_passes_when_clean(clean_frame, y):
    p = _bspline(missing_policy="error").fit(clean_frame, y)
    out = p.transform(clean_frame, return_array=True)
    assert np.isfinite(out).all()


# --- propagate -----------------------------------------------------------------


def test_propagate_lets_nan_through(frame_with_nan, y):
    # MinMaxScaler maintains NaNs at transform; with no imputer they survive.
    p = Preprocessor(numerical_method="minmax", missing_policy="propagate").fit(frame_with_nan, y)
    out = p.transform(frame_with_nan, return_array=True)
    assert np.isnan(out).any()


# --- impute --------------------------------------------------------------------


def test_impute_removes_nan(frame_with_nan, y):
    p = Preprocessor(numerical_method="minmax", missing_policy="impute").fit(frame_with_nan, y)
    out = p.transform(frame_with_nan, return_array=True)
    assert not np.isnan(out).any()


def test_impute_adds_no_indicator(frame_with_nan, y):
    p = Preprocessor(numerical_method="minmax", missing_policy="impute").fit(frame_with_nan, y)
    names = list(p.get_feature_names_out())
    assert not any("missing" in n for n in names)


# --- impute_with_indicator -----------------------------------------------------


def test_impute_with_indicator_appends_columns(frame_with_nan, y):
    plain = Preprocessor(numerical_method="minmax", missing_policy="impute").fit(frame_with_nan, y)
    withind = Preprocessor(numerical_method="minmax", missing_policy="impute_with_indicator").fit(frame_with_nan, y)
    assert withind.total_output_dim_ > plain.total_output_dim_
    out = withind.transform(frame_with_nan, return_array=True)
    assert not np.isnan(out).any()


# --- separate_state ------------------------------------------------------------


def test_separate_state_emits_missing_column(frame_with_nan, y):
    p = _bspline(missing_policy="separate_state").fit(frame_with_nan, y)
    names = list(p.get_feature_names_out())
    missing_cols = [n for n in names if n.endswith("__missing")]
    assert len(missing_cols) == 2  # one per input feature


def test_separate_state_output_is_finite(frame_with_nan, y):
    p = _bspline(missing_policy="separate_state").fit(frame_with_nan, y)
    out = p.transform(frame_with_nan, return_array=True)
    assert np.isfinite(out).all()


def test_separate_state_indicator_marks_missing_rows(frame_with_nan, y):
    p = _bspline(missing_policy="separate_state").fit(frame_with_nan, y)
    names = list(p.get_feature_names_out())
    arr = p.transform(frame_with_nan, return_array=True)
    a_missing_name = next(n for n in names if n.startswith("num_a") and n.endswith("__missing"))
    a_missing = arr[:, names.index(a_missing_name)]
    # Column "a" is missing at row index 2.
    assert a_missing[2] == 1.0
    assert a_missing[0] == 0.0


def test_separate_state_on_categorical(y):
    frame = pd.DataFrame({"c": ["x", "y", None, "x", "y", "x"]})
    p = Preprocessor(categorical_method="one-hot", missing_policy="separate_state").fit(frame, y)
    names = list(p.get_feature_names_out())
    assert any(n.endswith("__missing") for n in names)


# --- validation ----------------------------------------------------------------


def test_invalid_missing_policy_raises(frame_with_nan, y):
    with pytest.raises(InvalidParamError):
        _bspline(missing_policy="nonsense").fit(frame_with_nan, y)
