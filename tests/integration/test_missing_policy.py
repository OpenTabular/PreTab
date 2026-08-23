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
    cloned = clone(p)
    assert isinstance(cloned, Preprocessor)
    assert cloned.missing_policy == "separate_state"


def test_default_still_imputes(frame_with_nan, y):
    # missing_policy=None keeps numerical_imputation="median" authoritative.
    p = Preprocessor(numerical_method="minmax").fit(frame_with_nan, y)
    out = p.transform(frame_with_nan, return_array=True)
    assert isinstance(out, np.ndarray)
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
    assert isinstance(out, np.ndarray)
    assert np.isfinite(out).all()


# --- propagate -----------------------------------------------------------------


def test_propagate_lets_nan_through(frame_with_nan, y):
    # MinMaxScaler maintains NaNs at transform; with no imputer they survive.
    p = Preprocessor(numerical_method="minmax", missing_policy="propagate").fit(frame_with_nan, y)
    out = p.transform(frame_with_nan, return_array=True)
    assert isinstance(out, np.ndarray)
    assert np.isnan(out).any()


# --- impute --------------------------------------------------------------------


def test_impute_removes_nan(frame_with_nan, y):
    p = Preprocessor(numerical_method="minmax", missing_policy="impute").fit(frame_with_nan, y)
    out = p.transform(frame_with_nan, return_array=True)
    assert isinstance(out, np.ndarray)
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
    assert isinstance(out, np.ndarray)
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
    assert isinstance(out, np.ndarray)
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


def test_separate_state_feature_info_reports_both_branches(frame_with_nan, y):
    p = _bspline(missing_policy="separate_state").fit(frame_with_nan, y)

    numerical, categorical, embeddings = p.get_feature_info(verbose=False)

    assert categorical == {}
    assert embeddings == {}
    for feature in frame_with_nan.columns:
        assert "representation(" in numerical[feature]["preprocessing"]
        assert "+ missing" in numerical[feature]["preprocessing"]
        assert numerical[feature]["dimension"] == p.output_dims_[feature]


def test_separate_state_verbose_two_does_not_break_fit(frame_with_nan, y):
    fitted = _bspline(missing_policy="separate_state", verbose=2).fit(frame_with_nan, y)
    assert fitted.total_output_dim_ > 0


def test_separate_state_lineage_distinguishes_missing_indicator(frame_with_nan, y):
    p = Preprocessor(numerical_method="minmax", missing_policy="separate_state").fit(frame_with_nan, y)

    lineage = p.get_feature_lineage()
    representation = [record for record in lineage if record.family != "missing_state"]
    missing = [record for record in lineage if record.family == "missing_state"]

    assert len(missing) == len(frame_with_nan.columns)
    assert all(record.family == "minmax" and record.component == "raw" for record in representation)
    assert all(record.component == "indicator" for record in missing)
    assert all(record.output_feature.endswith("__missing") for record in missing)
    assert [record.output_feature for record in lineage] == list(p.get_feature_names_out())


# --- validation ----------------------------------------------------------------


def test_invalid_missing_policy_raises(frame_with_nan, y):
    with pytest.raises(InvalidParamError):
        _bspline(missing_policy="nonsense").fit(frame_with_nan, y)


# --- add_missing_indicator with imputation disabled ----------------------------


def test_add_missing_indicator_with_imputation_none_emits_standalone_column(frame_with_nan, y):
    """Regression guard: add_missing_indicator=True must work even when the
    imputer for that column kind is disabled, per the documented "standalone
    indicator when imputation is disabled" contract."""
    p = Preprocessor(
        numerical_method="minmax",
        numerical_imputation=None,
        add_missing_indicator=True,
    ).fit(frame_with_nan, y)

    names = list(p.get_feature_names_out())
    missing_cols = [n for n in names if n.endswith("__missing")]
    assert len(missing_cols) == len(frame_with_nan.columns)


def test_add_missing_indicator_with_imputation_none_still_propagates_nan(frame_with_nan, y):
    """The representation branch is unaffected: NaN still reaches the transformer
    unchanged, only the standalone indicator is added alongside it."""
    p = Preprocessor(
        numerical_method="minmax",
        numerical_imputation=None,
        add_missing_indicator=True,
    ).fit(frame_with_nan, y)

    out = p.transform(frame_with_nan, return_array=True)
    assert np.isnan(out).any()


def test_add_missing_indicator_numerical_only_disabled(y):
    """Regression guard: a numerical-only imputation=None combined with a
    categorical column that has no missing values must still produce the
    standalone numerical indicator (previously silently dropped)."""
    frame = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "c": ["x", "y", "x", "y"]})
    p = Preprocessor(
        numerical_method="minmax",
        categorical_method="int",
        numerical_imputation=None,
        add_missing_indicator=True,
    ).fit(frame, y[:4])

    names = list(p.get_feature_names_out())
    assert any(n.endswith("__missing") for n in names)


def test_add_missing_indicator_no_longer_requires_imputation_enabled(frame_with_nan, y):
    """add_missing_indicator=True with both imputations disabled used to raise
    IncompatibleParamsError; it must now fit successfully via standalone
    indicators for every column."""
    p = Preprocessor(
        numerical_method="minmax",
        numerical_imputation=None,
        categorical_imputation=None,
        add_missing_indicator=True,
    ).fit(frame_with_nan, y)
    assert p.total_output_dim_ > 0
