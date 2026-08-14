"""Unit tests for :mod:`pretab.compose.inspection`."""

import numpy as np
import pandas as pd
import pytest

from pretab.compose.factory import build_column_transformer
from pretab.compose.inspection import (
    build_feature_info,
    build_transformer_summary,
    clean_feature_names,
    get_output_slices,
)


@pytest.fixture
def fitted_ct(make_config, sample_frame):
    ct = build_column_transformer(make_config(numerical_method="standardization"), ["age"], ["city"])
    ct.fit(sample_frame, np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))
    return ct, sample_frame


def test_get_output_slices_are_ordered_and_named(fitted_ct):
    ct, X = fitted_ct
    slices = get_output_slices(ct, X)
    names = [name for name, _, _ in slices]
    assert "num_age" in names and "cat_city" in names
    starts = [start for _, start, _ in slices]
    assert starts == sorted(starts)
    assert all(width >= 1 for _, _, width in slices)


def test_build_feature_info_splits_numerical_and_categorical(fitted_ct):
    ct, _ = fitted_ct
    numerical, categorical, embeddings = build_feature_info(ct, embeddings=False, embedding_dimensions={})
    assert "age" in numerical
    assert "city" in categorical
    assert embeddings == {}


def test_build_feature_info_reports_embeddings(fitted_ct):
    ct, _ = fitted_ct
    _, _, embeddings = build_feature_info(ct, embeddings=True, embedding_dimensions={"embedding_1": 8})
    assert embeddings == {"embedding_1": {"preprocessing": None, "dimension": 8, "categories": None}}


def test_build_transformer_summary_has_header_and_rows():
    numerical = {"age": {"preprocessing": "imputer -> standardization", "dimension": 1, "categories": None}}
    categorical = {"city": {"preprocessing": "imputer -> continuous_ordinal", "dimension": 1, "categories": 3}}
    lines = build_transformer_summary(numerical, categorical, {})
    assert lines[0].startswith("feature")
    assert any("age" in line for line in lines)
    assert any("city" in line for line in lines)


def test_build_transformer_summary_empty_returns_empty():
    assert build_transformer_summary({}, {}, {}) == []


def test_clean_feature_names_collapses_1_to_1_step(fitted_ct):
    ct, _ = fitted_ct
    raw = [str(name) for name in ct.get_feature_names_out()]
    assert any("__" in name for name in raw), "sanity: sklearn's default naming should duplicate here"
    assert clean_feature_names(ct, raw) == ["num_age", "cat_city"]


def test_clean_feature_names_handles_underscore_in_feature_name(make_config):
    # "annual_income" itself contains "_", so a naive string split on "_" would
    # mis-collapse this; the fix must use the ColumnTransformer's own column metadata.
    df = pd.DataFrame({"annual_income": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    ct = build_column_transformer(make_config(numerical_method="bspline", output_dim=5), ["annual_income"], [])
    ct.fit(df)
    raw = [str(name) for name in ct.get_feature_names_out()]
    cleaned = clean_feature_names(ct, raw)
    assert all("__" not in name for name in cleaned)
    assert all(name.startswith("num_annual_income_bs") for name in cleaned)


def test_clean_feature_names_leaves_unmatched_names_untouched(fitted_ct):
    ct, _ = fitted_ct
    untouched = ["remainder__extra", "totally_unrelated_name"]
    assert clean_feature_names(ct, untouched) == untouched


def test_clean_feature_names_leaves_non_matching_inner_name_untouched(fitted_ct):
    ct, _ = fitted_ct
    raw = ["num_age__somethingelse"]
    assert clean_feature_names(ct, raw) == raw
