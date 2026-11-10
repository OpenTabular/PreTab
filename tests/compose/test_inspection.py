"""Unit tests for :mod:`pretab.compose.inspection`."""

import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, PolynomialFeatures

from pretab.compose.factory import build_column_transformer, get_categorical_transformer_steps
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
    ct, _ = fitted_ct
    slices = get_output_slices(ct)
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


def test_build_feature_info_does_not_infer_kind_from_cat_in_feature_name(make_config):
    feature = "education_category_score"
    frame = pd.DataFrame({feature: np.linspace(0.0, 1.0, 8)})
    ct = build_column_transformer(make_config(numerical_method="robust"), [feature], [])
    ct.fit(frame)

    numerical, categorical, _ = build_feature_info(ct, embeddings=False, embedding_dimensions={})

    assert feature in numerical
    assert feature not in categorical


def test_build_feature_info_uses_fitted_width_without_transform(monkeypatch):
    frame = pd.DataFrame({"age": [1.0, 2.0, 4.0]})
    pipeline = Pipeline([("positive_only", FunctionTransformer(np.log, feature_names_out="one-to-one"))])
    ct = ColumnTransformer([("num_age", pipeline, ["age"])]).fit(frame)
    expected = ct.transform(frame)
    expected_names = ct.get_feature_names_out()

    def reject_transform(*_args, **_kwargs):
        raise AssertionError("Feature inspection must not invoke transform")

    fitted_transformer = ct.named_transformers_["num_age"].named_steps["positive_only"]
    monkeypatch.setattr(fitted_transformer, "transform", reject_transform)

    numerical, _, _ = build_feature_info(ct, embeddings=False, embedding_dimensions={})

    assert numerical["age"]["dimension"] == np.shape(expected)[1] == len(expected_names) == 1
    assert numerical["age"]["dimension"] == get_output_slices(ct)[0][2]


@pytest.mark.parametrize(("method", "dimension"), [("rbf", 3), ("standardization", 1)])
def test_build_feature_info_reads_width_after_frame_fit_without_warning(make_config, sample_frame, method, dimension):
    """Fitting on named columns must not trigger feature-name warnings during inspection."""
    config = make_config(numerical_method=method, numerical_imputation=None, output_dim=3)
    ct = build_column_transformer(config, ["age"], ["city"]).fit(sample_frame)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        numerical, _, _ = build_feature_info(ct, embeddings=False, embedding_dimensions={})

    assert numerical["age"]["dimension"] == dimension


def test_build_feature_info_reports_a_zero_width_block():
    # scikit-learn's imputer default drops a column without observed values, so
    # the encoder after it receives no column at all.
    steps = get_categorical_transformer_steps("int", imputer_kwargs={"keep_empty_features": False})
    ct = ColumnTransformer([("cat_city", Pipeline(steps), ["city"])])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Skipping features without any observed values", UserWarning)
        ct.fit(pd.DataFrame({"city": pd.Series([np.nan] * 4, dtype=object)}))

    _, categorical, _ = build_feature_info(ct, embeddings=False, embedding_dimensions={})

    assert categorical["city"]["dimension"] == 0


@pytest.mark.parametrize("sparse_output", [False, True])
def test_build_feature_info_reports_zero_width_after_dropping_only_category(sparse_output):
    frame = pd.DataFrame({"city": ["Rome", "Rome", "Rome"], "age": [1.0, 2.0, 3.0]})
    steps = get_categorical_transformer_steps("one-hot", sparse_output=sparse_output, drop="first")
    ct = ColumnTransformer([("cat_city", Pipeline(steps), ["city"]), ("num_age", "passthrough", ["age"])]).fit(frame)

    _, categorical, _ = build_feature_info(ct, embeddings=False, embedding_dimensions={})

    span = ct.output_indices_["cat_city"]
    assert categorical["city"]["dimension"] == np.shape(ct.transform(frame)[:, span])[1] == 0
    assert categorical["city"]["categories"] == 1
    assert np.shape(ct.transform(frame))[1] == len(ct.get_feature_names_out()) == 1
    assert get_output_slices(ct) == [("num_age", 0, 1)]


@pytest.mark.parametrize("sparse_output", [False, True])
@pytest.mark.parametrize("suffix", ["to_float", "scaler", "polynomial"])
@pytest.mark.parametrize(
    "encoder_options",
    [{}, {"drop": "first"}, {"min_frequency": 2}, {"max_categories": 2}, {"drop": "first", "min_frequency": 2}],
)
def test_build_feature_info_separates_vocabulary_from_emitted_width(
    monkeypatch, sparse_output, suffix, encoder_options
):
    frame = pd.DataFrame({"city": ["Rome", "Rome", "Berlin", "Oslo", "Paris"]})
    steps = get_categorical_transformer_steps(
        "one-hot", add_imputer=False, sparse_output=sparse_output, **encoder_options
    )
    if suffix == "scaler":
        steps.append(("scaler", MaxAbsScaler()))
    elif suffix == "polynomial":
        steps.append(("polynomial", PolynomialFeatures(degree=2, include_bias=False)))
    ct = ColumnTransformer([("cat_city", Pipeline(steps), ["city"])]).fit(frame)
    expected = ct.transform(frame)
    expected_names = ct.get_feature_names_out()
    reject_transform = Mock(side_effect=AssertionError("Feature inspection must not invoke transform"))
    for _, step in ct.named_transformers_["cat_city"].steps:
        monkeypatch.setattr(step, "transform", reject_transform)

    numerical, categorical, _ = build_feature_info(ct, embeddings=False, embedding_dimensions={})

    assert numerical == {}
    assert categorical["city"]["categories"] == 4
    assert categorical["city"]["dimension"] == np.shape(expected)[1] == len(expected_names)
    assert categorical["city"]["dimension"] == get_output_slices(ct)[0][2]
    reject_transform.assert_not_called()


@pytest.mark.parametrize("missing_options", [{"add_missing_indicator": True}, {"missing_policy": "separate_state"}])
def test_build_feature_info_uses_combined_indicator_span(make_config, missing_options):
    frame = pd.DataFrame({"age": [1.0, np.nan, 3.0, 4.0], "city": ["Rome", np.nan, "Oslo", "Rome"]})
    config = make_config(numerical_method="rbf", output_dim=3, categorical_method="one-hot", **missing_options)
    ct = build_column_transformer(config, ["age"], ["city"]).fit(frame)

    numerical, categorical, _ = build_feature_info(ct, embeddings=False, embedding_dimensions={})
    spans = {name: width for name, _, width in get_output_slices(ct)}

    assert numerical["age"]["dimension"] == spans["num_age"] == 4
    assert categorical["city"]["dimension"] == spans["cat_city"] == 3
    assert categorical["city"]["categories"] == 2
    assert sum(spans.values()) == np.shape(ct.transform(frame))[1] == len(ct.get_feature_names_out())


@pytest.mark.parametrize("passthrough", ["direct", "pipeline"])
def test_build_feature_info_handles_passthrough_and_dropped_blocks(passthrough):
    frame = pd.DataFrame({"age": [1.0, 2.0, 3.0], "ignored": [4.0, 5.0, 6.0]})
    transformer = "passthrough" if passthrough == "direct" else Pipeline([("passthrough", "passthrough")])
    ct = ColumnTransformer([("num_age", transformer, ["age"]), ("num_ignored", "drop", ["ignored"])]).fit(frame)

    numerical, categorical, _ = build_feature_info(ct, embeddings=False, embedding_dimensions={})

    assert list(numerical) == ["age"]
    assert categorical == {}
    assert numerical["age"]["dimension"] == np.shape(ct.transform(frame))[1] == len(ct.get_feature_names_out()) == 1


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
