"""Unit tests for :mod:`pretab.compose.inspection`."""

import warnings
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, PolynomialFeatures

from pretab.compose.factory import build_column_transformer, get_categorical_transformer_steps
from pretab.compose.inspection import (
    _iter_fitted_blocks,
    _locate_block_representation,
    block_uses_target,
    build_feature_info,
    build_feature_lineage,
    build_transformer_summary,
    clean_feature_names,
    feature_names_out,
    get_output_slices,
)
from pretab.core.representation import RepresentationSpec


@pytest.fixture
def fitted_ct(make_config, sample_frame):
    ct = build_column_transformer(make_config(numerical_method="standardization"), ["age"], ["city"])
    ct.fit(sample_frame, np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))
    return ct, sample_frame


@pytest.fixture(params=["tensorspline", "tprs"])
def grouped_ct(request, make_config):
    values = np.random.default_rng(42).normal(size=(24, 3))
    frame = pd.DataFrame(values, columns=pd.Index(["age", "x% west", "y+east"]))
    frame["city"] = ["a", "b"] * 12
    options = (
        {"output_dim": 3, "degree": 2}
        if request.param == "tensorspline"
        else {"n_components": 3, "landmark_strategy": "subsample", "random_state": 0}
    )
    config = make_config(
        feature_groups={
            "location": {"columns": ["y+east", "x% west"], "representation": {"method": request.param, **options}}
        }
    )
    transformer = build_column_transformer(config, ["age", "x% west", "y+east"], ["city"])
    transformer.fit(frame)
    return transformer, frame


def test_group_feature_info_counts_joint_width_once(grouped_ct):
    transformer, _ = grouped_ct
    info = build_feature_info(transformer, embeddings=False, embedding_dimensions={})
    assert len(info) == 3
    numerical, categorical, embeddings = info
    assert list(numerical) == ["age", "group_location"]
    assert list(categorical) == ["city"]
    assert embeddings == {}
    group = numerical["group_location"]
    span = transformer.output_indices_["group_location"]
    assert group["dimension"] == span.stop - span.start
    assert group["group"] == "location"
    assert group["source_features"] == ("y+east", "x% west")
    assert group["categories"] is None
    assert set(numerical["age"]) == {"preprocessing", "dimension", "categories"}


def test_group_names_encode_ordered_sources_and_preserve_leaf_names(grouped_ct):
    transformer, _ = grouped_ct
    raw = transformer.get_feature_names_out()
    names = feature_names_out(transformer)
    span = transformer.output_indices_["group_location"]
    expected = [f"group_location__y%2Beast+x%25%20west__{name.partition('__')[2]}" for name in raw[span]]
    assert names[span] == expected
    assert len(set(names)) == len(names)
    assert names[0] == "num_age"


def test_explicit_group_display_names_preserve_identity_without_mutating_fit(grouped_ct):
    transformer, _ = grouped_ct
    original = feature_names_out(transformer)
    fitted = transformer.feature_names_in_.copy()
    renamed = feature_names_out(transformer, ["years", "west%", "east+", "town"])
    span = transformer.output_indices_["group_location"]
    assert all(name.startswith("group_location__east%2B+west%25__") for name in renamed[span])
    assert renamed[0] == "num_years"
    assert len(renamed) == len(original)
    assert feature_names_out(transformer) == original
    np.testing.assert_array_equal(transformer.feature_names_in_, fitted)
    with pytest.raises(ValueError):
        feature_names_out(transformer, ["too", "short"])


def test_group_lineage_covers_actual_span_with_one_identity(grouped_ct):
    transformer, _ = grouped_ct
    names = feature_names_out(transformer)
    lineage = build_feature_lineage(transformer)
    span = transformer.output_indices_["group_location"]
    assert [record.output_feature for record in lineage] == names
    assert [record.output_index for record in lineage] == list(range(len(names)))
    for record in lineage[span]:
        assert record.group_id == "group_location"
        assert record.source_features == ("y+east", "x% west")
        assert record.to_dict()["group_id"] == "group_location"
    assert [record.component_index for record in lineage[span]] == list(range(span.stop - span.start))
    for record in [*lineage[: span.start], *lineage[span.stop :]]:
        assert record.group_id is None
        assert "group_id" not in record.to_dict()


@pytest.fixture
def declared_representation():
    spec = RepresentationSpec(
        family="declared_joint_basis",
        component_kind="basis",
        scope="multivariate",
        supervision="optional",
        uses_target=True,
        is_interaction=True,
        input_features=("age", "income"),
        output_features=("age_income",),
        output_dim=1,
        degree=None,
        include_bias=False,
        periodic=False,
        period=None,
        local_support=False,
        location_kind=None,
        locations=None,
    )
    representation = Mock(spec=["get_representation_spec", "n_features_in_"])
    representation.n_features_in_ = 2
    representation.get_representation_spec.return_value = spec
    return representation


@pytest.mark.parametrize("step_name", ["representation", "scaler", "onehot"])
@pytest.mark.parametrize("trailing_helper", [False, True])
def test_locate_representation_spec_precedes_legacy_steps(declared_representation, step_name, trailing_helper):
    steps = [("imputer", FunctionTransformer()), (step_name, declared_representation)]
    if trailing_helper:
        steps.append(("minmax", MaxAbsScaler()))
    pipeline = Pipeline(steps)

    assert _locate_block_representation(pipeline, ["age", "income"]) == (
        1,
        ("declared_joint_basis", "basis", True, True),
    )
    assert block_uses_target(pipeline) is True


@pytest.mark.parametrize(
    ("columns", "fitted_width", "expected_names"),
    [
        (["age", "income"], 2, ["age", "income"]),
        (("age", "income"), 2, ["age", "income"]),
        (["age"], 2, None),
        (None, 2, None),
    ],
)
def test_locate_representation_only_forwards_matching_input_names(
    declared_representation, columns, fitted_width, expected_names
):
    declared_representation.n_features_in_ = fitted_width

    assert _locate_block_representation(declared_representation, columns) == (
        0,
        ("declared_joint_basis", "basis", True, True),
    )
    declared_representation.get_representation_spec.assert_called_once_with(input_features=expected_names)


def test_locate_representation_uses_the_rightmost_declared_spec(declared_representation):
    earlier = Mock(spec=["get_representation_spec"])
    earlier.get_representation_spec.side_effect = AssertionError("An earlier declaration must not be inspected")
    pipeline = Pipeline([("earlier", earlier), ("representation", declared_representation), ("scaler", MaxAbsScaler())])

    assert _locate_block_representation(pipeline, ["age", "income"]) == (
        1,
        ("declared_joint_basis", "basis", True, True),
    )
    earlier.get_representation_spec.assert_not_called()


@pytest.mark.parametrize(
    ("step_names", "expected"),
    [
        (["scaler"], (0, ("standardization", "raw", False, False))),
        (["onehot", "scaler"], (1, ("standardization", "raw", False, False))),
        (["scaler", "onehot"], (1, ("onehot", "category", False, False))),
        (["imputer", "to_float"], (None, ("passthrough", "raw", False, False))),
    ],
)
def test_locate_representation_legacy_fallback_is_the_rightmost_known_step(step_names, expected):
    pipeline = Pipeline([(name, FunctionTransformer()) for name in step_names])

    assert _locate_block_representation(pipeline, ["age"]) == expected
    assert block_uses_target(pipeline) is False


def test_get_output_slices_are_ordered_and_named(fitted_ct):
    ct, _ = fitted_ct
    slices = get_output_slices(ct)
    names = [name for name, _, _ in slices]
    assert "num_age" in names and "cat_city" in names
    starts = [start for _, start, _ in slices]
    assert starts == sorted(starts)
    assert all(width >= 1 for _, _, width in slices)


def test_fitted_block_view_retains_entries_and_fitted_references():
    transformer = Mock(spec=["fit", "transform", "get_feature_names_out", "get_representation_spec"])
    columns = ["age__years"]
    ct = SimpleNamespace(
        transformers_=[
            ("num_0", transformer, columns),
            ("cat_city", "drop", ["city"]),
            ("empty", "passthrough", []),
            ("unmapped", "passthrough", ["other"]),
            ("remainder", "passthrough", [1]),
        ],
        output_indices_={
            "num_0": slice(0, 2),
            "cat_city": slice(0, 0),
            "empty": slice(2, 2),
            "remainder": slice(2, 3),
        },
    )
    original_blocks = ct.transformers_.copy()
    original_indices = ct.output_indices_.copy()

    blocks = list(_iter_fitted_blocks(ct))

    assert [block.step_name for block in blocks] == [entry[0] for entry in original_blocks]
    for block, (_, fitted, selector) in zip(blocks, original_blocks, strict=True):
        assert block.transformer is fitted
        assert block.columns is selector
    assert blocks[0].public_name == "num_age__years"
    assert blocks[0].span is ct.output_indices_["num_0"]
    with pytest.raises(AttributeError):
        cast(Any, blocks[0]).step_name = "replacement"
    with pytest.raises(KeyError, match="unmapped"):
        _ = blocks[3].span
    assert get_output_slices(ct) == [("num_age__years", 0, 2), ("remainder", 2, 1)]
    assert ct.transformers_ == original_blocks
    assert ct.output_indices_ == original_indices
    assert transformer.mock_calls == []


def test_fitted_block_view_does_not_eagerly_read_names_or_spans():
    columns = slice(0, 2)
    transformer = object()
    ct = SimpleNamespace(transformers_=[("joint", transformer, columns)])

    block = next(_iter_fitted_blocks(ct))

    assert block.step_name == "joint"
    assert block.transformer is transformer
    assert block.columns is columns
    with pytest.raises(AttributeError, match="output_indices_"):
        _ = block.span


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

    assert numerical["age"]["dimension"] == len(expected_names) == 1
    assert expected.shape == (len(frame), 1)
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
    transformed = ct.transform(frame)
    assert categorical["city"]["dimension"] == transformed[:, span].shape[1] == 0
    assert categorical["city"]["categories"] == 1
    assert transformed.shape == (len(frame), 1)
    assert len(ct.get_feature_names_out()) == 1
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
    width = categorical["city"]["dimension"]
    assert width == len(expected_names) == get_output_slices(ct)[0][2]
    assert expected.shape == (len(frame), width)
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
    assert sum(spans.values()) == len(ct.get_feature_names_out())
    assert ct.transform(frame).shape == (len(frame), sum(spans.values()))


@pytest.mark.parametrize("passthrough", ["direct", "pipeline"])
def test_build_feature_info_handles_passthrough_and_dropped_blocks(passthrough):
    frame = pd.DataFrame({"age": [1.0, 2.0, 3.0], "ignored": [4.0, 5.0, 6.0]})
    transformer = "passthrough" if passthrough == "direct" else Pipeline([("passthrough", "passthrough")])
    ct = ColumnTransformer([("num_age", transformer, ["age"]), ("num_ignored", "drop", ["ignored"])]).fit(frame)

    numerical, categorical, _ = build_feature_info(ct, embeddings=False, embedding_dimensions={})

    assert list(numerical) == ["age"]
    assert categorical == {}
    assert numerical["age"]["dimension"] == len(ct.get_feature_names_out()) == 1
    assert ct.transform(frame).shape == (len(frame), 1)


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
