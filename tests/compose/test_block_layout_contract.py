"""Agreement between fitted block spans and the public output-layout APIs."""

import json
from contextlib import nullcontext
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils.validation import check_is_fitted

from pretab import BaseRepresentation, Preprocessor, RepresentationConfig, register_representation
from pretab.compose import registry
from pretab.compose.inspection import block_name, build_feature_lineage, get_output_slices
from pretab.embedding.language import LanguageEmbeddingTransformer
from pretab.exceptions import ConfigWarning, IncompatibleParamsError, OutputBudgetError


class _PositiveOnly(BaseRepresentation):
    representation_name = "layout_positive_only"
    feature_kind = "numerical"

    def fit(self, X, y=None):
        self._validate(X, reset=True)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        values = self._validate(X, reset=False)
        if np.any(values <= 0):
            raise ValueError("Positive input required")
        return np.log(values)

    def _output_sizes(self):
        return [1] * self.n_features_in_


@pytest.fixture
def positive_only_method():
    saved_registry = dict(registry.TRANSFORMER_REGISTRY)
    saved_numerical = dict(registry.NUMERICAL_METHODS)
    saved_categorical = set(registry.CATEGORICAL_METHODS)
    try:
        register_representation("layout_positive_only", _PositiveOnly)
        yield "layout_positive_only"
    finally:
        registry.TRANSFORMER_REGISTRY.clear()
        registry.TRANSFORMER_REGISTRY.update(saved_registry)
        registry.NUMERICAL_METHODS.clear()
        registry.NUMERICAL_METHODS.update(saved_numerical)
        registry.CATEGORICAL_METHODS.clear()
        registry.CATEGORICAL_METHODS.update(saved_categorical)


@pytest.fixture
def layout_data():
    values = np.linspace(1.0, 9.0, 80)
    frame = pd.DataFrame(
        {
            "value": values,
            "other": values**2,
            "city": np.resize(["Rome", "Oslo", "Berlin", "Rome"], len(values)),
        }
    )
    frame.loc[[3, 11], "value"] = np.nan
    frame.loc[[5, 17], "city"] = np.nan
    return frame, np.sin(values)


def _dense(values):
    return values.toarray() if sparse.issparse(values) else np.asarray(values)


def _assert_block_layout(preprocessor, frame, *, embeddings=None):
    transformed = (
        preprocessor.transform(frame, return_array=True)
        if embeddings is None
        else preprocessor.column_transformer_.transform(frame)
    )
    blocks = preprocessor.transform(frame, embeddings=embeddings, return_array=False)
    assert isinstance(blocks, dict)
    names = preprocessor.get_feature_names_out()
    lineage = build_feature_lineage(preprocessor.column_transformer_)
    assert lineage == preprocessor.get_feature_lineage()
    numerical, categorical, embedding_info = preprocessor.get_feature_info(verbose=False)
    assert embedding_info == {
        name: {"preprocessing": None, "dimension": width, "categories": None}
        for name, width in preprocessor.embedding_dimensions_.items()
    }
    metadata = {**numerical, **categorical}
    dimensions = preprocessor.output_dims_
    slices = get_output_slices(preprocessor.column_transformer_)
    sources = {
        block_name(name, columns): str(columns[0])
        for name, transformer, columns in preprocessor.column_transformer_.transformers_
        if transformer != "drop" and len(columns) == 1
    }

    assert list(blocks) == [name for name, _, _ in slices] + list(embedding_info)
    assert [record.output_index for record in lineage] == list(range(len(names)))
    assert [record.output_feature for record in lineage] == list(names)
    offset = 0
    for name, start, width in slices:
        source = sources[name]
        records = [record for record in lineage if record.source_features == (source,)]
        stop = start + width
        assert start == offset
        assert [record.output_index for record in records] == list(range(start, stop))
        assert [record.output_feature for record in records] == list(names[start:stop])
        assert width == len(records) == len(names[start:stop])
        assert width == dimensions[source] == metadata[source]["dimension"]
        assert np.shape(blocks[name]) == (len(frame), width)
        np.testing.assert_array_equal(_dense(blocks[name]), _dense(transformed[:, start:stop]))
        offset = stop

    assert sum(dimensions.values()) == offset == preprocessor.total_output_dim_
    assert offset == np.shape(transformed)[1] == len(names) == len(lineage)
    np.testing.assert_array_equal(np.hstack([_dense(blocks[name]) for name, _, _ in slices]), _dense(transformed))


@pytest.mark.parametrize("output_format", ["dense", "sparse"])
@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param({"numerical_method": "bspline", "categorical_method": "one-hot"}, id="spline-onehot"),
        pytest.param({"numerical_method": "rbf", "categorical_method": "int"}, id="feature-map-ordinal"),
        pytest.param(
            {"numerical_method": "ple", "target_aware": True, "placement_strategy": "cart"},
            id="target-aware-ple",
        ),
        pytest.param(
            {"numerical_method": "bspline", "categorical_method": "one-hot", "add_missing_indicator": True},
            id="separate-indicators",
        ),
        pytest.param(
            {"numerical_method": "rbf", "missing_policy": "separate_state"},
            id="separate-state-policy",
        ),
        pytest.param(
            {
                "numerical_method": "rbf",
                "output_dim": 9,
                "adaptive": True,
                "min_output_dim": 3,
                "max_output_dim": 6,
                "target_aware": True,
                "placement_strategy": "cart",
            },
            id="adaptive-widths",
        ),
        pytest.param({"numerical_method": "none"}, id="passthrough"),
    ],
)
def test_fitted_block_layout_agrees_across_views(layout_data, output_format, parameters):
    frame, target = layout_data
    options = {
        "output_dim": 5,
        "adaptive": False,
        "target_aware": False,
        "placement_strategy": "uniform",
        "output_format": output_format,
        "random_state": 0,
        **parameters,
    }
    preprocessor = Preprocessor(**options)
    Pipeline([("preprocessor", preprocessor)]).fit(frame, target)

    _assert_block_layout(preprocessor, frame)
    dimensions = preprocessor.output_dims_
    indicators = options.get("add_missing_indicator", False) or options.get("missing_policy") == "separate_state"
    for feature in ("value", "other"):
        if options["adaptive"]:
            assert 3 <= dimensions[feature] <= 6
            assert dimensions[feature] < options["output_dim"]
        else:
            expected = 1 if options["numerical_method"] == "none" else options["output_dim"]
            has_indicator = indicators and (
                options.get("missing_policy") == "separate_state" or frame[feature].isna().any()
            )
            assert dimensions[feature] == expected + int(has_indicator)
    category_width = frame["city"].nunique() if options.get("categorical_method") == "one-hot" else 1
    assert dimensions["city"] == category_width + int(indicators)
    if indicators:
        blocks = preprocessor.transform(frame, return_array=False)
        assert isinstance(blocks, dict)
        for name, _, _ in get_output_slices(preprocessor.column_transformer_):
            assert isinstance(preprocessor.column_transformer_.named_transformers_[name], FeatureUnion)
        for name, feature in (("num_value", "value"), ("num_other", "other"), ("cat_city", "city")):
            if options.get("missing_policy") == "separate_state" or frame[feature].isna().any():
                np.testing.assert_array_equal(_dense(blocks[name])[:, -1], frame[feature].isna().astype(float))


@pytest.mark.parametrize("typed", [False, True], ids=["mapping", "value"])
@pytest.mark.parametrize("output_format", ["dense", "sparse"])
@pytest.mark.parametrize("missing_policy", [None, "separate_state"])
@pytest.mark.parametrize("sizing", ["fixed", "adaptive", "clamped"])
def test_structured_fitted_widths_agree_across_views(layout_data, typed, output_format, missing_policy, sizing):
    frame, target = layout_data
    requests = {
        "value": {"method": "rbf", "output_dim": 12, "adaptive": True}
        if sizing == "adaptive"
        else {"method": "rbf", "output_dim": 2},
        "other": {"method": "bspline", "output_dim": 3 if sizing == "clamped" else 6, "params": {"degree": 2}},
        "city": {"method": "one-hot", "params": {"drop": "first"}},
    }
    preprocessor = Preprocessor(
        output_dim=12,
        adaptive=True,
        min_output_dim=3,
        max_output_dim=4,
        target_aware=True,
        placement_strategy="cart",
        random_state=17,
        missing_policy=missing_policy,
        output_format=output_format,
        feature_preprocessing={
            feature: RepresentationConfig.from_dict(settings) if typed else settings
            for feature, settings in requests.items()
        },
    )
    with pytest.warns(ConfigWarning) if sizing == "clamped" else nullcontext():
        Pipeline([("preprocessor", preprocessor)]).fit(frame, target)

    _assert_block_layout(preprocessor, frame)
    indicator = int(missing_policy == "separate_state")
    dimensions = preprocessor.output_dims_
    if sizing == "adaptive":
        assert 3 <= dimensions["value"] - indicator <= 4
        assert dimensions["value"] - indicator < requests["value"]["output_dim"]
    else:
        assert dimensions["value"] == 2 + indicator
    assert dimensions["other"] == (5 if sizing == "clamped" else 6) + indicator
    assert dimensions["city"] == frame["city"].nunique() - 1 + indicator

    names = preprocessor.get_feature_names_out()
    transformed = _dense(preprocessor.transform(frame, return_array=True))
    fingerprint = preprocessor.fingerprint_
    report = preprocessor.reproducibility_report()
    spec = json.loads(json.dumps(preprocessor.to_spec(), allow_nan=False))
    representations = {entry["columns"][0]: entry for entry in spec["representations"]}
    assert set(representations) == {"value", "other"}
    for feature, entry in representations.items():
        assert entry["output_dim"] == len(entry["output_features"]) == dimensions[feature] - indicator
    assert spec["feature_names_out"] == list(names)
    assert report["n_features_in"] == frame.shape[1]
    assert report["n_output_features"] == transformed.shape[1] == sum(dimensions.values())
    assert report["fingerprint"] == fingerprint
    assert report["representations"] == {feature: entry["family"] for feature, entry in representations.items()}

    restored = Preprocessor.from_spec(spec)
    _assert_block_layout(restored, frame)
    assert restored.output_dims_ == dimensions
    assert restored.get_feature_info(verbose=False) == preprocessor.get_feature_info(verbose=False)
    assert restored.get_feature_lineage() == preprocessor.get_feature_lineage()
    assert restored.reproducibility_report() == report
    assert restored.fingerprint_ == fingerprint
    np.testing.assert_array_equal(restored.get_feature_names_out(), names)
    np.testing.assert_array_equal(_dense(restored.transform(frame, return_array=True)), transformed)

    assert preprocessor.estimate_output_shape(frame) == transformed.shape
    assert preprocessor.estimate_memory(frame) == transformed.nbytes
    limits = {
        "max_output_features": transformed.shape[1],
        "max_features_per_input": max(dimensions.values()),
        "max_dense_memory": transformed.nbytes,
    }
    bounded = clone(preprocessor)
    assert isinstance(bounded, Preprocessor)
    bounded.set_params(**limits)
    with pytest.warns(ConfigWarning) if sizing == "clamped" else nullcontext():
        bounded.fit(frame, target)
    _assert_block_layout(bounded, frame)
    assert bounded.output_dims_ == dimensions
    np.testing.assert_array_equal(_dense(bounded.transform(frame, return_array=True)), transformed)

    expected_details = {
        "max_output_features": f"total output columns ({transformed.shape[1]})",
        "max_features_per_input": f"feature 'other' expands to {dimensions['other']} columns",
        "max_dense_memory": f"needs ~{transformed.nbytes} bytes",
    }
    for budget, limit in limits.items():
        rejected = clone(preprocessor)
        assert isinstance(rejected, Preprocessor)
        rejected.set_params(**{budget: limit - 1})
        with pytest.raises(OutputBudgetError, match=budget) as error:
            with pytest.warns(ConfigWarning) if sizing == "clamped" else nullcontext():
                rejected.fit(frame, target)
        assert expected_details[budget] in str(error.value)


@pytest.mark.parametrize("output_format", ["dense", "sparse"])
def test_language_embedding_block_layout(layout_data, monkeypatch, output_format):
    frame, _ = layout_data
    model = Mock()
    model.get_sentence_embedding_dimension.return_value = 3
    model.encode.side_effect = lambda values, **kwargs: np.array(
        [[len(value), len(set(value)), sum(map(ord, value))] for value in values], dtype=float
    )
    monkeypatch.setattr(LanguageEmbeddingTransformer, "_resolve_model", lambda self: model)
    preprocessor = Preprocessor(
        numerical_method="none",
        categorical_method="pretrained",
        target_aware=False,
        placement_strategy="uniform",
        output_format=output_format,
    ).fit(frame)

    _assert_block_layout(preprocessor, frame)

    assert preprocessor.output_dims_ == {"value": 1, "other": 1, "city": 3}
    embedding_records = [record for record in preprocessor.get_feature_lineage() if record.source_features == ("city",)]
    assert len(embedding_records) == 3
    assert all(
        record.family == "language_embedding" and record.component == "embedding" for record in embedding_records
    )
    model.encode.assert_called()


@pytest.mark.parametrize("output_format", ["dense", "sparse"])
def test_external_embedding_blocks_are_separate_from_column_layout(layout_data, output_format):
    frame, _ = layout_data
    embeddings = [np.arange(len(frame) * width).reshape(len(frame), width) for width in (2, 4)]
    preprocessor = Preprocessor(
        numerical_method="none",
        target_aware=False,
        placement_strategy="uniform",
        output_structure="blocks",
        output_format=output_format,
    ).fit(frame, embeddings=embeddings)

    _assert_block_layout(preprocessor, frame, embeddings=embeddings)

    blocks = preprocessor.transform(frame, embeddings=embeddings)
    assert isinstance(blocks, dict)
    _, _, metadata = preprocessor.get_feature_info(verbose=False)
    for name, values in zip(("embedding_1", "embedding_2"), embeddings, strict=True):
        assert metadata[name]["dimension"] == values.shape[1]
        np.testing.assert_array_equal(_dense(blocks[name]), values)
    assert preprocessor.total_output_dim_ == 3
    assert sum(np.shape(block)[1] for block in blocks.values()) == preprocessor.total_output_dim_ + 6
    with pytest.raises(IncompatibleParamsError, match="dictionary output"):
        preprocessor.transform(frame, embeddings=embeddings, return_array=True)


@pytest.mark.parametrize("output_format", ["dense", "sparse"])
def test_positive_only_plugin_layout_does_not_probe_transform(positive_only_method, monkeypatch, output_format):
    frame = pd.DataFrame({"value": [1.0, 2.0, 4.0, 8.0], "other": [2.0, 3.0, 5.0, 7.0]})
    preprocessor = Preprocessor(
        numerical_method=positive_only_method,
        numerical_imputation=None,
        target_aware=False,
        placement_strategy="uniform",
        output_format=output_format,
    )
    preprocessor.set_params(scaling=None)
    preprocessor.fit(frame)
    _assert_block_layout(preprocessor, frame)
    expected_names = preprocessor.get_feature_names_out()
    expected_lineage = preprocessor.get_feature_lineage()
    reject_transform = Mock(side_effect=AssertionError("Layout inspection must not invoke transform"))
    for pipeline in preprocessor.column_transformer_.named_transformers_.values():
        monkeypatch.setattr(pipeline.named_steps[positive_only_method], "transform", reject_transform)

    numerical, categorical, embedding_info = preprocessor.get_feature_info(verbose=False)

    assert categorical == embedding_info == {}
    assert preprocessor.output_dims_ == {"value": 1, "other": 1}
    assert [numerical[feature]["dimension"] for feature in frame] == [1, 1]
    assert preprocessor.total_output_dim_ == len(expected_names) == len(expected_lineage) == 2
    assert get_output_slices(preprocessor.column_transformer_) == [("num_value", 0, 1), ("num_other", 1, 1)]
    np.testing.assert_array_equal(preprocessor.get_feature_names_out(), expected_names)
    assert preprocessor.get_feature_lineage() == expected_lineage
    reject_transform.assert_not_called()
