"""End-to-end grouped configuration, fitted layout, and replay contracts."""

import json
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from pretab import FeatureGroup, FrozenRepresentationError, OutputBudgetError, Preprocessor, RepresentationConfig
from pretab.compose.inspection import representation_leaf
from pretab.exceptions import DataWarning, PretabDataError
from pretab.transformers import TensorProductSplineTransformer


@pytest.fixture
def frame():
    rng = np.random.default_rng(17)
    return pd.DataFrame(rng.normal(size=(48, 5)), columns=pd.Index(["east", "north", "depth", "time", "single"]))


def _representation(method, width=3):
    if method == "tensorspline":
        return {"method": method, "output_dim": width, "degree": 2}
    return {"method": method, "n_components": width, "landmark_strategy": "subsample", "random_state": 11}


def _processor(groups, **kwargs):
    return Preprocessor(
        feature_groups=groups,
        numerical_method="none",
        categorical_method="one-hot",
        target_aware=False,
        placement_strategy="uniform",
        missing_policy="separate_state",
        **kwargs,
    )


def _groups(typed, reverse=False):
    entries = [
        ("location", ["north", "east"], _representation("tensorspline")),
        ("trajectory", ["time", "depth"], _representation("tprs", width=4)),
    ]
    if reverse:
        entries.reverse()
    if typed:
        return [
            FeatureGroup(name, columns, RepresentationConfig.from_dict(options)) for name, columns, options in entries
        ]
    return {name: {"columns": columns, "representation": options} for name, columns, options in entries}


def _dense(output):
    assert not isinstance(output, dict)
    return output.toarray() if sparse.issparse(output) else np.asarray(output)


@pytest.mark.parametrize("columns", [("north", "east"), ("east", "north"), (20, 10)])
def test_grouped_tensor_typed_penalties_follow_fitted_sources_and_exclude_indicators(frame, columns):
    data = frame[["east", "north", "single"]].copy()
    data.loc[0, "east"] = np.nan
    data.loc[1, "north"] = np.nan
    if isinstance(columns[0], int):
        data = data.rename(columns={"north": 20, "east": 10})
    groups = {"location": {"columns": list(columns), "representation": _representation("tensorspline")}}
    processor = _processor(groups).fit(data)
    restored = Preprocessor.from_spec(json.loads(json.dumps(processor.to_spec(), allow_nan=False)))
    expected = None
    for fitted in (processor, restored):
        leaf = representation_leaf(fitted.column_transformer_.named_transformers_["group_location"])
        assert isinstance(leaf, TensorProductSplineTransformer)
        specs = leaf.get_penalty_specs()
        assert [spec.axis for spec in specs] == [(str(column),) for column in columns]
        assert all(spec.coefficient_names == tuple(leaf.get_feature_names_out()) for spec in specs)
        assert all(spec.coefficient_indices == tuple(range(9)) and spec.matrix.shape == (9, 9) for spec in specs)
        blocks = fitted.transform(data, return_array=False)
        assert isinstance(blocks, dict)
        assert blocks["group_location"].shape[1] == 11
        assert blocks["num_single"].shape[1] == 2
        if expected is not None:
            assert specs == expected
        expected = specs


@pytest.mark.parametrize("typed", [False, True], ids=["mapping", "value"])
@pytest.mark.parametrize("mixed", [False, True], ids=["grouped-only", "mixed"])
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
@pytest.mark.parametrize("output_format", ["dense", "sparse"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_grouped_layout_matches_independent_blocks(frame, typed, mixed, reverse, output_format, dtype):
    data = frame.copy() if mixed else frame.drop(columns="single")
    if mixed:
        data["city"] = np.resize(["Rome", "Lima", "Oslo"], len(data))
    data.loc[0, "east"] = np.nan
    data.loc[1, "time"] = np.nan
    before = data.copy(deep=True)
    groups = _groups(typed, reverse)
    declarations = deepcopy(groups)
    processor = _processor(groups, output_format=output_format, dtype=dtype).fit(data)
    fingerprint = processor.fingerprint_
    actual = processor.transform(data)
    assert not isinstance(actual, dict)
    assert sparse.issparse(actual) == (output_format == "sparse")

    references = []
    if mixed:
        references.append(_processor(None, dtype=dtype).fit(data[["single"]]))
    ordered = ["trajectory", "location"] if reverse else ["location", "trajectory"]
    for name in ordered:
        columns = ["north", "east"] if name == "location" else ["time", "depth"]
        method, width = ("tensorspline", 3) if name == "location" else ("tprs", 4)
        reference = _processor(
            {name: {"columns": columns, "representation": _representation(method, width)}}, dtype=dtype
        )
        references.append(reference.fit(data[columns]))
    if mixed:
        references.append(_processor(None, dtype=dtype).fit(data[["city"]]))

    expected = np.concatenate([_dense(reference.transform(data)) for reference in references], axis=1)
    expected_names = np.concatenate([reference.get_feature_names_out() for reference in references])
    np.testing.assert_array_equal(_dense(actual), expected)
    np.testing.assert_array_equal(processor.get_feature_names_out(), expected_names)
    assert actual.dtype == dtype
    expected_keys = (["num_single"] if mixed else []) + [f"group_{name}" for name in ordered]
    expected_keys += ["cat_city"] if mixed else []
    blocks = processor.transform(data, return_array=False)
    assert isinstance(blocks, dict)
    assert list(blocks) == expected_keys
    offset = 0
    for block in blocks.values():
        width = block.shape[1]
        np.testing.assert_array_equal(_dense(block), expected[:, offset : offset + width])
        offset += width
    assert offset == processor.total_output_dim_ == len(expected_names)
    assert sum(processor.output_dims_.values()) == offset
    lineage = processor.get_feature_lineage()
    assert [record.output_index for record in lineage] == list(range(offset))
    assert [record.output_feature for record in lineage] == list(expected_names)
    for name in ordered:
        records = [record for record in lineage if record.group_id == f"group_{name}"]
        assert records
        sources = ("north", "east") if name == "location" else ("time", "depth")
        assert all(record.source_features == sources for record in records if record.component != "indicator")
    np.testing.assert_array_equal(_dense(processor.transform(data.iloc[:, ::-1])), expected)
    assert processor.fingerprint_ == fingerprint
    assert groups == declarations
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("method", ["tensorspline", "tprs"])
def test_typed_group_spec_round_trip_preserves_fitted_contract(frame, method):
    data = frame[["east", "north"]]
    group = FeatureGroup("location", ["north", "east"], RepresentationConfig.from_dict(_representation(method)))
    processor = _processor([group]).fit(data)
    expected = processor.transform(data)
    fingerprint = processor.fingerprint_

    restored = Preprocessor.from_spec(json.loads(json.dumps(processor.to_spec(), allow_nan=False)))

    np.testing.assert_array_equal(restored.transform(data), expected)
    np.testing.assert_array_equal(restored.get_feature_names_out(), processor.get_feature_names_out())
    assert restored.get_params(deep=False) == processor.get_params(deep=False)
    assert restored.get_feature_info(verbose=False) == processor.get_feature_info(verbose=False)
    assert restored.output_dims_ == processor.output_dims_
    assert restored.fingerprint_ == fingerprint


@pytest.mark.parametrize("method", ["tensorspline", "tprs"])
@pytest.mark.parametrize("typed", [False, True], ids=["mapping", "value"])
@pytest.mark.parametrize("output_structure", ["matrix", "blocks"])
@pytest.mark.parametrize("restore_format", ["spec", "file", "pickle"])
def test_grouped_mixed_replay_preserves_frozen_fitted_state(
    frame, tmp_path, method, typed, output_structure, restore_format
):
    data = frame.assign(city=np.resize(["Rome", "Lima", "Oslo"], len(frame)))
    data.loc[0, "north"] = np.nan
    options = _representation(method)
    groups = (
        [FeatureGroup("location", ["north", "east"], RepresentationConfig.from_dict(options))]
        if typed
        else {"location": {"columns": ["north", "east"], "representation": options}}
    )
    processor = _processor(groups, output_structure=output_structure, output_format="sparse", dtype=np.float32)
    processor.fit(data).freeze()
    expected = processor.transform(data)
    fingerprint = processor.fingerprint_
    if restore_format == "pickle":
        restored = pickle.loads(pickle.dumps(processor))  # noqa: S301
    elif restore_format == "file":
        path = tmp_path / "groups.json"
        processor.to_spec(path)
        restored = Preprocessor.from_spec(path)
    else:
        restored = Preprocessor.from_spec(json.loads(json.dumps(processor.to_spec(), allow_nan=False)))

    actual = restored.transform(data.iloc[:, ::-1])
    if output_structure == "blocks":
        assert isinstance(expected, dict) and isinstance(actual, dict)
        assert list(actual) == list(expected)
        for key in expected:
            assert sparse.issparse(actual[key])
            np.testing.assert_array_equal(_dense(actual[key]), _dense(expected[key]))
            assert actual[key].dtype == np.float32
    else:
        assert sparse.issparse(actual)
        np.testing.assert_array_equal(_dense(actual), _dense(expected))
    assert restored.get_params(deep=False) == processor.get_params(deep=False)
    assert restored.get_resolved_config() == processor.get_resolved_config()
    assert restored.get_feature_info(verbose=False) == processor.get_feature_info(verbose=False)
    assert restored.get_feature_lineage() == processor.get_feature_lineage()
    np.testing.assert_array_equal(restored.get_feature_names_out(), processor.get_feature_names_out())
    assert restored.output_dims_ == processor.output_dims_
    assert restored.output_format_ == "sparse"
    assert restored.output_report_ == processor.output_report_
    assert restored.reproducibility_report() == processor.reproducibility_report()
    assert restored.fingerprint_ == processor.fingerprint_ == fingerprint
    assert restored.is_frozen()
    with pytest.raises(FrozenRepresentationError, match="frozen"):
        restored.set_params(feature_groups=None)
    assert Preprocessor.from_spec(restored.to_spec()).fingerprint_ == fingerprint


@pytest.mark.parametrize("method", ["tensorspline", "tprs"])
@pytest.mark.parametrize("typed", [False, True], ids=["mapping", "value"])
@pytest.mark.parametrize("via_pipeline", [False, True], ids=["direct", "pipeline"])
def test_grouped_clone_and_set_params_refit_without_changing_original(frame, method, typed, via_pipeline):
    options = _representation(method)
    groups = (
        [FeatureGroup("location", ["north", "east"], RepresentationConfig.from_dict(options))]
        if typed
        else {"location": {"columns": ["north", "east"], "representation": options}}
    )
    processor = _processor(groups).fit(frame)
    expected = processor.transform(frame)
    names = processor.get_feature_names_out()
    dimensions = processor.output_dims_.copy()
    lineage = processor.get_feature_lineage()
    original_graph = processor.column_transformer_
    caller = Pipeline([("pre", processor)]) if via_pipeline else processor
    copied = clone(caller)
    assert isinstance(copied, (Pipeline, Preprocessor))
    fresh = copied.named_steps["pre"] if isinstance(copied, Pipeline) else copied
    assert isinstance(fresh, Preprocessor)
    assert fresh.lifecycle_state_ == "UNFITTED"
    assert not hasattr(fresh, "column_transformer_")
    assert fresh.get_params(deep=False) == processor.get_params(deep=False)
    copied.fit(frame)
    np.testing.assert_array_equal(copied.transform(frame), expected)
    assert fresh.column_transformer_ is not original_graph

    replacement = [
        FeatureGroup("location", ["east", "north"], RepresentationConfig.from_dict(_representation(method, 4)))
    ]
    params = {"pre__feature_groups" if via_pipeline else "feature_groups": replacement}
    assert caller.set_params(**params) is caller
    assert processor.feature_groups is replacement
    assert processor.column_transformer_ is original_graph
    np.testing.assert_array_equal(processor.transform(frame), expected)
    np.testing.assert_array_equal(processor.get_feature_names_out(), names)
    assert processor.output_dims_ == dimensions
    assert processor.get_feature_lineage() == lineage
    old_fingerprint = fresh.fingerprint_
    caller.fit(frame)
    reference = _processor(replacement).fit(frame)
    np.testing.assert_array_equal(caller.transform(frame), reference.transform(frame))
    np.testing.assert_array_equal(processor.get_feature_names_out(), reference.get_feature_names_out())
    assert processor.output_dims_ == reference.output_dims_
    assert processor.output_dims_["group_location"] != dimensions["group_location"]
    assert processor.column_transformer_ is not original_graph
    assert processor.fingerprint_ != old_fingerprint
    assert fresh.fingerprint_ == old_fingerprint
    np.testing.assert_array_equal(copied.transform(frame), expected)


@pytest.mark.parametrize("method", ["tensorspline", "tprs"])
@pytest.mark.parametrize("typed", [False, True], ids=["mapping", "value"])
@pytest.mark.parametrize("output_structure", ["matrix", "blocks"])
def test_grouped_budget_uses_full_actual_width_and_remains_reusable(frame, method, typed, output_structure):
    options = _representation(method)
    groups = (
        [FeatureGroup("location", ["north", "east"], RepresentationConfig.from_dict(options))]
        if typed
        else {"location": {"columns": ["north", "east"], "representation": options}}
    )
    processor = _processor(groups, output_structure=output_structure).fit(frame)
    width = processor.output_dims_["group_location"]
    expected = processor.transform(frame, return_array=True)
    processor.set_params(max_features_per_input=width - 1)
    with pytest.raises(OutputBudgetError, match="group_location") as failure:
        processor.fit(frame)
    assert "('north', 'east')" in str(failure.value)
    assert f"expands to {width} columns" in str(failure.value)
    processor.set_params(max_features_per_input=width).fit(frame)
    np.testing.assert_array_equal(processor.transform(frame, return_array=True), expected)
    assert processor.output_dims_["group_location"] == width


@pytest.mark.parametrize("typed", [False, True], ids=["mapping", "value"])
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
@pytest.mark.parametrize("replace_requested", [False, True], ids=["active-groups", "replaced-groups"])
def test_grouped_spec_replays_fitted_order_after_requested_groups_change(frame, typed, reverse, replace_requested):
    processor = _processor(_groups(typed, reverse), output_structure="blocks").fit(frame)
    expected = processor.transform(frame)
    assert isinstance(expected, dict)
    names = processor.get_feature_names_out()
    lineage = processor.get_feature_lineage()
    if replace_requested:
        processor.set_params(feature_groups=None)
    processor.freeze()
    fingerprint = processor.fingerprint_
    restored = Preprocessor.from_spec(json.loads(json.dumps(processor.to_spec(), allow_nan=False)))
    actual = restored.transform(frame)
    assert isinstance(actual, dict)
    assert list(actual) == list(expected)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    np.testing.assert_array_equal(restored.get_feature_names_out(), names)
    assert restored.get_feature_lineage() == lineage
    assert restored.output_dims_ == processor.output_dims_
    assert restored.feature_groups == processor.feature_groups
    assert restored.get_resolved_config() == processor.get_resolved_config()
    assert restored.fingerprint_ == fingerprint


@pytest.mark.parametrize("integer_labels", [False, True], ids=["strings", "integers"])
@pytest.mark.parametrize("empty_positions", [(0,), (1,), (0, 1)], ids=["first", "second", "both"])
def test_grouped_empty_source_failure_names_group_and_ordered_original_sources(frame, integer_labels, empty_positions):
    data = frame[["east", "north"]].copy()
    if integer_labels:
        data.columns = pd.Index([7, 3])
    columns = tuple(data.columns[::-1])
    empty = tuple(columns[position] for position in empty_positions)
    for column in empty:
        data[column] = np.nan
    before = data.copy(deep=True)
    processor = _processor({"location": {"columns": columns, "representation": _representation("tensorspline")}})
    with pytest.warns(DataWarning), pytest.raises(PretabDataError, match="group_location") as failure:
        processor.fit(data)
    message = str(failure.value)
    assert f"with sources {columns!r}" in message
    assert f"for sources {empty!r}" in message
    assert "'tensorspline'" in message
    assert "feature_groups" in message
    assert failure.value.__cause__ is not None
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("integer_labels", [False, True], ids=["strings", "integers"])
@pytest.mark.parametrize("empty_position", [0, 1], ids=["first", "second"])
def test_grouped_empty_source_is_not_blamed_when_observed_stand_in_also_fails(frame, integer_labels, empty_position):
    data = frame[["east", "north"]].copy()
    if integer_labels:
        data.columns = pd.Index([7, 3])
    columns = tuple(data.columns[::-1])
    data[columns[empty_position]] = np.nan
    data[columns[1 - empty_position]] = 5.0
    before = data.copy(deep=True)
    processor = _processor({"location": {"columns": columns, "representation": _representation("tensorspline")}})
    with pytest.warns(DataWarning), pytest.raises(ValueError, match="constant") as failure:
        processor.fit(data)
    assert "has no observed" not in str(failure.value)
    assert "group_location" not in str(failure.value)
    pd.testing.assert_frame_equal(data, before)
