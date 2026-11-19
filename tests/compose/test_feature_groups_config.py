"""Pre-fit ownership and input-capability contracts for explicit groups."""

from copy import deepcopy
from dataclasses import replace
from re import escape

import numpy as np
import pandas as pd
import pytest

from pretab import FeatureGroup, Preprocessor, RepresentationConfig
from pretab.compose.config import PreprocessorConfig
from pretab.compose.registry import TRANSFORMER_REGISTRY
from pretab.exceptions import IncompatibleParamsError, InvalidParamError, OptionalDependencyError, PretabDataError


@pytest.fixture
def stop_after_group_preflight(monkeypatch):
    def reached(*args, **kwargs):
        raise RuntimeError("group preflight complete")

    monkeypatch.setattr("pretab.preprocessor.build_column_transformer", reached)


@pytest.fixture
def group_frame():
    return pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [2.0, 3.0, 4.0],
            "z": [4.0, 5.0, 6.0],
            "flag": [True, False, True],
        }
    )


@pytest.mark.parametrize("method", ["tensorspline", "tprs"])
@pytest.mark.parametrize(
    ("columns", "other_groups", "overrides", "expected"),
    [
        (["x", "x"], {}, None, "feature_groups['location'].columns[1]"),
        (
            ["x", "y"],
            {"other": {"columns": ["y", "z"], "representation": "tprs"}},
            None,
            "feature_groups['other'].columns[0]",
        ),
        (["x", "y"], {}, {"x": "tprs"}, "feature_preprocessing['x']"),
        (["x", "absent"], {}, None, "feature_groups['location'].columns[1]"),
        (["x"], {}, None, "feature_groups['location'].columns"),
        (["x", "flag"], {}, None, "feature_groups['location'].columns[1]"),
    ],
    ids=["duplicate", "overlap", "override", "missing", "min-arity", "boolean"],
)
def test_group_conflicts_precede_factory(group_frame, monkeypatch, method, columns, other_groups, overrides, expected):
    def reject_construction(*args, **kwargs):
        raise AssertionError("Group validation must precede factory construction")

    monkeypatch.setattr("pretab.preprocessor.build_column_transformer", reject_construction)
    groups = {"location": {"columns": columns, "representation": method}, **other_groups}
    original = deepcopy(groups)
    processor = Preprocessor(feature_groups=groups, feature_preprocessing=overrides)
    with pytest.raises(InvalidParamError, match=escape(expected)) as failure:
        processor.fit(group_frame)
    assert "location" in str(failure.value)
    assert groups == original
    assert not hasattr(processor, "column_transformer_")
    assert not hasattr(processor, "numerical_features_")


@pytest.mark.parametrize("method", ["tensor", "thinplate"])
@pytest.mark.parametrize("mapping_form", [False, True])
def test_valid_groups_keep_declared_order_and_unclaimed_sources(make_config, group_frame, method, mapping_form):
    declaration = FeatureGroup("location", ["z", "x"], method)
    groups = {"location": {"columns": ["z", "x"], "representation": method}} if mapping_form else [declaration]
    config = make_config(feature_groups=groups)
    original = config.feature_groups
    before = group_frame.copy(deep=True)
    assert config.validate_feature_groups(group_frame.columns, ["x", "y", "z"]) is None
    assert config.feature_groups is original
    assert original[0].columns == ("z", "x")
    assert original[0].representation.method == {"tensor": "tensorspline", "thinplate": "tprs"}[method]
    assert isinstance(declaration.representation, RepresentationConfig)
    assert declaration.representation.method == method
    pd.testing.assert_frame_equal(group_frame, before)


@pytest.mark.parametrize(
    ("bounds", "columns", "constraint"),
    [
        ({"min_arity": 3}, ["x", "y"], "at least 3"),
        ({"max_arity": 2}, ["x", "y", "z"], "2 to 2"),
    ],
)
def test_group_arity_uses_live_registry_bounds(group_frame, monkeypatch, bounds, columns, constraint):
    spec = TRANSFORMER_REGISTRY["tprs"]
    monkeypatch.setitem(TRANSFORMER_REGISTRY, "tprs", replace(spec, **bounds))
    processor = Preprocessor(feature_groups={"location": {"columns": columns, "representation": "tprs"}})
    with pytest.raises(InvalidParamError, match=escape("feature_groups['location'].columns")) as failure:
        processor.fit(group_frame)
    assert constraint in str(failure.value)


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([True, False, True]),
        pd.Series([True, None, False], dtype="boolean"),
        pd.Series(["a", "b", "a"], dtype="string"),
        pd.Series([1.0, 2.0, 1.0], dtype="category"),
        pd.Series([1, 1, 2]),
        pd.Series([1, None, 2], dtype="Int64"),
    ],
    ids=["boolean", "nullable-boolean", "string", "category", "low-cardinality-integer", "nullable-integer"],
)
def test_categorical_sources_are_not_coerced_to_numerical(group_frame, values):
    processor = Preprocessor(
        cat_cutoff=3,
        feature_groups={"location": {"columns": ["x", "source"], "representation": "tprs"}},
    )
    with pytest.raises(InvalidParamError, match=escape("feature_groups['location'].columns[1]")) as failure:
        processor.fit(group_frame.assign(source=values))
    assert "detected categorical" in str(failure.value)
    assert "'source'" in str(failure.value)


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_explicit_integer_routing_remains_available_to_groups(group_frame, dtype, stop_after_group_preflight):
    processor = Preprocessor(
        cat_cutoff=3,
        treat_all_integers_as_numerical=True,
        feature_groups={"location": {"columns": ["x", "code"], "representation": "tprs"}},
    )
    with pytest.raises(RuntimeError, match="group preflight complete"):
        processor.fit(group_frame.assign(code=pd.Series([1, 1, 2], dtype=dtype)))


@pytest.mark.parametrize(
    ("named", "columns", "missing_index"),
    [
        (True, [7, 3], None),
        (True, ["7", "3"], 0),
        (True, [0, 1], 0),
        (False, ["feature_1", "feature_0"], None),
        (False, [1, 0], 0),
        (False, ["feature_0", "absent"], 1),
    ],
)
def test_group_sources_follow_existing_label_route(named, columns, missing_index, stop_after_group_preflight):
    values = np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]])
    data = pd.DataFrame(values, columns=pd.Index([7, 3])) if named else values
    before = data.copy()
    processor = Preprocessor(feature_groups={"location": {"columns": columns, "representation": "tprs"}})
    expected = (
        "group preflight complete"
        if missing_index is None
        else escape(f"feature_groups['location'].columns[{missing_index}]")
    )
    error = RuntimeError if missing_index is None else InvalidParamError
    with pytest.raises(error, match=expected):
        processor.fit(data)
    if named:
        pd.testing.assert_frame_equal(data, before)
    else:
        np.testing.assert_array_equal(data, before)


def test_groups_do_not_bypass_existing_string_label_collision_validation():
    data = pd.DataFrame([[0.0, 1.0], [2.0, 3.0]], columns=pd.Index([1, "1"], dtype=object))
    processor = Preprocessor(feature_groups={"location": {"columns": [1, "1"], "representation": "tprs"}})
    with pytest.raises(PretabDataError, match="converted to strings"):
        processor.fit(data)
    assert list(data.columns) == [1, "1"]


@pytest.mark.parametrize("groups", [None, {}, [], ()])
def test_empty_groups_do_not_enter_group_preflight(group_frame, monkeypatch, groups):
    def unexpected(*args, **kwargs):
        pytest.fail("The no-groups route must not call group preflight")

    monkeypatch.setattr(PreprocessorConfig, "validate_feature_groups", unexpected)
    processor = Preprocessor(numerical_method="standardization", categorical_method="int", feature_groups=groups).fit(
        group_frame
    )
    assert processor.feature_groups is groups


@pytest.mark.parametrize("usage", ["required", "optional"])
@pytest.mark.parametrize("has_target", [False, True])
def test_group_target_requirements_keep_typed_errors(
    group_frame, monkeypatch, usage, has_target, stop_after_group_preflight
):
    def unexpected(*args, **kwargs):
        pytest.fail("Group target validation must not construct a transformer")

    spec = TRANSFORMER_REGISTRY["tprs"]
    monkeypatch.setitem(
        TRANSFORMER_REGISTRY,
        "tprs",
        replace(spec, target_usage=usage, placement_strategies=frozenset({"cart"}), transformer_cls=unexpected),
    )
    processor = Preprocessor(
        feature_groups={
            "location": {
                "columns": ["x", "y"],
                "representation": {"method": "tprs", "target_aware": True, "placement_strategy": "cart"},
            }
        }
    )
    target = np.array([0.0, 1.0, 2.0]) if has_target else None
    if has_target:
        with pytest.raises(RuntimeError, match="group preflight complete"):
            processor.fit(group_frame, target)
    else:
        with pytest.raises(IncompatibleParamsError, match=escape("feature_groups['location'].representation.method")):
            processor.fit(group_frame)
    assert not hasattr(processor, "column_transformer_")


@pytest.mark.parametrize("local_off", [False, True])
def test_group_target_check_uses_resolved_settings(group_frame, monkeypatch, local_off, stop_after_group_preflight):
    spec = TRANSFORMER_REGISTRY["tprs"]
    monkeypatch.setitem(
        TRANSFORMER_REGISTRY,
        "tprs",
        replace(spec, target_usage="optional", placement_strategies=frozenset({"uniform", "cart"})),
    )
    representation = {"method": "tprs", "target_aware": False, "placement_strategy": "uniform"} if local_off else "tprs"
    processor = Preprocessor(
        target_aware=True,
        placement_strategy="cart",
        feature_groups={"location": {"columns": ["x", "y"], "representation": representation}},
    )
    error = RuntimeError if local_off else IncompatibleParamsError
    expected = "group preflight complete" if local_off else "requires y at fit time"
    with pytest.raises(error, match=expected):
        processor.fit(group_frame)


@pytest.mark.parametrize("available", [False, True])
def test_group_dependency_check_is_contextual_without_import_or_construction(
    group_frame, monkeypatch, available, stop_after_group_preflight
):
    discovered = []

    def discover(module_name):
        discovered.append(module_name)
        return object() if available else None

    def unexpected(*args, **kwargs):
        pytest.fail("Group validation must not import optional packages or construct a transformer")

    spec = TRANSFORMER_REGISTRY["tprs"]
    monkeypatch.setitem(
        TRANSFORMER_REGISTRY, "tprs", replace(spec, optional_dependency="embeddings", transformer_cls=unexpected)
    )
    monkeypatch.setattr("pretab.core.dependencies.importlib.util.find_spec", discover)
    monkeypatch.setattr("pretab.core.dependencies.importlib.import_module", unexpected)
    processor = Preprocessor(feature_groups={"location": {"columns": ["x", "y"], "representation": "tprs"}})
    if available:
        with pytest.raises(RuntimeError, match="group preflight complete"):
            processor.fit(group_frame)
    else:
        with pytest.raises(OptionalDependencyError) as failure:
            processor.fit(group_frame)
        assert "feature_groups['location'].representation.method" in str(failure.value)
        assert "pip install pretab[embeddings]" in str(failure.value)
    assert discovered
    assert set(discovered) == {"sentence_transformers"}


def test_group_placement_dependency_keeps_context_and_install_hint(group_frame, monkeypatch):
    spec = TRANSFORMER_REGISTRY["tprs"]
    monkeypatch.setitem(
        TRANSFORMER_REGISTRY,
        "tprs",
        replace(spec, target_usage="optional", placement_strategies=frozenset({"lightgbm"})),
    )
    monkeypatch.setattr("pretab.core.dependencies.importlib.util.find_spec", lambda module_name: None)
    processor = Preprocessor(
        feature_groups={
            "location": {
                "columns": ["x", "y"],
                "representation": {"method": "tprs", "target_aware": True, "placement_strategy": "lightgbm"},
            }
        }
    )
    with pytest.raises(OptionalDependencyError) as failure:
        processor.fit(group_frame, np.array([0.0, 1.0, 2.0]))
    assert "feature_groups['location'].representation.placement_strategy" in str(failure.value)
    assert "pip install pretab[lightgbm]" in str(failure.value)


def test_builtin_group_preflight_does_not_discover_unselected_extras(
    group_frame, monkeypatch, stop_after_group_preflight
):
    def unexpected(*args, **kwargs):
        pytest.fail("No optional dependency is selected")

    monkeypatch.setattr("pretab.core.dependencies.importlib.util.find_spec", unexpected)
    processor = Preprocessor(feature_groups={"location": {"columns": ["x", "y"], "representation": "tprs"}})
    with pytest.raises(RuntimeError, match="group preflight complete"):
        processor.fit(group_frame)
