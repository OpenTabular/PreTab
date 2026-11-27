"""Value semantics for immutable, portable representation settings."""

import copy
import json
import pickle
from collections import UserDict
from dataclasses import FrozenInstanceError, is_dataclass, replace
from types import MappingProxyType
from typing import Any, cast

import pytest
from sklearn.base import clone

from pretab import Preprocessor
from pretab.compose.config import FeatureGroup, RepresentationConfig, _normalize_representation_config
from pretab.compose.factory import get_categorical_transformer_steps, get_numerical_transformer_steps
from pretab.compose.registry import TRANSFORMER_REGISTRY
from pretab.core.parameters import UNSET
from pretab.exceptions import InvalidParamError, OptionalDependencyError


@pytest.mark.parametrize("form", ["string", "mapping", "value"])
def test_feature_group_value_accepts_existing_representation_forms(form):
    requested = {"string": "tprs", "mapping": {"method": "tprs"}, "value": RepresentationConfig("tprs")}[form]
    group = FeatureGroup("location", [1, "latitude"], requested)

    assert group.columns == (1, "latitude")
    assert group.representation == RepresentationConfig("tprs")
    with pytest.raises(FrozenInstanceError):
        cast(Any, group).name = "replacement"


def test_feature_group_value_detaches_columns_and_nested_settings():
    columns = ["longitude", "latitude"]
    settings = {"method": "tprs", "params": {"options": {"levels": [1, 2]}}}
    group = FeatureGroup("location", columns, settings)
    columns.reverse()
    settings["params"]["options"]["levels"].append(3)

    assert group.columns == ("longitude", "latitude")
    assert isinstance(group.representation, RepresentationConfig)
    assert group.representation.params["options"]["levels"] == (1, 2)
    with pytest.raises(TypeError):
        group.representation.params["options"]["levels"][0] = 9


@pytest.mark.parametrize("name", ["", " ", "_location", "1location", "a-b", "a b", "a__b", "caf\u00e9", None, True, 1])
def test_feature_group_value_rejects_invalid_identifiers(name):
    with pytest.raises(InvalidParamError, match=r"FeatureGroup\.name"):
        FeatureGroup(name, ["x", "y"], "tprs")


@pytest.mark.parametrize(
    "columns", [[], (), "xy", b"xy", {"x", "y"}, {"x": 1}, slice(0, 2), lambda: ["x"], None, [["x"]]]
)
def test_feature_group_value_rejects_unordered_or_invalid_columns(columns):
    with pytest.raises(InvalidParamError, match=r"FeatureGroup\.columns"):
        FeatureGroup("location", columns, "tprs")


@pytest.mark.parametrize(
    "restore",
    [
        copy.copy,
        copy.deepcopy,
        lambda value: pickle.loads(pickle.dumps(value)),  # noqa: S301
        lambda value: clone(value, safe=False),
    ],
)
def test_feature_group_value_preserves_immutable_settings_through_copy_and_pickle(restore):
    group = FeatureGroup("location", [1, "y"], {"method": "tprs", "params": {"options": {"levels": [1, 2]}}})
    restored = restore(group)

    assert restored == group and restored is not group
    assert restored.columns == (1, "y")
    with pytest.raises(TypeError):
        restored.representation.params["options"]["levels"][0] = 9


@pytest.fixture
def nested_registry_option(monkeypatch):
    for method in ("rbf", "cubicspline"):
        spec = TRANSFORMER_REGISTRY[method]
        monkeypatch.setitem(TRANSFORMER_REGISTRY, method, replace(spec, allowed_args=(*spec.allowed_args, "options")))


def test_omitted_fields_remain_unset():
    config = RepresentationConfig(method="rbf")

    assert is_dataclass(config)
    for key in ("output_dim", "target_aware", "placement_strategy", "adaptive", "min_output_dim", "max_output_dim"):
        assert getattr(config, key) is UNSET
    assert config.to_dict() == {"method": "rbf", "params": {}, "name": None}
    assert "RepresentationConfig(method='rbf'" in repr(config)
    with pytest.raises(FrozenInstanceError):
        cast(Any, config).method = "ple"
    with pytest.raises(TypeError):
        hash(config)


def test_nested_values_are_detached_and_read_only():
    original = {"options": {"locations": [1.0, 2.0]}, "labels": ["a", "b"]}
    config = RepresentationConfig(method="rbf", params=original)
    original["options"]["locations"][0] = 9.0
    original["labels"].append("c")
    original["extra"] = True

    assert config.params["options"]["locations"] == (1.0, 2.0)
    assert config.params["labels"] == ("a", "b")
    assert "extra" not in config.params
    with pytest.raises(TypeError):
        cast(Any, config.params)["extra"] = True
    with pytest.raises(TypeError):
        config.params["options"]["extra"] = True
    with pytest.raises(TypeError):
        config.params["options"]["locations"][0] = 9.0


def test_json_round_trip_and_returned_containers_are_detached():
    config = RepresentationConfig(
        method="rbf",
        output_dim=3,
        adaptive=True,
        max_output_dim=None,
        params={"options": {"values": [None, False, 2, 0.5, "x"]}},
        name="age",
    )
    data = config.to_dict()
    assert RepresentationConfig.from_dict(json.loads(json.dumps(data, allow_nan=False))) == config
    data["params"]["options"]["values"].append("changed")
    assert len(config.to_dict()["params"]["options"]["values"]) == 5
    assert config.to_dict()["max_output_dim"] is None


def test_mapping_shorthand_and_order_have_value_equality():
    shorthand = {"method": "RBF", "output_dim": 3, "gamma": 0.5, "params": {"scale": [1, 2]}}
    config = RepresentationConfig.from_dict(shorthand)

    assert config == RepresentationConfig(method="RBF", output_dim=3, params={"gamma": 0.5, "scale": (1, 2)})
    assert config != RepresentationConfig(method="rbf", output_dim=3, params={"gamma": 0.5, "scale": [1, 2]})
    assert config.method == "RBF"
    assert shorthand == {"method": "RBF", "output_dim": 3, "gamma": 0.5, "params": {"scale": [1, 2]}}
    with pytest.raises(KeyError):
        config.params["missing"]


@pytest.mark.parametrize(
    "restore",
    [
        copy.copy,
        copy.deepcopy,
        lambda value: pickle.loads(pickle.dumps(value)),  # noqa: S301
        lambda value: clone(value, safe=False),
    ],
)
def test_copy_pickle_and_clone_preserve_immutable_values(restore):
    config = RepresentationConfig(method="rbf", params={"locations": [{"value": 0.5}]})
    restored = restore(config)

    assert restored == config and restored is not config
    assert restored.output_dim is UNSET
    with pytest.raises(TypeError):
        restored.params["locations"][0]["value"] = 1.0


def test_value_can_be_cloned_as_an_estimator_parameter():
    config = RepresentationConfig(method="rbf", output_dim=3, params={"gamma": 0.5})
    requested = {"age": config}
    original = Preprocessor(feature_preprocessing=requested)
    copied = cast(Preprocessor, clone(original))

    assert original.get_params(deep=False)["feature_preprocessing"] is requested
    assert copied.feature_preprocessing is not None
    assert copied.feature_preprocessing == requested
    assert copied.feature_preprocessing is not requested
    assert copied.feature_preprocessing["age"] is not config


@pytest.mark.parametrize(
    "key",
    [
        "method",
        "output_dim",
        "target_aware",
        "placement_strategy",
        "adaptive",
        "min_output_dim",
        "max_output_dim",
        "params",
        "name",
    ],
)
def test_common_fields_cannot_be_hidden_in_params(key):
    with pytest.raises(InvalidParamError, match="top-level field"):
        RepresentationConfig(method="rbf", params={key: 1})


def test_duplicate_family_options_are_rejected():
    with pytest.raises(InvalidParamError, match="also be supplied inside params"):
        RepresentationConfig.from_dict({"method": "rbf", "gamma": 0.5, "params": {"gamma": 0.5}})


@pytest.mark.parametrize("key", ["method", "placement_strategy", "name"])
@pytest.mark.parametrize("value", ["", "  ", 3, False, []])
def test_invalid_text_fields_raise_typed_errors(key, value):
    with pytest.raises(InvalidParamError, match=key):
        RepresentationConfig(**{"method": "rbf", key: value})


@pytest.mark.parametrize("key", ["output_dim", "min_output_dim", "max_output_dim"])
@pytest.mark.parametrize("value", [0, -1, True, 2.5, "3"])
def test_invalid_counts_raise_typed_errors(key, value):
    with pytest.raises(InvalidParamError, match=key):
        RepresentationConfig(**{"method": "rbf", key: value})


@pytest.mark.parametrize("key", ["target_aware", "adaptive"])
@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_invalid_boolean_fields_raise_typed_errors(key, value):
    with pytest.raises(InvalidParamError, match=key):
        RepresentationConfig(**{"method": "rbf", key: value})


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), b"raw", {1, 2}, object(), lambda: 1, UNSET]
)
def test_opaque_or_nonfinite_nested_values_are_rejected(value):
    with pytest.raises(InvalidParamError, match=r"params\['nested'\]\[0\]"):
        RepresentationConfig(method="rbf", params={"nested": [value]})


def test_cyclic_containers_are_rejected_without_recursion_error():
    values = []
    values.append(values)
    with pytest.raises(InvalidParamError, match="circular references"):
        RepresentationConfig(method="rbf", params={"nested": values})


@pytest.mark.parametrize("data", [None, [], {}, {1: "rbf"}, {"method": "rbf", "params": []}])
def test_invalid_mapping_shape_raises_typed_error(data):
    with pytest.raises(InvalidParamError):
        RepresentationConfig.from_dict(data)


def test_mapping_keys_and_required_values_are_validated():
    for data in [{1: "invalid"}, {"nested": {1: "invalid"}}]:
        with pytest.raises(InvalidParamError, match="keys must be strings"):
            cast(Any, RepresentationConfig)(method="rbf", params=data)
    with pytest.raises(InvalidParamError, match="method"):
        cast(Any, RepresentationConfig)(method=UNSET)
    with pytest.raises(InvalidParamError, match="output_dim"):
        cast(Any, RepresentationConfig)(method="rbf", output_dim=None)
    with pytest.raises(InvalidParamError, match="name"):
        cast(Any, RepresentationConfig)(method="rbf", name=UNSET)
    with pytest.raises(InvalidParamError, match="params"):
        cast(Any, RepresentationConfig)(method="rbf", params=None)


@pytest.mark.parametrize("form", ["string", "mapping", "value"])
@pytest.mark.parametrize(
    ("method", "is_numerical", "canonical"),
    [
        ("rbf", True, "rbf"),
        (" R_B F ", True, "rbf"),
        ("cubic", True, "cubicspline"),
        ("std", True, "standardization"),
        ("binning", True, "custombin"),
        (" O_H E ", False, "one-hot"),
        ("ordinal", False, "int"),
        ("one_hot", False, "one-hot"),
        ("identity", True, "none"),
        ("identity", False, "none"),
    ],
)
def test_normalization_forms_are_equivalent_in_their_namespace(form, method, is_numerical, canonical):
    requested = {"string": method, "mapping": {"method": method}, "value": RepresentationConfig(method=method)}[form]
    normalized = _normalize_representation_config(requested, is_numerical=is_numerical)

    assert normalized == RepresentationConfig(method=canonical)
    assert normalized.output_dim is UNSET
    assert normalized.to_dict() == {"method": canonical, "params": {}, "name": None}


@pytest.mark.parametrize("form", ["mapping", "value"])
@pytest.mark.parametrize(
    ("method", "is_numerical", "canonical"), [("RBF", True, "rbf"), ("Cubic", True, "cubicspline")]
)
def test_normalization_preserves_explicit_and_omitted_settings(
    form, method, is_numerical, canonical, nested_registry_option
):
    data = {
        "method": method,
        "output_dim": 3,
        "target_aware": False,
        "placement_strategy": "uniform",
        "min_output_dim": None,
        "name": "feature",
        "params": {"options": {"values": [None, False, 2, 0.5, "x"]}},
    }
    requested = data if form == "mapping" else RepresentationConfig.from_dict(data)
    normalized = _normalize_representation_config(requested, is_numerical=is_numerical)

    assert normalized.to_dict() == {**data, "method": canonical}
    assert normalized.adaptive is UNSET
    assert normalized.max_output_dim is UNSET
    assert normalized.min_output_dim is None
    assert data["method"] == method
    if isinstance(requested, RepresentationConfig):
        assert requested.method == method
    assert _normalize_representation_config(normalized, is_numerical=is_numerical) == normalized


@pytest.mark.parametrize("mapping_type", [dict, UserDict, MappingProxyType])
def test_normalization_detaches_mapping_at_the_boundary(mapping_type, nested_registry_option):
    data = {"method": "RBF", "gamma": 0.5, "params": {"options": {"values": [1, 2]}}}
    requested = mapping_type(data)
    preprocessor = Preprocessor(feature_preprocessing={"age": requested})
    data["params"]["options"]["values"][0] = 3
    normalized = _normalize_representation_config(requested, is_numerical=True)
    data["params"]["options"]["values"].append(4)

    assert normalized == RepresentationConfig(method="rbf", params={"gamma": 0.5, "options": {"values": [3, 2]}})
    assert preprocessor.get_params(deep=False)["feature_preprocessing"]["age"] is requested
    assert requested["method"] == "RBF"
    assert "gamma" in requested and "gamma" not in requested["params"]
    assert _normalize_representation_config(requested, is_numerical=True).params["options"]["values"] == (3, 2, 4)
    with pytest.raises(TypeError):
        cast(Any, normalized.params)["changed"] = True


@pytest.mark.parametrize("is_numerical", [False, True])
@pytest.mark.parametrize("value", [None, UNSET, [], 1, True, object(), lambda: "rbf"])
def test_normalization_rejects_unsupported_input_types(value, is_numerical):
    with pytest.raises(InvalidParamError, match="must be a string, mapping, or RepresentationConfig"):
        _normalize_representation_config(value, is_numerical=is_numerical)


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"method": None},
        {"method": "rbf", "output_dim": True},
        {"method": "rbf", "params": {"method": "ple"}},
        {"method": "rbf", "gamma": 0.5, "params": {"gamma": 0.5}},
        {"method": "rbf", "params": {"gamma": float("nan")}},
    ],
)
def test_normalization_reuses_value_shape_validation(data):
    with pytest.raises(InvalidParamError):
        _normalize_representation_config(data, is_numerical=True)


@pytest.mark.parametrize("form", ["shorthand", "params", "value"])
@pytest.mark.parametrize(
    ("key", "suggestion"),
    [("output_dimm", "output_dim"), ("target_awre", "target_aware"), ("gama", "gamma"), ("not_a_real_option", None)],
)
def test_path_errors_for_unknown_keys_include_close_matches(form, key, suggestion):
    data: dict[str, Any] = {"method": "RBF"}
    if form == "shorthand":
        data[key] = 3
    else:
        data["params"] = {key: 3}
    requested = RepresentationConfig.from_dict(data) if form == "value" else data
    path = "feature_preprocessing['age']"
    with pytest.raises(InvalidParamError) as failure:
        _normalize_representation_config(requested, is_numerical=True, path=path)

    location = key if form == "shorthand" else f"params[{key!r}]"
    message = str(failure.value)
    assert f"{path}.{location} = 3 is invalid" in message
    if suggestion is None:
        assert "Did you mean" not in message
    else:
        assert "Did you mean" in message
        assert repr(suggestion) in message.split("\nValid values:")[0]
    assert data["method"] == "RBF"


@pytest.mark.parametrize("form", ["string", "mapping", "value"])
@pytest.mark.parametrize(
    ("method", "is_numerical"),
    [("ordinal", True), ("binning", False), ("not_a_method", True), ("not_a_method", False), ("rbf", False)],
)
def test_path_errors_for_methods_reuse_canonical_factory_choices(form, method, is_numerical):
    requested = {"string": method, "mapping": {"method": method}, "value": RepresentationConfig(method=method)}[form]
    path = "feature_preprocessing['feature']"
    with pytest.raises(InvalidParamError) as failure:
        _normalize_representation_config(requested, is_numerical=is_numerical, path=path)
    factory = get_numerical_transformer_steps if is_numerical else get_categorical_transformer_steps
    with pytest.raises(InvalidParamError) as existing:
        factory(method, add_imputer=False)

    assert f"{path}.method" in str(failure.value)
    assert str(failure.value).split("\nConstraint:")[1] == str(existing.value).split("\nConstraint:")[1]


@pytest.mark.parametrize(
    ("data", "location"),
    [
        ({}, "method"),
        ({"method": None}, "method"),
        ({"method": "rbf", "output_dim": 0}, "output_dim"),
        ({"method": "rbf", "output_dim": True}, "output_dim"),
        ({"method": "rbf", "target_aware": 1}, "target_aware"),
        ({"method": "rbf", "placement_strategy": ""}, "placement_strategy"),
        ({"method": "rbf", "adaptive": None}, "adaptive"),
        ({"method": "rbf", "min_output_dim": -1}, "min_output_dim"),
        ({"method": "rbf", "params": []}, "params"),
        ({"method": "rbf", "params": {"method": "ple"}}, "params['method']"),
        ({"method": "rbf", "params": {"gamma": [float("inf")]}}, "params['gamma'][0]"),
        ({"method": "rbf", "gamma": 0.5, "params": {"gamma": 0.5}}, "gamma"),
    ],
)
def test_path_errors_reuse_common_and_nested_value_checks(data, location):
    path = "feature_preprocessing['a.b[0]']"
    with pytest.raises(InvalidParamError) as failure:
        _normalize_representation_config(data, is_numerical=True, path=path)
    assert f"{path}.{location}" in str(failure.value)


@pytest.mark.parametrize("succeeds", [False, True])
def test_path_errors_do_not_leak_into_standalone_value_validation(succeeds):
    if succeeds:
        _normalize_representation_config("rbf", is_numerical=True, path="feature_preprocessing['age']")
    else:
        with pytest.raises(InvalidParamError):
            _normalize_representation_config({"method": "rbf", "output_dim": 0}, is_numerical=True, path="other")
    with pytest.raises(InvalidParamError) as failure:
        RepresentationConfig(method="")
    assert str(failure.value).startswith("RepresentationConfig.method")


def test_path_errors_follow_the_live_registry_and_check_applicability(monkeypatch):
    data = {"method": "rbf", "plugin_strength": 0.5}
    with pytest.raises(InvalidParamError, match="plugin_strength"):
        _normalize_representation_config(data, is_numerical=True)
    spec = TRANSFORMER_REGISTRY["rbf"]
    monkeypatch.setitem(
        TRANSFORMER_REGISTRY, "rbf", replace(spec, allowed_args=(*spec.allowed_args, "plugin_strength"))
    )
    assert _normalize_representation_config(data, is_numerical=True).params["plugin_strength"] == 0.5
    with pytest.raises(InvalidParamError, match="degree"):
        _normalize_representation_config({"method": "ohe", "degree": 3}, is_numerical=False)


@pytest.mark.parametrize(
    ("data", "is_numerical", "location"),
    [
        ({"method": "ohe", "degree": 3}, False, "degree"),
        ({"method": "rbf", "params": {"degree": 3}}, True, "params['degree']"),
        ({"method": "polynomial", "gamma": 1.0}, True, "gamma"),
        ({"method": "ohe", "output_dim": 3}, False, "output_dim"),
        ({"method": "minmax", "adaptive": False}, True, "adaptive"),
        ({"method": "custombin", "adaptive": True}, True, "adaptive"),
        ({"method": "ohe", "min_output_dim": None}, False, "min_output_dim"),
        ({"method": "ple", "target_aware": False}, True, "target_aware"),
        ({"method": "pspline", "target_aware": True}, True, "target_aware"),
        ({"method": "pspline", "placement_strategy": "quantile"}, True, "placement_strategy"),
        ({"method": "ohe", "placement_strategy": "uniform"}, False, "placement_strategy"),
        ({"method": "rbf", "placement_strategy": "even"}, True, "placement_strategy"),
        ({"method": "rbf", "target_aware": False, "placement_strategy": "cart"}, True, "placement_strategy"),
        ({"method": "rbf", "target_aware": True, "placement_strategy": "uniform"}, True, "placement_strategy"),
        ({"method": "rbf", "adaptive": False, "max_output_dim": None}, True, "max_output_dim"),
        ({"method": "rbf", "adaptive": True, "min_output_dim": 5, "max_output_dim": 3}, True, "max_output_dim"),
    ],
)
def test_capability_errors_name_the_explicit_option_and_source(data, is_numerical, location):
    path = "feature_preprocessing['age']"
    with pytest.raises(InvalidParamError) as failure:
        _normalize_representation_config(data, is_numerical=is_numerical, path=path)
    assert f"{path}.{location}" in str(failure.value)


@pytest.mark.parametrize(
    "data",
    [
        {"method": "rbf", "output_dim": 3, "gamma": 0.5},
        {"method": "bspline", "output_dim": 3},
        {"method": "mspline", "output_dim": 5},
        {"method": "ispline", "output_dim": 5},
        {"method": "pspline", "output_dim": 5, "target_aware": False, "placement_strategy": "uniform"},
        {"method": "ple", "target_aware": True, "placement_strategy": "cart", "adaptive": True},
        {"method": "rbf", "placement_strategy": "quantile"},
        {"method": "rbf", "target_aware": False},
        {"method": "rbf", "adaptive": True, "min_output_dim": None, "max_output_dim": None},
    ],
)
def test_capability_validation_preserves_explicit_requests_without_filling_defaults(data):
    normalized = _normalize_representation_config(data, is_numerical=True)
    assert normalized == RepresentationConfig.from_dict(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("feature_kind", frozenset({"categorical"})),
        ("preprocessor_compatible", False),
        ("supports_adaptive_resolution", False),
    ],
)
def test_capability_validation_uses_live_metadata_even_when_derived_views_are_unchanged(monkeypatch, field, value):
    spec = TRANSFORMER_REGISTRY["rbf"]
    monkeypatch.setitem(TRANSFORMER_REGISTRY, "rbf", replace(spec, **{field: value}))
    with pytest.raises(InvalidParamError):
        _normalize_representation_config({"method": "rbf", "adaptive": True}, is_numerical=True)


@pytest.mark.parametrize("method", ["tensor", "thinplate", "rff", "nystroem"])
def test_capability_validation_does_not_make_standalone_methods_selectable(method):
    with pytest.raises(InvalidParamError, match="method"):
        _normalize_representation_config(method, is_numerical=True)


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize(("method", "is_numerical"), [("pretrained", False), ("rbf", True)])
def test_optional_dependency_metadata_is_checked_without_import_or_construction(
    monkeypatch, method, is_numerical, available
):
    discovered = []

    def discover(module_name):
        discovered.append(module_name)
        return object() if available else None

    def unexpected(*args, **kwargs):
        pytest.fail("normalization must not import optional packages or construct a transformer")

    spec = TRANSFORMER_REGISTRY[method]
    monkeypatch.setitem(
        TRANSFORMER_REGISTRY, method, replace(spec, optional_dependency="embeddings", transformer_cls=unexpected)
    )
    monkeypatch.setattr("pretab.core.dependencies.importlib.util.find_spec", discover)
    monkeypatch.setattr("pretab.core.dependencies.importlib.import_module", unexpected)
    path = "feature_preprocessing['city']"
    if available:
        assert _normalize_representation_config(method, is_numerical=is_numerical, path=path).method == method
    else:
        with pytest.raises(OptionalDependencyError) as failure:
            _normalize_representation_config(method, is_numerical=is_numerical, path=path)
        assert f"{path}.method" in str(failure.value)
        assert "pip install pretab[embeddings]" in str(failure.value)
    assert discovered == ["sentence_transformers"]


@pytest.mark.parametrize("available", [False, True])
def test_explicit_lightgbm_placement_uses_import_free_dependency_check(monkeypatch, available):
    discovered = []

    def discover(module_name):
        discovered.append(module_name)
        return object() if available else None

    monkeypatch.setattr("pretab.core.dependencies.importlib.util.find_spec", discover)
    data = {"method": "rbf", "target_aware": True, "placement_strategy": "lightgbm"}
    if available:
        assert _normalize_representation_config(data, is_numerical=True).placement_strategy == "lightgbm"
    else:
        with pytest.raises(OptionalDependencyError, match=r"RepresentationConfig.placement_strategy.*lightgbm"):
            _normalize_representation_config(data, is_numerical=True)
    assert discovered == ["lightgbm"]


def test_capability_validation_does_not_check_unselected_optional_dependencies(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("no optional dependency is selected")

    monkeypatch.setattr("pretab.core.dependencies.importlib.util.find_spec", unexpected)
    assert _normalize_representation_config("rbf", is_numerical=True).target_aware is UNSET
