"""Value semantics for immutable, portable representation settings."""

import copy
import json
import pickle
from collections import UserDict
from dataclasses import FrozenInstanceError, is_dataclass
from types import MappingProxyType
from typing import Any, cast

import pytest
from sklearn.base import clone

from pretab import Preprocessor
from pretab.compose.config import RepresentationConfig, _normalize_representation_config
from pretab.core.parameters import UNSET
from pretab.exceptions import InvalidParamError


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
        ("ordinal", True, "ordinal"),
        ("binning", False, "binning"),
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
@pytest.mark.parametrize(("method", "is_numerical", "canonical"), [("RBF", True, "rbf"), ("OHE", False, "one-hot")])
def test_normalization_preserves_explicit_and_omitted_settings(form, method, is_numerical, canonical):
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
def test_normalization_detaches_mapping_at_the_boundary(mapping_type):
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
