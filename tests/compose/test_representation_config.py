"""Value semantics for immutable, portable representation settings."""

import copy
import json
import pickle
from dataclasses import FrozenInstanceError, is_dataclass
from typing import Any, cast

import pytest
from sklearn.base import clone

from pretab import Preprocessor
from pretab.compose.config import RepresentationConfig
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
