"""Unit tests for :class:`pretab.compose.config.PreprocessorConfig`."""

import inspect
import json
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from pretab import Preprocessor, RepresentationConfig
from pretab.compose.config import FeatureGroup
from pretab.compose.registry import TRANSFORMER_REGISTRY
from pretab.core.parameters import UNSET
from pretab.exceptions import InvalidParamError
from pretab.preprocessor import PRESETS


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize(
    ("preset", "overrides"),
    [
        (None, {}),
        ("standard", {}),
        ("expanded", {"categorical_method": "one-hot", "output_dim": 10}),
        ("adaptive", {"adaptive": True, "max_output_dim": 15}),
    ],
)
def test_resolved_config_matches_inline_preset_contract(task, preset, overrides):
    assert set(PRESETS) == {"standard", "expanded", "adaptive"}
    preprocessor = Preprocessor(preset=preset, task=task)
    requested = preprocessor.get_params(deep=False)
    expected = {
        "numerical_method": "bspline" if preset is not None and task == "regression" else "ple",
        "categorical_method": "int",
        "feature_preprocessing": None,
        "output_dim": 7,
        "degree": 3,
        "target_aware": True,
        "placement_strategy": "cart",
        "task": task,
        "adaptive": False,
        "min_output_dim": 7,
        "max_output_dim": 10,
        "random_state": None,
        "scaling": "minmax",
        "cat_cutoff": 0.03,
        "treat_all_integers_as_numerical": False,
        "numerical_imputation": "median",
        "categorical_imputation": "most_frequent",
        "add_missing_indicator": False,
        "missing_policy": None,
        "policy": None,
        "max_output_features": None,
        "max_features_per_input": None,
        "max_dense_memory": None,
        "overflow_policy": "error",
        "output_structure": "matrix",
        "output_format": "dense",
        "dtype": None,
        "verbose": 0,
        "feature_groups": None,
        **overrides,
    }

    assert preprocessor.get_resolved_config() == expected
    assert preprocessor.get_params(deep=False) == requested
    assert "preset" not in expected


@pytest.mark.parametrize("is_numerical", [False, True])
@pytest.mark.parametrize("configured_imputation", [False, True])
@pytest.mark.parametrize(
    "missing_policy", [None, "error", "propagate", "impute", "impute_with_indicator", "separate_state"]
)
def test_imputation_plan_routes_policy_and_feature_kind(
    make_config, is_numerical, configured_imputation, missing_policy
):
    config = make_config(
        numerical_imputation="mean" if configured_imputation else None,
        categorical_imputation="constant" if configured_imputation else None,
        add_missing_indicator=True,
        missing_policy=missing_policy,
    )
    plans = {
        None: (configured_imputation, configured_imputation, not configured_imputation),
        "error": (False, False, False),
        "propagate": (False, False, False),
        "impute": (True, False, False),
        "impute_with_indicator": (True, True, False),
        "separate_state": (True, False, True),
    }
    add_imputer, add_indicator, separate_state = plans[missing_policy]
    numerical_strategy = "mean" if configured_imputation else "median"
    categorical_strategy = "constant" if configured_imputation else "most_frequent"

    assert config.imputation_plan(is_numerical=is_numerical) == {
        "add_imputer": add_imputer,
        "add_indicator": add_indicator,
        "separate_state": separate_state,
        "strategy": numerical_strategy if is_numerical else categorical_strategy,
    }


def test_normalized_config_retains_every_representation_setting(make_config):
    parameters = {
        "numerical_method": "cubic",
        "categorical_method": "ohe",
        "feature_preprocessing": {"age": "rbf", "city": "int"},
        "output_dim": 11,
        "degree": 2,
        "target_aware": True,
        "placement_strategy": "cart",
        "task": "classification",
        "adaptive": True,
        "min_output_dim": 3,
        "max_output_dim": 8,
        "random_state": 17,
        "scaling": "standardization",
        "cat_cutoff": 6,
        "treat_all_integers_as_numerical": True,
        "numerical_imputation": "mean",
        "categorical_imputation": "constant",
        "add_missing_indicator": True,
        "missing_policy": "separate_state",
        "verbose": 2,
    }

    config = make_config(**parameters)

    assert asdict(config) == {
        **parameters,
        "numerical_method": "cubicspline",
        "categorical_method": "one-hot",
        "feature_groups": (),
    }
    assert config.feature_preprocessing is not parameters["feature_preprocessing"]


@pytest.mark.parametrize("form", ["mapping", "sequence", "values"])
def test_group_normalization_preserves_declaration_and_source_order(make_config, form):
    declarations = [
        {"name": "second", "columns": [1, "latitude"], "representation": {"method": "thinplate", "n_components": 4}},
        {"name": "first", "columns": ["b", "a"], "representation": "tensor"},
    ]
    requested = {
        "mapping": {
            item["name"]: {key: value for key, value in item.items() if key != "name"} for item in declarations
        },
        "sequence": declarations,
        "values": [FeatureGroup(**item) for item in declarations],
    }[form]
    config = make_config(feature_groups=requested)

    assert isinstance(config.feature_groups, tuple)
    assert [group.name for group in config.feature_groups] == ["second", "first"]
    assert [group.columns for group in config.feature_groups] == [(1, "latitude"), ("b", "a")]
    assert [group.representation for group in config.feature_groups] == [
        RepresentationConfig("tprs", params={"n_components": 4}),
        RepresentationConfig("tensorspline"),
    ]
    declarations[0]["columns"].append("changed")
    assert config.feature_groups[0].columns == (1, "latitude")


@pytest.mark.parametrize("requested", [None, {}, [], ()])
def test_empty_groups_normalize_to_an_empty_tuple(make_config, requested):
    assert make_config(feature_groups=requested).feature_groups == ()


def test_group_representation_uses_existing_inheritance_and_local_priority(make_config):
    config = make_config(
        feature_groups={
            "surface": {"columns": ["b", "a"], "representation": {"method": "tensor", "output_dim": 4, "degree": 2}},
            "location": {"columns": ["x", "y"], "representation": {"method": "thinplate", "n_components": 6}},
        },
        output_dim=9,
        degree=3,
        random_state=17,
    )

    surface, location = [config.group_representation_for(group) for group in config.feature_groups]
    assert surface == RepresentationConfig(
        "tensorspline", output_dim=4, placement_strategy="uniform", params={"degree": 2}
    )
    assert location == RepresentationConfig("tprs", params={"n_components": 6, "random_state": 17})


@pytest.mark.parametrize(
    "requested",
    [
        "location",
        {"x", "y"},
        ["x", "y"],
        {"location": {"columns": ["x", "y"]}},
        {"location": {"columns": ["x", "y"], "representation": "tprs", "name": "location"}},
        [{"columns": ["x", "y"], "representation": "tprs"}],
        [{"name": "location", "columns": ["x", "y"], "representation": "tprs", "extra": True}],
        {"location": {"columns": ["x", "y"], "representation": "rff"}},
        {"location": {"columns": ["x", "y"], "representation": {"method": "tprs", "n_componnts": 4}}},
        {"location": {"columns": ["x", "y"], "representation": {"method": "tprs", "name": "other"}}},
        [FeatureGroup("location", ["x", "y"], "tprs"), FeatureGroup("location", ["a", "b"], "tensor")],
    ],
)
def test_invalid_group_declarations_raise_contextual_errors(make_config, requested):
    with pytest.raises(InvalidParamError, match="feature_groups"):
        make_config(feature_groups=requested)


def test_group_parameter_is_last_keyword_only_and_remains_raw_for_sklearn():
    declarations = {
        "location": {"columns": [1, "latitude"], "representation": {"method": "thinplate", "n_components": 4}}
    }
    processor = Preprocessor(feature_groups=declarations, random_state=17)
    parameters = inspect.signature(Preprocessor).parameters

    assert list(parameters)[-1] == "feature_groups"
    assert parameters["feature_groups"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["feature_groups"].default is None
    assert processor.get_params(deep=False)["feature_groups"] is declarations
    resolved = processor.get_resolved_config()
    assert resolved["feature_groups"] == [
        {
            "name": "location",
            "columns": [1, "latitude"],
            "representation": RepresentationConfig("tprs", params={"n_components": 4, "random_state": 17}).to_dict(),
        }
    ]
    resolved["feature_groups"][0]["columns"].append("changed")
    resolved["feature_groups"][0]["representation"]["params"]["n_components"] = 8
    assert processor.get_resolved_config()["feature_groups"][0]["representation"]["params"]["n_components"] == 4
    declarations["location"]["representation"]["n_components"] = 6
    assert processor.get_resolved_config()["feature_groups"][0]["representation"]["params"]["n_components"] == 6
    assert processor.feature_groups is declarations


@pytest.mark.parametrize("typed", [False, True])
def test_group_requests_clone_and_set_params_without_normalizing_constructor_values(typed):
    requested = (
        [FeatureGroup("location", ["x", "y"], "thinplate")]
        if typed
        else {"location": {"columns": ["x", "y"], "representation": "thinplate"}}
    )
    processor = Preprocessor(feature_groups=requested)
    copied = clone(processor)
    assert isinstance(copied, Preprocessor)
    assert copied.feature_groups == requested
    assert copied.feature_groups is not requested
    assert copied.get_resolved_config() == processor.get_resolved_config()
    processor.set_params(feature_groups=None)
    assert processor.feature_groups is None
    assert processor.get_resolved_config()["feature_groups"] is None


@pytest.mark.parametrize("requested", [None, {}, [], ()])
def test_empty_groups_preserve_single_column_values_names_and_fingerprint(sample_frame, requested):
    parameters = {"numerical_method": "rbf", "target_aware": False, "placement_strategy": "uniform"}
    reference = Preprocessor(**parameters).fit(sample_frame)
    processor = Preprocessor(feature_groups=requested, **parameters).fit(sample_frame)

    np.testing.assert_array_equal(processor.transform(sample_frame), reference.transform(sample_frame))
    np.testing.assert_array_equal(processor.get_feature_names_out(), reference.get_feature_names_out())
    assert processor.output_dims_ == reference.output_dims_
    assert processor.fingerprint_ == reference.fingerprint_
    assert processor.feature_groups is requested
    expected_config = reference.get_resolved_config()
    expected_config["feature_groups"] = None if requested is None else []
    assert processor.get_resolved_config() == expected_config


def test_nonempty_groups_cannot_be_silently_ignored_by_single_column_fitting(sample_frame, monkeypatch):
    def reject_construction(*args, **kwargs):
        raise AssertionError("Group refusal must precede factory construction and estimator fitting")

    monkeypatch.setattr("pretab.preprocessor.build_column_transformer", reject_construction)
    processor = Preprocessor(feature_groups={"location": {"columns": ["age", "second"], "representation": "tprs"}})
    with pytest.raises(InvalidParamError, match="single-column composition route"):
        processor.fit(sample_frame.assign(second=sample_frame["age"] + 1))
    assert not hasattr(processor, "column_transformer_")


def test_none_method_normalizes_to_none(make_config):
    cfg = make_config(numerical_method=None, categorical_method=None)
    assert cfg.numerical_method == "none"
    assert cfg.categorical_method == "none"


def test_aliases_resolve_to_canonical(make_config):
    assert make_config(numerical_method="cubic").numerical_method == "cubicspline"
    assert make_config(categorical_method="ohe").categorical_method == "one-hot"


def test_invalid_placement_combo_raises(make_config):
    with pytest.raises(InvalidParamError):
        make_config(target_aware=True, placement_strategy="uniform")
    with pytest.raises(InvalidParamError):
        make_config(target_aware=False, placement_strategy="cart")


@pytest.mark.parametrize("task", ["Regression", "classif", None])
def test_invalid_task_raises(make_config, task):
    with pytest.raises(InvalidParamError, match=r"Preprocessor\.task = .* is invalid"):
        make_config(task=task)


def test_feature_preprocessing_is_copied(make_config):
    fp = {"age": "standardization"}
    cfg = make_config(feature_preprocessing=fp)
    assert cfg.feature_preprocessing == fp
    assert cfg.feature_preprocessing is not fp


def test_none_feature_preprocessing_becomes_empty_dict(make_config):
    assert make_config(feature_preprocessing=None).feature_preprocessing == {}


def test_method_for_override_wins_over_global(make_config):
    cfg = make_config(numerical_method="standardization", feature_preprocessing={"age": "minmax"})
    assert cfg.method_for("age", is_numerical=True) == "minmax"
    assert cfg.method_for("height", is_numerical=True) == "standardization"


def test_method_for_resolves_in_the_requested_namespace(make_config):
    cfg = make_config(feature_preprocessing={"c": "ohe"})
    assert cfg.method_for("c", is_numerical=False) == "one-hot"


def test_seed_kwargs_reflects_random_state(make_config):
    assert make_config(random_state=None).seed_kwargs == {}
    assert make_config(random_state=42).seed_kwargs == {"random_state": 42}


@pytest.mark.parametrize("typed", [False, True])
def test_structured_overrides_detach_and_keep_string_method_lookup(make_config, typed):
    data = {"method": "RBF", "output_dim": 3, "gamma": 0.5}
    value = RepresentationConfig.from_dict(data) if typed else data
    config = make_config(feature_preprocessing={"age": value}, adaptive=True)
    data["output_dim"] = 9
    requested = config.feature_preprocessing["age"]
    resolved = config.representation_for("age", is_numerical=True)

    assert requested == RepresentationConfig(method="RBF", output_dim=3, params={"gamma": 0.5})
    assert config.method_for("age", is_numerical=True) == "rbf"
    assert resolved.output_dim == 3
    assert resolved.adaptive is False
    assert resolved.min_output_dim is UNSET
    assert resolved.max_output_dim is UNSET
    assert resolved.params == {"gamma": 0.5, "task": "regression"}
    assert requested.target_aware is UNSET


@pytest.mark.parametrize("local", [{"adaptive": True}, {}])
def test_representation_resolution_inherits_applicable_defaults(make_config, local):
    config = make_config(
        feature_preprocessing={"age": {"method": "cubic", **local}},
        output_dim=11,
        degree=2,
        adaptive=True,
        min_output_dim=3,
        max_output_dim=15,
        random_state=17,
    )
    resolved = config.representation_for("age", is_numerical=True)
    assert resolved == RepresentationConfig(
        method="cubicspline",
        output_dim=11,
        target_aware=False,
        placement_strategy="uniform",
        adaptive=True,
        min_output_dim=3,
        max_output_dim=15,
        params={"degree": 2, "task": "regression", "random_state": 17},
    )


def test_representation_resolution_keeps_explicit_adaptive_width_and_bounds(make_config):
    config = make_config(
        feature_preprocessing={
            "age": {"method": "rbf", "output_dim": 3, "adaptive": True, "min_output_dim": None, "max_output_dim": 12}
        }
    )
    resolved = config.representation_for("age", is_numerical=True)
    assert (resolved.output_dim, resolved.adaptive, resolved.min_output_dim, resolved.max_output_dim) == (
        3,
        True,
        None,
        12,
    )


@pytest.mark.parametrize("method", ["ohe", "ordinal", "identity"])
def test_representation_resolution_filters_unrelated_constructor_defaults(make_config, method):
    config = make_config(
        feature_preprocessing={"city": {"method": method}}, target_aware=True, placement_strategy="cart", adaptive=True
    )
    resolved = config.representation_for("city", is_numerical=False)
    assert resolved == RepresentationConfig(method=config.method_for("city", is_numerical=False))


def test_representation_resolution_rejects_effective_placement_conflict(make_config):
    config = make_config(
        target_aware=True,
        placement_strategy="cart",
        feature_preprocessing={"age": {"method": "rbf", "target_aware": False}},
    )
    with pytest.raises(InvalidParamError, match=r"feature_preprocessing\['age'\].placement_strategy"):
        config.representation_for("age", is_numerical=True)


def test_representation_resolution_rejects_explicit_bounds_in_fixed_mode(make_config):
    config = make_config(feature_preprocessing={"age": {"method": "rbf", "min_output_dim": 2}})
    with pytest.raises(InvalidParamError, match=r"feature_preprocessing\['age'\].min_output_dim"):
        config.representation_for("age", is_numerical=True)


@pytest.mark.parametrize("form", ["string", "mapping", "value"])
@pytest.mark.parametrize(
    ("method", "is_numerical"),
    [("RBF", True), ("cubic", True), ("b-spline", True), ("OHE", False), ("identity", False)],
)
def test_representation_resolution_forms_are_equivalent(make_config, form, method, is_numerical):
    requested = {"string": method, "mapping": {"method": method}, "value": RepresentationConfig(method=method)}[form]
    config = make_config(feature_preprocessing={"feature": requested})
    expected = make_config(feature_preprocessing={"feature": method}).representation_for(
        "feature", is_numerical=is_numerical
    )
    assert config.representation_for("feature", is_numerical=is_numerical) == expected
    assert config.method_for("feature", is_numerical=is_numerical) == expected.method


@pytest.mark.parametrize("form", ["string", "mapping", "value"])
@pytest.mark.parametrize(("method", "is_numerical"), [("RBF", True), ("OHE", False)])
def test_representation_forms_agree_through_clone_json_report_and_fit(form, method, is_numerical):
    requested = {"string": method, "mapping": {"method": method}, "value": RepresentationConfig(method=method)}[form]
    overrides = {"feature": requested}
    settings: dict[str, Any] = {
        "output_dim": 7,
        "target_aware": False,
        "placement_strategy": "uniform",
        "scaling": "none",
        "random_state": 17,
    }
    values = np.linspace(0, 1, 40) if is_numerical else ["a", "b"] * 20
    frame = pd.DataFrame({"feature": values})
    processor = Preprocessor(feature_preprocessing=overrides, **settings)
    fresh = clone(processor)
    assert isinstance(fresh, Preprocessor)
    assert fresh.get_params() == processor.get_params()
    assert processor.get_params()["feature_preprocessing"] is overrides
    assert overrides["feature"] is requested

    report = fresh.get_resolved_config()
    restored_report = json.loads(json.dumps(report, allow_nan=False))
    assert restored_report == report
    resolved = RepresentationConfig.from_dict(restored_report["feature_preprocessing"]["feature"])
    assert resolved.to_dict() == report["feature_preprocessing"]["feature"]

    actual = np.asarray(fresh.fit_transform(frame, return_array=True))
    reference = Preprocessor(feature_preprocessing={"feature": method}, **settings).fit(frame)
    expected = np.asarray(reference.transform(frame, return_array=True))
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
    np.testing.assert_array_equal(fresh.get_feature_names_out(), reference.get_feature_names_out())
    assert fresh.output_dims_ == reference.output_dims_
    assert fresh.total_output_dim_ == reference.total_output_dim_


@pytest.mark.parametrize("typed", [False, True], ids=["raw", "typed"])
def test_fit_boundary_detaches_params_without_changing_requested_identity(typed):
    params = {"gamma": 0.5}
    settings = {"method": "rbf", "output_dim": 3, "params": params}
    requested = RepresentationConfig.from_dict(settings) if typed else settings
    overrides = {"age": requested}
    processor = Preprocessor(
        feature_preprocessing=overrides, target_aware=False, placement_strategy="uniform", scaling="none"
    )
    params["gamma"] = 0.75
    expected_gamma = 0.5 if typed else 0.75
    frame = pd.DataFrame({"age": np.linspace(0, 1, 40)})
    processor.fit(frame)
    original = np.asarray(processor.transform(frame, return_array=True))
    basis = processor.column_transformer_.named_transformers_["num_age"].named_steps["rbf"]
    assert basis.gamma == expected_gamma

    report = processor.get_resolved_config()
    report["feature_preprocessing"]["age"]["params"]["gamma"] = 9.0
    params["gamma"] = 1.25

    assert processor.get_params()["feature_preprocessing"] is overrides
    assert overrides["age"] is requested
    assert basis.gamma == expected_gamma
    assert np.asarray(processor.transform(frame, return_array=True)).tobytes() == original.tobytes()
    live = processor.get_resolved_config()["feature_preprocessing"]["age"]
    assert live["params"]["gamma"] == (0.5 if typed else 1.25)


def test_normalized_feature_requests_snapshot_nested_values_at_the_boundary(make_config, monkeypatch):
    spec = TRANSFORMER_REGISTRY["rbf"]
    monkeypatch.setitem(TRANSFORMER_REGISTRY, "rbf", replace(spec, allowed_args=(*spec.allowed_args, "options")))
    nested = {"values": [1, {"weight": 0.5}]}
    raw = {"age": {"method": "rbf", "params": {"options": nested}}}
    preprocessor = Preprocessor(feature_preprocessing=raw)
    assert preprocessor.get_params(deep=False)["feature_preprocessing"] is raw
    nested["values"].append(2)
    config = make_config(feature_preprocessing=raw)
    nested["values"].append(3)
    raw["age"]["params"]["options"] = {"changed": True}

    requested = config.feature_preprocessing["age"]
    assert requested.params["options"]["values"] == (1, {"weight": 0.5}, 2)
    resolved = config.representation_for("age", is_numerical=True)
    view = resolved.to_dict()
    view["params"]["options"]["values"].append(4)
    assert config.representation_for("age", is_numerical=True) == resolved
    assert requested.params["options"]["values"] == (1, {"weight": 0.5}, 2)


@pytest.mark.parametrize(
    ("data", "location"),
    [
        ({}, "method"),
        ({"method": "rbf", "output_dim": True}, "output_dim"),
        ({"method": "rbf", "gamma": 0.5, "params": {"gamma": 0.5}}, "gamma"),
        ({"method": "rbf", "params": {"output_dim": 3}}, "params"),
        (None, "value"),
    ],
)
def test_structured_override_shape_errors_keep_the_feature_path(make_config, data, location):
    with pytest.raises(InvalidParamError) as failure:
        make_config(feature_preprocessing={"age": data})
    assert "feature_preprocessing['age']" in str(failure.value)
    assert location in str(failure.value)
    with pytest.raises(InvalidParamError) as standalone:
        RepresentationConfig(method="")
    assert str(standalone.value).startswith("RepresentationConfig.method")


def test_structured_override_namespace_validation_waits_for_feature_detection(make_config):
    config = make_config(feature_preprocessing={"city": {"method": "ohe"}})
    assert config.method_for("city", is_numerical=False) == "one-hot"
    with pytest.raises(InvalidParamError, match=r"feature_preprocessing\['city'\].method"):
        config.method_for("city", is_numerical=True)


@pytest.mark.parametrize("method", ["not_a_method", "ordinal"])
def test_legacy_method_lookup_still_defers_invalid_names(make_config, method):
    config = make_config(feature_preprocessing={"age": method})
    assert config.method_for("age", is_numerical=True) == method


def test_representation_resolution_uses_the_global_method_without_an_override(make_config):
    config = make_config(numerical_method="cubic", degree=2)
    resolved = config.representation_for("age", is_numerical=True)
    assert resolved.method == "cubicspline"
    assert resolved.output_dim == 7
    assert resolved.params == {"degree": 2, "task": "regression"}
    assert resolved.adaptive is False
    assert resolved.min_output_dim is UNSET
    assert resolved.max_output_dim is UNSET


@pytest.mark.parametrize("random_state", [None, 0, 17])
def test_representation_resolution_preserves_seed_forwarding(make_config, random_state):
    config = make_config(numerical_method="rbf", random_state=random_state)
    resolved = config.representation_for("age", is_numerical=True)
    assert ("random_state" in resolved.params) == (random_state is not None)
    if random_state is not None:
        assert resolved.params["random_state"] == random_state


def test_explicit_family_options_override_constructor_controls_independently(make_config):
    config = make_config(
        degree=3,
        task="regression",
        random_state=17,
        feature_preprocessing={"age": {"method": "cubic", "degree": 2, "task": "classification", "random_state": None}},
    )
    resolved = config.representation_for("age", is_numerical=True)
    assert resolved.params == {"degree": 2, "task": "classification", "random_state": None}
    assert config.task == "regression"
    assert config.random_state == 17


@pytest.mark.parametrize("local", [{"output_dim": 3}, {"adaptive": False}, {"output_dim": 3, "adaptive": False}])
def test_fixed_local_requests_ignore_inherited_bounds(make_config, local):
    config = make_config(
        adaptive=True,
        min_output_dim=8,
        max_output_dim=5,
        feature_preprocessing={"age": {"method": "rbf", **local}},
    )
    resolved = config.representation_for("age", is_numerical=True)
    assert resolved.adaptive is False
    assert resolved.min_output_dim is UNSET
    assert resolved.max_output_dim is UNSET


@pytest.mark.parametrize("adaptive", [False, True])
def test_local_bounds_alone_do_not_opt_into_adaptive_resolution(make_config, adaptive):
    config = make_config(adaptive=adaptive, feature_preprocessing={"age": {"method": "rbf", "max_output_dim": 12}})
    if adaptive:
        assert config.representation_for("age", is_numerical=True).max_output_dim == 12
    else:
        with pytest.raises(InvalidParamError, match=r"feature_preprocessing\['age'\].max_output_dim"):
            config.representation_for("age", is_numerical=True)


def test_active_inherited_bound_relationships_are_validated(make_config):
    config = make_config(
        adaptive=True, min_output_dim=8, max_output_dim=5, feature_preprocessing={"age": {"method": "rbf"}}
    )
    with pytest.raises(InvalidParamError, match=r"feature_preprocessing\['age'\].max_output_dim"):
        config.representation_for("age", is_numerical=True)


def test_explicit_unsupervised_pair_overrides_supervised_constructor_settings(make_config):
    config = make_config(
        target_aware=True,
        placement_strategy="cart",
        feature_preprocessing={"age": {"method": "rbf", "target_aware": False, "placement_strategy": "quantile"}},
    )
    resolved = config.representation_for("age", is_numerical=True)
    assert resolved.target_aware is False
    assert resolved.placement_strategy == "quantile"


def test_fixed_placement_family_keeps_existing_factory_default_filtering(make_config):
    config = make_config(
        target_aware=True, placement_strategy="cart", feature_preprocessing={"age": {"method": "pspline"}}
    )
    resolved = config.representation_for("age", is_numerical=True)
    assert resolved.output_dim == 7
    assert resolved.params == {"degree": 3}
    assert resolved.placement_strategy is UNSET
    assert resolved.target_aware is UNSET
