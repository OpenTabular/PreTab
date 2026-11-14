"""Unit tests for :mod:`pretab.compose.factory`."""

from dataclasses import fields
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline

from pretab import Preprocessor
from pretab.compose.config import PreprocessorConfig, RepresentationConfig
from pretab.compose.factory import (
    _placement_kwargs,
    build_column_transformer,
    create_transformer,
    get_categorical_transformer_steps,
    get_numerical_transformer_steps,
)
from pretab.compose.registry import TransformerSpec, get_spec
from pretab.core.parameters import UNSET
from pretab.exceptions import ConfigWarning, InvalidParamError


def _names(steps):
    return [name for name, _ in steps]


# --------------------------------------------------------------------------- #
# numerical step assembly
# --------------------------------------------------------------------------- #
def test_imputer_is_first_step_by_default():
    assert _names(get_numerical_transformer_steps("standardization"))[0] == "imputer"


def test_imputer_omitted_when_disabled():
    assert "imputer" not in _names(get_numerical_transformer_steps("standardization", add_imputer=False))


def test_imputer_keeps_a_column_without_observed_values():
    empty = np.full((4, 1), np.nan)
    assert Pipeline(get_numerical_transformer_steps("minmax")).fit_transform(empty).shape == (4, 1)
    assert Pipeline(get_categorical_transformer_steps("one-hot")).fit_transform(empty.astype(object)).shape == (4, 1)
    # imputer_kwargs can still restore scikit-learn's default of dropping it.
    steps = dict(get_numerical_transformer_steps("minmax", imputer_kwargs={"keep_empty_features": False}))
    assert steps["imputer"].keep_empty_features is False


def test_none_method_uses_noop_step():
    assert _names(get_numerical_transformer_steps("none", add_imputer=False)) == ["noop"]


def test_box_cox_scales_positive_first():
    assert _names(get_numerical_transformer_steps("box-cox", add_imputer=False)) == ["scale_positive", "boxcox"]


def test_scaling_injected_only_when_different_from_method():
    with_scaler = _names(get_numerical_transformer_steps("ple", add_imputer=False, scaling="standardization"))
    assert "scaler" in with_scaler
    same = _names(get_numerical_transformer_steps("standardization", add_imputer=False, scaling="standardization"))
    assert "scaler" not in same
    assert same.count("standardization") == 1


def test_unknown_scaling_raises():
    """Regression guard for issue #35: an unrecognized scaling must raise, not

    silently produce an unscaled pipeline.
    """
    with pytest.raises(InvalidParamError):
        get_numerical_transformer_steps("ple", add_imputer=False, scaling="not_a_scaler")


def test_scaling_none_disables_scaling():
    for none_spelling in (None, "none"):
        steps = _names(get_numerical_transformer_steps("ple", add_imputer=False, scaling=none_spelling))
        assert "scaler" not in steps
        assert "minmax" not in steps


def test_bmi_spline_output_dim_is_clamped_with_warning():
    with pytest.warns(ConfigWarning):
        get_numerical_transformer_steps("bspline", add_imputer=False, output_dim=100)


def test_unknown_numerical_method_raises():
    with pytest.raises(InvalidParamError):
        get_numerical_transformer_steps("does-not-exist", add_imputer=False)


# --------------------------------------------------------------------------- #
# categorical step assembly
# --------------------------------------------------------------------------- #
def test_one_hot_appends_to_float():
    pipe = Pipeline(get_categorical_transformer_steps("one-hot", add_imputer=False))
    transformed = pipe.fit_transform(np.array([["a"], ["b"]]))
    assert transformed.dtype == np.float64


def test_int_uses_continuous_ordinal():
    assert _names(get_categorical_transformer_steps("int", add_imputer=False)) == ["continuous_ordinal"]


def test_unknown_categorical_method_raises():
    with pytest.raises(InvalidParamError):
        get_categorical_transformer_steps("does-not-exist", add_imputer=False)


def test_explicit_numerical_missing_value_override_is_preserved():
    options = {"missing_values": -1}
    pipe = Pipeline(get_numerical_transformer_steps("none", imputer_kwargs=options))
    result = pipe.fit_transform(np.array([[1.0], [-1.0], [3.0]]))

    np.testing.assert_array_equal(result.ravel(), [1.0, 2.0, 3.0])
    assert options == {"missing_values": -1}


def test_explicit_categorical_missing_value_override_is_preserved():
    options = {"missing_values": "missing"}
    pipe = Pipeline(get_categorical_transformer_steps("int", imputer_kwargs=options))
    result = pipe.fit_transform(np.array([["a"], ["missing"], ["b"]], dtype=object))

    np.testing.assert_array_equal(result.ravel(), [1, 1, 2])
    assert options == {"missing_values": "missing"}


@pytest.mark.parametrize("missing_values", [None, np.nan, pd.NA], ids=["None", "NaN", "pd.NA"])
def test_explicit_imputer_missing_marker_is_authoritative(missing_values):
    options = {"missing_values": missing_values}
    pipe = Pipeline(get_categorical_transformer_steps("none", imputer_kwargs=options))

    result = pipe.fit_transform(np.array([["a"], [None], ["b"], ["a"]], dtype=object))

    if missing_values is np.nan:
        assert result[1, 0] is None
    else:
        assert result[1, 0] == "a"
    assert options["missing_values"] is missing_values


# --------------------------------------------------------------------------- #
# placement kwargs by capability class
# --------------------------------------------------------------------------- #
def test_placement_optional_forwards_target_aware_and_strategy():
    spec = get_spec("rbf")  # target_usage == optional
    assert _placement_kwargs(spec, {"target_aware": False}) == {"target_aware": False}
    assert _placement_kwargs(spec, {"target_aware": False, "placement_strategy": "quantile"}) == {
        "target_aware": False,
        "placement_strategy": "quantile",
    }


def test_placement_required_only_when_target_aware_supervised():
    spec = get_spec("ple")  # target_usage == required
    assert _placement_kwargs(spec, {"target_aware": True, "placement_strategy": "cart"}) == {
        "placement_strategy": "cart"
    }
    assert _placement_kwargs(spec, {"target_aware": False, "placement_strategy": "cart"}) == {}


def test_placement_forbidden_uses_unsupervised_only():
    spec = get_spec("pspline")  # target_usage == forbidden, unsupervised placement
    assert _placement_kwargs(spec, {"target_aware": False, "placement_strategy": "uniform"}) == {
        "placement_strategy": "uniform"
    }
    assert _placement_kwargs(spec, {"target_aware": True, "placement_strategy": "uniform"}) == {}


def test_placement_absent_when_method_has_no_strategies():
    spec = get_spec("standardization")  # no placement strategies
    assert _placement_kwargs(spec, {"target_aware": True, "placement_strategy": "cart"}) == {}


def test_placement_optional_without_strategies_still_forwards_target_aware():
    # e.g. a registered supervision="optional" representation with no placement
    # strategies: target_aware decides whether it uses y, placement does not apply.
    spec = TransformerSpec(name="custom", transformer_cls=object, target_usage="optional")
    assert _placement_kwargs(spec, {"target_aware": False, "placement_strategy": "uniform"}) == {"target_aware": False}
    assert _placement_kwargs(spec, {"target_aware": True, "placement_strategy": "cart"}) == {"target_aware": True}


# --------------------------------------------------------------------------- #
# ColumnTransformer assembly
# --------------------------------------------------------------------------- #
def test_build_column_transformer_prefixes_and_passthrough(make_config):
    ct = build_column_transformer(make_config(), ["age"], ["city"])
    assert isinstance(ct, ColumnTransformer)
    assert [name for name, _, _ in ct.transformers] == ["num_age", "cat_city"]
    assert ct.remainder == "passthrough"


def test_build_column_transformer_fits_and_transforms(make_config, sample_frame):
    ct = build_column_transformer(make_config(), ["age"], ["city"])
    out = ct.fit_transform(sample_frame, np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == len(sample_frame)


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("random_state", [None, 17])
@pytest.mark.parametrize(
    "missing_policy", [None, "error", "propagate", "impute", "impute_with_indicator", "separate_state"]
)
def test_create_transformer_forwards_normalized_numerical_options(make_config, adaptive, random_state, missing_policy):
    config = make_config(
        numerical_method="rbf",
        output_dim=11,
        degree=2,
        target_aware=True,
        placement_strategy="cart",
        task="classification",
        adaptive=adaptive,
        min_output_dim=3,
        max_output_dim=8,
        random_state=random_state,
        scaling="standardization",
        numerical_imputation=None,
        add_missing_indicator=True,
        missing_policy=missing_policy,
    )
    plan = config.imputation_plan(is_numerical=True)

    with patch(
        "pretab.compose.factory.get_numerical_transformer_steps", wraps=get_numerical_transformer_steps
    ) as numerical_steps:
        block = create_transformer(config.numerical_method, is_numerical=True, config=config)

    numerical_steps.assert_called_once_with(
        method="rbf",
        task="classification",
        target_aware=True,
        add_imputer=plan["add_imputer"],
        imputer_strategy="median",
        output_dim=11,
        adaptive=adaptive,
        min_output_dim=3 if adaptive else None,
        max_output_dim=8 if adaptive else None,
        degree=2,
        scaling="standardization",
        placement_strategy="cart",
        **({} if random_state is None else {"random_state": random_state}),
    )
    if plan["separate_state"] or plan["add_indicator"]:
        assert isinstance(block, FeatureUnion)
        representation = dict(block.transformer_list)["representation"]
    else:
        representation = block
    assert isinstance(representation, Pipeline)
    assert ("imputer" in representation.named_steps) is plan["add_imputer"]
    if plan["add_imputer"]:
        assert representation.named_steps["imputer"].strategy == "median"


@pytest.mark.parametrize("preset", [None, "standard", "expanded", "adaptive"])
@pytest.mark.parametrize("missing_policy", [None, "impute_with_indicator", "separate_state"])
def test_preprocessor_routes_one_normalized_config_through_the_factory(sample_frame, preset, missing_policy):
    feature_overrides = {"age": "rbf", "city": "int"}
    preprocessor = Preprocessor(
        preset=preset,
        numerical_method="cubic",
        categorical_method="ohe",
        feature_preprocessing=feature_overrides,
        degree=2,
        target_aware=False,
        placement_strategy="uniform",
        task="classification",
        adaptive=False,
        min_output_dim=3,
        max_output_dim=8,
        random_state=17,
        scaling="standardization",
        cat_cutoff=0.1,
        treat_all_integers_as_numerical=True,
        numerical_imputation="mean",
        categorical_imputation="constant",
        add_missing_indicator=True,
        missing_policy=missing_policy,
        max_output_features=64,
        max_features_per_input=16,
        max_dense_memory=1_000_000,
        output_structure="blocks",
        output_format="sparse",
        dtype="float32",
    )
    requested = preprocessor.get_params(deep=False)
    resolved = preprocessor.get_resolved_config()
    config_fields = {field.name for field in fields(PreprocessorConfig)}
    facade_fields = {
        "policy",
        "max_output_features",
        "max_features_per_input",
        "max_dense_memory",
        "overflow_policy",
        "output_structure",
        "output_format",
        "dtype",
    }
    assert set(resolved) == config_fields | facade_fields
    assert config_fields.isdisjoint(facade_fields)

    normalized_configs: list[PreprocessorConfig] = []
    from_params = PreprocessorConfig.from_params

    def normalize_config(**parameters):
        config = from_params(**parameters)
        normalized_configs.append(config)
        return config

    with (
        patch.object(PreprocessorConfig, "from_params", side_effect=normalize_config) as normalize,
        patch.object(
            PreprocessorConfig, "method_for", autospec=True, side_effect=PreprocessorConfig.method_for
        ) as method_for,
        patch.object(
            PreprocessorConfig, "imputation_plan", autospec=True, side_effect=PreprocessorConfig.imputation_plan
        ) as imputation_plan,
        patch("pretab.compose.factory.create_transformer", wraps=create_transformer) as create,
    ):
        route = Mock()
        for name, spy in (
            ("normalize", normalize),
            ("method", method_for),
            ("imputation", imputation_plan),
            ("create", create),
        ):
            route.attach_mock(spy, name)

        preprocessor.fit(sample_frame, np.array([0, 1, 0, 1, 0, 1]))

        normalize.assert_called_once_with(**{name: resolved[name] for name in config_fields})
        (config,) = normalized_configs
        assert all(item.kwargs["config"] is config for item in create.call_args_list)
        assert all(item.args[0] is config for item in method_for.call_args_list + imputation_plan.call_args_list)
        assert config.numerical_method == "cubicspline"
        assert config.categorical_method == "one-hot"
        assert config.feature_preprocessing == feature_overrides
        assert config.feature_preprocessing is not feature_overrides
        assert method_for.call_args_list == [
            call(config, "age", is_numerical=True),
            call(config, "city", is_numerical=False),
        ]
        assert create.call_args_list == [
            call("rbf", is_numerical=True, config=config),
            call("int", is_numerical=False, config=config),
        ]
        assert imputation_plan.call_args_list == [call(config, is_numerical=True), call(config, is_numerical=False)]
        assert [item[0] for item in route.mock_calls] == [
            "normalize",
            "method",
            "create",
            "imputation",
            "method",
            "create",
            "imputation",
        ]

    for name, strategy in (("num_age", "mean"), ("cat_city", "constant")):
        block = preprocessor.column_transformer_.named_transformers_[name]
        assert isinstance(block, FeatureUnion)
        representation = dict(block.transformer_list)["representation"]
        assert representation.named_steps["imputer"].strategy == strategy
    numerical = dict(preprocessor.column_transformer_.named_transformers_["num_age"].transformer_list)["representation"]
    assert numerical.named_steps["rbf"].output_dim == resolved["output_dim"]
    assert numerical.named_steps["rbf"].random_state == 17
    assert preprocessor.get_params(deep=False) == requested
    assert preprocessor.get_resolved_config() == resolved
    assert feature_overrides == {"age": "rbf", "city": "int"}


@pytest.mark.parametrize("output_dim", [2, 3, 5])
def test_resolved_representation_controls_width_and_family_options(make_config, output_dim):
    config = make_config(
        output_dim=11,
        adaptive=True,
        feature_preprocessing={"age": {"method": "rbf", "output_dim": output_dim, "gamma": 0.5}},
    )
    resolved = config.representation_for("age", is_numerical=True)
    block = create_transformer("rbf", is_numerical=True, config=config, representation=resolved)
    assert isinstance(block, Pipeline)
    transformer = block.named_steps["rbf"]
    assert transformer.output_dim == output_dim
    assert transformer.gamma == 0.5
    assert transformer.adaptive is False
    assert block.fit_transform(np.linspace(0, 1, 16).reshape(-1, 1)).shape == (16, output_dim)
    assert config.output_dim == 11
    assert config.adaptive is True


def test_resolved_representation_target_usage_overrides_run_wide_setting(make_config):
    config = make_config(
        feature_preprocessing={"age": {"method": "ple", "target_aware": True, "placement_strategy": "cart"}}
    )
    resolved = config.representation_for("age", is_numerical=True)
    block = create_transformer("ple", is_numerical=True, config=config, representation=resolved)
    assert isinstance(block, Pipeline)
    assert block.named_steps["ple"].placement_strategy == "cart"
    assert config.target_aware is False


def test_resolved_representation_method_must_match_selected_method(make_config):
    with pytest.raises(InvalidParamError, match=r"create_transformer.representation.method.*rbf"):
        create_transformer(
            "standardization",
            is_numerical=True,
            config=make_config(),
            representation=RepresentationConfig(method="rbf"),
        )


@pytest.mark.parametrize("method", ["bspline", "mspline", "ispline"])
@pytest.mark.parametrize(("output_dim", "expected"), [(3, 5), (51, 50)])
def test_resolved_representation_reuses_bmi_clamping(make_config, method, output_dim, expected):
    config = make_config(feature_preprocessing={"age": {"method": method, "output_dim": output_dim}})
    resolved = config.representation_for("age", is_numerical=True)
    with pytest.warns(ConfigWarning, match=f"output_dim={output_dim}.*using {expected}"):
        block = create_transformer(method, is_numerical=True, config=config, representation=resolved)
    assert isinstance(block, Pipeline)
    assert block.named_steps[method].output_dim == expected
    assert resolved.output_dim == output_dim


@pytest.mark.parametrize(("method", "is_numerical"), [("cubic", True), ("b-spline", True), ("ohe", False)])
def test_resolved_representation_accepts_matching_method_aliases(make_config, method, is_numerical):
    config = make_config(feature_preprocessing={"feature": {"method": method}})
    resolved = config.representation_for("feature", is_numerical=is_numerical)
    block = create_transformer(method, is_numerical=is_numerical, config=config, representation=resolved)
    assert isinstance(block, Pipeline)
    assert get_spec(resolved.method).transformer_cls in {type(step) for step in block.named_steps.values()}


@pytest.mark.parametrize(
    ("random_state", "options", "expected"),
    [(None, {}, UNSET), (17, {}, 17), (17, {"random_state": None}, None), (17, {"random_state": 0}, 0)],
)
def test_resolved_representation_preserves_omitted_and_explicit_seeds(make_config, random_state, options, expected):
    config = make_config(random_state=random_state, feature_preprocessing={"age": {"method": "rbf", **options}})
    resolved = config.representation_for("age", is_numerical=True)
    with patch(
        "pretab.compose.factory.get_numerical_transformer_steps", wraps=get_numerical_transformer_steps
    ) as steps:
        block = create_transformer("rbf", is_numerical=True, config=config, representation=resolved)
    assert isinstance(block, Pipeline)
    if expected is UNSET:
        assert "random_state" not in steps.call_args.kwargs
        assert block.named_steps["rbf"].random_state == get_spec("rbf").transformer_cls().random_state
    else:
        assert steps.call_args.kwargs["random_state"] == expected
        assert block.named_steps["rbf"].random_state == expected


def test_resolved_representation_keeps_explicit_adaptive_bounds(make_config):
    config = make_config(
        feature_preprocessing={
            "age": {
                "method": "rbf",
                "output_dim": 9,
                "adaptive": True,
                "min_output_dim": 3,
                "max_output_dim": 12,
            }
        }
    )
    resolved = config.representation_for("age", is_numerical=True)
    block = create_transformer("rbf", is_numerical=True, config=config, representation=resolved)
    assert isinstance(block, Pipeline)
    transformer = block.named_steps["rbf"]
    assert (transformer.output_dim, transformer.adaptive, transformer.min_output_dim, transformer.max_output_dim) == (
        9,
        True,
        3,
        12,
    )


def test_resolved_representation_overrides_supervised_run_without_leaking_defaults(make_config):
    config = make_config(
        target_aware=True,
        placement_strategy="cart",
        feature_preprocessing={
            "age": {
                "method": "rbf",
                "target_aware": False,
                "placement_strategy": "quantile",
                "output_dim": 3,
            }
        },
    )
    resolved = config.representation_for("age", is_numerical=True)
    block = create_transformer("rbf", is_numerical=True, config=config, representation=resolved)
    assert isinstance(block, Pipeline)
    transformer = block.named_steps["rbf"]
    assert transformer.target_aware is False
    assert transformer.placement_strategy == "quantile"
    assert block.fit_transform(np.linspace(0, 1, 16).reshape(-1, 1)).shape == (16, 3)


@pytest.mark.parametrize("is_numerical", [False, True])
@pytest.mark.parametrize("missing_policy", [None, "impute", "impute_with_indicator", "separate_state"])
def test_resolved_representation_preserves_missing_value_and_indicator_assembly(
    make_config, is_numerical, missing_policy
):
    method = "rbf" if is_numerical else "one-hot"
    options = {"output_dim": 3} if is_numerical else {}
    config = make_config(
        missing_policy=missing_policy, feature_preprocessing={"feature": {"method": method, **options}}
    )
    resolved = config.representation_for("feature", is_numerical=is_numerical)
    block = create_transformer(method, is_numerical=is_numerical, config=config, representation=resolved)
    data = (
        np.array([[0.0], [1.0], [np.nan], [2.0]])
        if is_numerical
        else np.array([["a"], ["b"], [np.nan], ["a"]], dtype=object)
    )
    result = block.fit_transform(data)
    indicator = missing_policy in {"impute_with_indicator", "separate_state"}
    assert result.shape == (4, (3 if is_numerical else 2) + int(indicator))
    assert isinstance(block, FeatureUnion if indicator else Pipeline)
    representation = dict(block.transformer_list)["representation"] if isinstance(block, FeatureUnion) else block
    assert isinstance(representation, Pipeline)
    assert representation.named_steps["imputer"].strategy == ("median" if is_numerical else "most_frequent")
    if indicator:
        values = result.toarray() if issparse(result) else result
        np.testing.assert_array_equal(values[:, -1], [0, 0, 1, 0])


def test_resolved_representation_keeps_run_wide_scaling(make_config):
    config = make_config(scaling="standardization", feature_preprocessing={"age": {"method": "rbf", "output_dim": 3}})
    block = create_transformer(
        "rbf", is_numerical=True, config=config, representation=config.representation_for("age", is_numerical=True)
    )
    assert isinstance(block, Pipeline)
    assert _names(block.steps) == ["imputer", "scaler", "rbf"]


@pytest.mark.parametrize(("method", "is_numerical"), [("rbf", True), ("one-hot", False)])
def test_explicit_none_representation_keeps_exact_legacy_calls(make_config, method, is_numerical):
    config = make_config()
    target = (
        "pretab.compose.factory.get_numerical_transformer_steps"
        if is_numerical
        else "pretab.compose.factory.get_categorical_transformer_steps"
    )
    builder = get_numerical_transformer_steps if is_numerical else get_categorical_transformer_steps
    with patch(target, wraps=builder) as steps:
        create_transformer(method, is_numerical=is_numerical, config=config)
        omitted = steps.call_args
        create_transformer(method, is_numerical=is_numerical, config=config, representation=None)
    assert steps.call_args == omitted


@pytest.mark.parametrize("value", [{"method": "rbf"}, "rbf", False])
def test_factory_resolved_input_requires_a_typed_value(make_config, value):
    with pytest.raises(InvalidParamError, match="must be a RepresentationConfig or None"):
        create_transformer("rbf", is_numerical=True, config=make_config(), representation=value)


@pytest.mark.parametrize(("selected", "requested", "is_numerical"), [("int", "rbf", False), ("rff", "rff", True)])
def test_factory_resolved_input_does_not_broaden_method_selection(make_config, selected, requested, is_numerical):
    with pytest.raises(InvalidParamError):
        create_transformer(
            selected,
            is_numerical=is_numerical,
            config=make_config(),
            representation=RepresentationConfig(method=requested),
        )
