"""Unit tests for :class:`pretab.compose.config.PreprocessorConfig`."""

from dataclasses import asdict

import pytest

from pretab import Preprocessor
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
    }
    assert config.feature_preprocessing is not parameters["feature_preprocessing"]


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
