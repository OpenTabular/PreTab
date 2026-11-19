"""Unit tests for :mod:`pretab.compose.factory`."""

from dataclasses import fields, replace
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pretab import Preprocessor
from pretab.compose.config import PreprocessorConfig, RepresentationConfig
from pretab.compose.factory import (
    _clamp_spline_basis,
    _placement_kwargs,
    build_column_transformer,
    create_transformer,
    get_categorical_transformer_steps,
    get_numerical_transformer_steps,
)
from pretab.compose.feature_detection import with_string_labels
from pretab.compose.registry import TRANSFORMER_REGISTRY, TransformerSpec, get_spec
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


@pytest.mark.parametrize("method", ["tensorspline", "tprs"])
def test_grouped_factory_selects_sources_once_in_declared_order(make_config, method):
    config = make_config(feature_groups={"location": {"columns": ["y", "x"], "representation": method}})
    transformer = build_column_transformer(config, ["x", "untouched", "y"], ["city"])

    assert [(name, columns) for name, _, columns in transformer.transformers] == [
        ("num_untouched", ["untouched"]),
        ("group_location", ["y", "x"]),
        ("cat_city", ["city"]),
    ]


@pytest.mark.parametrize("reverse", [False, True])
def test_grouped_factory_preserves_declaration_order_between_singles(make_config, reverse):
    groups = [
        {"name": "first", "columns": ["y", "x"], "representation": "tensor"},
        {"name": "second", "columns": ["w", "z"], "representation": "thinplate"},
    ]
    if reverse:
        groups.reverse()
    config = make_config(feature_groups=groups)
    transformer = build_column_transformer(config, ["w", "x", "untouched", "y", "z"], ["city"])
    assert [(name, columns) for name, _, columns in transformer.transformers] == [
        ("num_untouched", ["untouched"]),
        *[(f"group_{group['name']}", group["columns"]) for group in groups],
        ("cat_city", ["city"]),
    ]
    assert [group.name for group in config.feature_groups] == [group["name"] for group in groups]


@pytest.mark.parametrize("method", ["tensorspline", "tprs"])
@pytest.mark.parametrize("scaling", [None, "standardization", "minmax"])
@pytest.mark.parametrize("missing_policy", ["impute", "impute_with_indicator", "separate_state"])
def test_grouped_factory_matches_joint_reference_without_indicator_inputs(make_config, method, scaling, missing_policy):
    frame = pd.DataFrame(np.random.default_rng(42).normal(size=(24, 2)), columns=pd.Index(["x", "y"]))
    frame.loc[[1, 7], "x"] = np.nan
    before = frame.copy(deep=True)
    options = (
        {"output_dim": 3, "degree": 2, "diff_order": 1, "placement_strategy": "quantile"}
        if method == "tensorspline"
        else {"n_components": 4, "landmark_strategy": "subsample", "rank_strategy": "nystroem", "random_state": 0}
    )
    config = make_config(
        numerical_method="ple",
        output_dim=13,
        degree=5,
        adaptive=True,
        target_aware=True,
        placement_strategy="cart",
        scaling=scaling,
        missing_policy=missing_policy,
        random_state=17,
        feature_groups={"location": {"columns": ["y", "x"], "representation": {"method": method, **options}}},
    )
    transformer = build_column_transformer(config, ["x", "y"], [], sparse_threshold=0.0)
    result = transformer.fit_transform(frame)
    assert isinstance(result, np.ndarray)
    block = transformer.named_transformers_["group_location"]
    pipeline = dict(block.transformer_list)["representation"] if isinstance(block, FeatureUnion) else block
    assert isinstance(pipeline, Pipeline)
    leaf = pipeline.named_steps[method]
    assert leaf.n_features_in_ == 2
    assert pipeline.named_steps["imputer"].statistics_.shape == (2,)
    assert pipeline.named_steps["imputer"].add_indicator is False
    for key, value in options.items():
        assert getattr(leaf, key) == value

    reference_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scaling is not None:
        scaler = StandardScaler() if scaling == "standardization" else MinMaxScaler(feature_range=(-1, 1))
        reference_steps.append(("scaler", scaler))
    reference_steps.append((method, get_spec(method).transformer_cls(**options)))
    reference = Pipeline(reference_steps)
    expected = reference.fit_transform(frame[["y", "x"]])
    width = expected.shape[1]
    assert width == (9 if method == "tensorspline" else 4)
    np.testing.assert_allclose(result[:, :width], expected, rtol=1e-12, atol=1e-12)

    missing_columns = ["x"] if missing_policy == "impute_with_indicator" else ["y", "x"]
    indicator_width = 0 if missing_policy == "impute" else len(missing_columns)
    assert result.shape == (len(frame), width + indicator_width)
    assert isinstance(block, Pipeline if missing_policy == "impute" else FeatureUnion)
    if indicator_width:
        np.testing.assert_array_equal(result[:, width:], frame[missing_columns].isna().to_numpy(dtype=float))

    heldout = frame.iloc[:4].copy()
    heldout.loc[0, "y"] = np.nan
    transformed = transformer.transform(heldout[["y", "x"]])
    assert isinstance(transformed, np.ndarray)
    np.testing.assert_allclose(transformed[:, :width], reference.transform(heldout[["y", "x"]]), rtol=1e-12, atol=1e-12)
    if indicator_width:
        np.testing.assert_array_equal(transformed[:, width:], heldout[missing_columns].isna().to_numpy(dtype=float))
    assert transformed.shape[1] == result.shape[1]
    pd.testing.assert_frame_equal(frame, before)


def test_grouped_factory_maps_integer_labels_without_positional_selection(make_config):
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(16, 3)), columns=pd.Index([7, 3, 9]))
    config = make_config(
        feature_groups={
            "location": {
                "columns": [3, 7],
                "representation": {"method": "tprs", "n_components": 2, "landmark_strategy": "subsample"},
            }
        },
        random_state=0,
    )
    transformer = build_column_transformer(config, [7, 3, 9], [])
    assert [(name, columns) for name, _, columns in transformer.transformers] == [
        ("num_9", ["9"]),
        ("group_location", ["3", "7"]),
    ]
    normalized = with_string_labels(frame)
    result = transformer.fit_transform(normalized)
    np.testing.assert_array_equal(result, transformer.transform(normalized[["9", "3", "7"]]))
    assert config.feature_groups[0].columns == (3, 7)
    assert list(frame.columns) == [7, 3, 9]


def test_grouped_factory_keeps_private_step_names_safe_and_unique(make_config):
    frame = pd.DataFrame(np.random.default_rng(1).normal(size=(16, 4)), columns=pd.Index(["_unsafe", "col0", "x", "y"]))
    frame["__city"] = ["a", "b"] * 8
    config = make_config(
        feature_groups={
            "num_col0": {
                "columns": ["x", "y"],
                "representation": {"method": "tprs", "n_components": 2, "landmark_strategy": "subsample"},
            }
        },
        random_state=0,
    )
    transformer = build_column_transformer(config, ["_unsafe", "col0", "x", "y"], ["__city"])
    names = [name for name, _, _ in transformer.transformers]
    assert names == ["num_col0_1", "num_col0", "group_num_col0", "cat_col3"]
    assert len(set(names)) == len(names)
    assert all("__" not in name for name in names)
    cloned = clone(transformer)
    assert isinstance(cloned, ColumnTransformer)
    np.testing.assert_array_equal(transformer.fit_transform(frame), cloned.fit_transform(frame))


@pytest.mark.parametrize("claimed_collision", [False, True])
def test_group_public_key_collision_precedes_all_construction(make_config, monkeypatch, claimed_collision):
    columns = ["group_location", "y"] if claimed_collision else ["x", "y"]
    config = make_config(feature_groups={"location": {"columns": columns, "representation": "tprs"}})

    def unexpected(*args, **kwargs):
        pytest.fail("A group public-key collision must be rejected before constructing any block")

    monkeypatch.setattr("pretab.compose.factory._create_transformer", unexpected)
    with pytest.raises(InvalidParamError, match=r"feature_groups\['location'\].name.*location") as failure:
        build_column_transformer(config, ["untouched", "x", "y", "group_location"], [])
    assert "public block key 'group_location'" in str(failure.value)


@pytest.mark.parametrize("method", ["tensor", "thinplate"])
def test_grouped_factory_capability_is_rechecked_before_construction(make_config, monkeypatch, method):
    config = make_config(feature_groups={"location": {"columns": ["x", "y"], "representation": method}})
    canonical = config.group_representation_for(config.feature_groups[0]).method
    spec = get_spec(canonical)
    monkeypatch.setitem(TRANSFORMER_REGISTRY, canonical, replace(spec, group_compatible=False))
    with pytest.raises(InvalidParamError, match="does not support explicit feature groups"):
        build_column_transformer(config, ["x", "y"], [])


@pytest.mark.parametrize("method", ["tensorspline", "tprs", "rff", "nystroem"])
def test_grouped_factory_does_not_unlock_single_column_step_selection(method):
    with pytest.raises(InvalidParamError, match="unrecognized numerical preprocessing method"):
        get_numerical_transformer_steps(method, is_group=True)


@pytest.mark.parametrize("method", ["tensorspline", "tprs"])
def test_grouped_representation_cannot_enter_public_single_column_builder(make_config, method):
    with pytest.raises(InvalidParamError, match=r"create_transformer\.representation\.method") as failure:
        create_transformer(
            method, is_numerical=True, config=make_config(), representation=RepresentationConfig(method=method)
        )
    assert "unrecognized numerical preprocessing method" in str(failure.value)


@pytest.mark.parametrize("typed", [False, True], ids=["mapping", "value"])
def test_preprocessor_fits_structured_representation(typed):
    settings = {"method": "rbf", "output_dim": 3, "params": {"gamma": 0.5}}
    representation = RepresentationConfig.from_dict(settings) if typed else settings
    frame = pd.DataFrame({"age": np.linspace(0, 1, 30)})
    processor = Preprocessor(
        numerical_method="none",
        feature_preprocessing={"age": representation},
        output_dim=9,
        target_aware=False,
        placement_strategy="quantile",
        scaling="none",
    ).fit(frame)

    basis = processor.column_transformer_.named_transformers_["num_age"].named_steps["rbf"]
    assert basis.output_dim == 3
    assert basis.gamma == 0.5
    transformed = processor.transform(frame, return_array=True)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (30, 3)


@pytest.mark.parametrize("typed", [False, True], ids=["mapping", "value"])
@pytest.mark.parametrize(
    ("method", "canonical", "output_dim", "params"),
    [("RBF", "rbf", 3, {"gamma": 0.5}), ("b-spline", "bspline", 6, {"degree": 2})],
)
def test_preprocessor_routes_local_options_through_both_step_builders(typed, method, canonical, output_dim, params):
    settings = {
        "age": {"method": method, "output_dim": output_dim, "params": params, "name": "age-basis"},
        "city": {"method": "OHE", "params": {"drop": "first", "sparse_output": False}},
    }
    overrides = {
        feature: RepresentationConfig.from_dict(value) if typed else value for feature, value in settings.items()
    }
    frame = pd.DataFrame({"age": np.linspace(0, 1, 40), "city": ["a", "b"] * 20})
    processor = Preprocessor(
        feature_preprocessing=overrides,
        output_dim=11,
        degree=3,
        random_state=17,
        target_aware=False,
        placement_strategy="quantile",
        scaling="standardization",
    )

    with (
        patch("pretab.compose.factory.create_transformer", wraps=create_transformer) as create,
        patch(
            "pretab.compose.factory.get_numerical_transformer_steps", wraps=get_numerical_transformer_steps
        ) as numerical,
        patch(
            "pretab.compose.factory.get_categorical_transformer_steps", wraps=get_categorical_transformer_steps
        ) as categorical,
    ):
        processor.fit(frame)

    assert [entry.args[0] for entry in create.call_args_list] == [canonical, "one-hot"]
    assert all(isinstance(entry.kwargs["representation"], RepresentationConfig) for entry in create.call_args_list)
    numerical.assert_called_once()
    options = numerical.call_args.kwargs
    assert options["method"] == canonical
    assert options["output_dim"] == output_dim
    assert options["target_aware"] is False
    assert options["placement_strategy"] == "quantile"
    assert options["scaling"] == "standardization"
    assert "name" not in options
    if canonical == "rbf":
        assert "degree" not in options
    for key, value in params.items():
        assert options[key] == value
    categorical.assert_called_once_with(
        "one-hot", add_imputer=True, imputer_strategy="most_frequent", drop="first", sparse_output=False
    )

    numeric = processor.column_transformer_.named_transformers_["num_age"]
    assert _names(numeric.steps) == ["imputer", "scaler", canonical]
    basis = numeric.named_steps[canonical]
    assert basis.output_dim == output_dim
    assert basis.target_aware is False
    assert basis.placement_strategy == "quantile"
    for key, value in params.items():
        assert getattr(basis, key) == value
    encoder = processor.column_transformer_.named_transformers_["cat_city"].named_steps["onehot"]
    assert encoder.drop == "first"
    assert encoder.sparse_output is False
    transformed = processor.transform(frame, return_array=True)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (40, output_dim + 1)


@pytest.mark.parametrize("typed", [False, True], ids=["mapping", "value"])
@pytest.mark.parametrize("method", ["bspline", "mspline", "ispline"])
def test_preprocessor_clamps_each_local_spline_width(typed, method):
    settings = {
        "age": {"method": method, "output_dim": 3, "placement_strategy": "quantile", "degree": 2},
        "income": {"method": method, "output_dim": 51, "placement_strategy": "uniform", "degree": 3},
    }
    overrides = {
        feature: RepresentationConfig.from_dict(value) if typed else value for feature, value in settings.items()
    }
    frame = pd.DataFrame({"age": np.linspace(0, 1, 80) ** 3, "income": np.linspace(2, 8, 80) ** 2})
    with (
        patch("pretab.compose.factory._clamp_spline_basis", wraps=_clamp_spline_basis) as clamp,
        pytest.warns(ConfigWarning) as emitted,
    ):
        processor = Preprocessor(
            feature_preprocessing=overrides,
            output_dim=11,
            target_aware=False,
            placement_strategy="uniform",
            scaling="none",
        ).fit(frame)

    assert clamp.call_args_list == [call(3), call(51)]
    assert len(emitted) == 2
    for feature, expected in (("age", 5), ("income", 50)):
        basis = processor.column_transformer_.named_transformers_[f"num_{feature}"].named_steps[method]
        assert basis.output_dim == expected
        assert basis.degree == settings[feature]["degree"]
        assert basis.placement_strategy == settings[feature]["placement_strategy"]
    transformed = processor.transform(frame, return_array=True)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (80, 55)


@pytest.mark.parametrize("typed", [False, True], ids=["mapping", "value"])
@pytest.mark.parametrize("handle_unknown", ["error", "ignore"])
def test_preprocessor_forwards_portable_one_hot_constructor_options(typed, handle_unknown):
    params = {
        "categories": [["a", "b", "c"]],
        "drop": ["a"],
        "sparse_output": False,
        "dtype": "float32",
        "handle_unknown": handle_unknown,
        "min_frequency": None,
        "max_categories": None,
        "feature_name_combiner": "concat",
    }
    settings = {"method": "one-hot", "params": params}
    requested = RepresentationConfig.from_dict(settings) if typed else settings
    frame = pd.DataFrame({"city": ["a", "b", "c"] * 10})
    processor = Preprocessor(feature_preprocessing={"city": requested}).fit(frame)
    block = processor.column_transformer_.named_transformers_["cat_city"]
    encoder = block.named_steps["onehot"]
    assert {key: encoder.get_params()[key] for key in params} == params
    np.testing.assert_array_equal(processor.transform(frame, return_array=True)[:3], [[0, 0], [1, 0], [0, 1]])
    if handle_unknown == "error":
        with pytest.raises(ValueError, match="unknown categories"):
            processor.transform(pd.DataFrame({"city": ["unseen"]}), return_array=True)


@pytest.mark.parametrize("typed", [False, True], ids=["mapping", "value"])
def test_preprocessor_applies_each_local_placement_capability(typed):
    settings = {
        "supervised": {"method": "ple", "output_dim": 3, "target_aware": True, "placement_strategy": "cart"},
        "unsupervised": {"method": "rbf", "output_dim": 4, "target_aware": False, "placement_strategy": "quantile"},
        "penalized": {"method": "pspline", "output_dim": 6, "target_aware": False, "placement_strategy": "uniform"},
    }
    overrides = {
        feature: RepresentationConfig.from_dict(value) if typed else value for feature, value in settings.items()
    }
    values = np.linspace(0, 1, 80)
    frame = pd.DataFrame({feature: values ** (index + 1) for index, feature in enumerate(settings)})
    with patch("pretab.compose.factory._placement_kwargs", wraps=_placement_kwargs) as placement:
        processor = Preprocessor(feature_preprocessing=overrides, scaling="none").fit(frame, np.sin(6 * values))

    assert [entry.args[0].name for entry in placement.call_args_list] == ["ple", "rbf", "pspline"]
    for feature, requested in settings.items():
        basis = processor.column_transformer_.named_transformers_[f"num_{feature}"].named_steps[requested["method"]]
        assert basis.output_dim == requested["output_dim"]
        assert basis.placement_strategy == requested["placement_strategy"]
        if requested["method"] == "rbf":
            assert basis.target_aware is False
    transformed = processor.transform(frame, return_array=True)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (80, 13)


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
    fit_parameters = {name: resolved[name] for name in config_fields}
    fit_parameters["feature_preprocessing"] = feature_overrides

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

        normalize.assert_called_once_with(**fit_parameters)
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
