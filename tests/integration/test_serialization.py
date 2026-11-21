"""Tests for fitted-state serialization through specs and pickle.

Covers the versioned JSON envelope, bit-for-bit transform reproduction across
representation families, fitted metadata, file round-trips, categorical / missing-value handling,
policy preservation, and the security allow-list that keeps loading a spec safe
(unlike ``pickle``).
"""

import json
import math
import pickle
import sys
from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn import config_context
from sklearn.base import clone

from pretab import Preprocessor, PretabSerializationError, RepresentationConfig, RepresentationPolicy
from pretab.compose.registry import TransformerSpec
from pretab.compose.serialize import SCHEMA_VERSION, _decode, _encode, _encode_mapping
from pretab.core.parameters import UNSET
from pretab.exceptions import InvalidParamError, OptionalDependencyError
from pretab.preprocessing.floats import NoTransformer, ToFloatTransformer


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": rng.random(60),
            "b": rng.random(60) * 10.0,
            "c": rng.choice(["x", "y", "z"], 60),
        }
    )


@pytest.fixture
def target():
    rng = np.random.default_rng(1)
    return rng.random(60)


@pytest.mark.parametrize("restore_format", ["pickle", "spec"])
def test_legacy_state_restores_constructor_defaults(frame, restore_format):
    processor = Preprocessor(numerical_method="rbf", target_aware=False, placement_strategy="uniform").fit(frame)
    expected = processor.transform(frame)
    fingerprint = processor.fingerprint_
    if restore_format == "pickle":
        del processor.feature_groups
        del processor.task_validation
        restored = pickle.loads(pickle.dumps(processor))  # noqa: S301
    else:
        spec = processor.to_spec()
        del spec["state"]["feature_groups"]
        del spec["params"]["feature_groups"]
        del spec["state"]["task_validation"]
        del spec["params"]["task_validation"]
        restored = Preprocessor.from_spec(spec)

    assert restored.get_params(deep=False)["feature_groups"] is None
    assert restored.get_params(deep=False)["task_validation"] == "off"
    assert restored.get_resolved_config()["feature_groups"] is None
    assert restored.get_resolved_config()["task_validation"] == "off"
    assert restored.fingerprint_ == fingerprint
    np.testing.assert_array_equal(restored.transform(frame), expected)
    copied = clone(restored)
    assert isinstance(copied, Preprocessor)
    assert copied.feature_groups is None
    assert copied.task_validation == "off"


def test_spec_preserves_nondefault_task_validation(frame):
    processor = Preprocessor(numerical_method="minmax", task_validation="raise").fit(frame)
    expected = processor.transform(frame)
    spec = processor.to_spec()
    assert spec["params"]["task_validation"] == "raise"
    restored = Preprocessor.from_spec(spec)
    assert restored.get_resolved_config()["task_validation"] == "raise"
    assert restored.fingerprint_ == processor.fingerprint_
    np.testing.assert_array_equal(restored.transform(frame), expected)


# Representation configs that round-trip; each is (params, id).
_CONFIGS = [
    {"numerical_method": "rbf", "target_aware": False, "placement_strategy": "quantile"},
    {"numerical_method": "sigmoid", "target_aware": False, "placement_strategy": "quantile"},
    {"numerical_method": "tanh", "target_aware": False, "placement_strategy": "quantile"},
    {"numerical_method": "relu", "target_aware": False, "placement_strategy": "quantile"},
    {"numerical_method": "bspline", "target_aware": False, "placement_strategy": "quantile"},
    {"numerical_method": "mspline", "target_aware": False, "placement_strategy": "quantile"},
    {"numerical_method": "ispline", "target_aware": False, "placement_strategy": "quantile"},
    {"numerical_method": "cubicspline", "target_aware": False, "placement_strategy": "quantile"},
    {"numerical_method": "naturalspline", "target_aware": False, "placement_strategy": "quantile"},
    {"numerical_method": "pspline", "target_aware": False, "placement_strategy": "uniform"},
    {"numerical_method": "ple", "target_aware": True, "placement_strategy": "cart"},
    {"numerical_method": "minmax", "target_aware": True, "placement_strategy": "cart"},
    {"numerical_method": "standardization", "target_aware": True, "placement_strategy": "cart"},
    {"numerical_method": "quantile", "target_aware": True, "placement_strategy": "cart"},
]


def _ids(configs):
    return [c["numerical_method"] for c in configs]


@pytest.mark.parametrize("params", _CONFIGS, ids=_ids(_CONFIGS))
@pytest.mark.parametrize("categorical_method", ["int", "one-hot"])
@pytest.mark.parametrize("restore_format", ["pickle", "spec"])
def test_round_trip_reproduces_transform_bit_for_bit(frame, target, params, categorical_method, restore_format):
    p = Preprocessor(output_dim=6, categorical_method=categorical_method, **params).fit(frame, target)
    fingerprint = p.fingerprint_
    names = p.get_feature_names_out()
    dimensions = p.output_dims_
    feature_info = p.get_feature_info(verbose=False)
    reference = np.asarray(p.transform(frame, return_array=True), dtype=float)

    if restore_format == "pickle":
        restored = pickle.loads(pickle.dumps(p))  # noqa: S301 - only locally generated test data
    else:
        restored = Preprocessor.from_spec(json.loads(json.dumps(p.to_spec(), allow_nan=False)))
    reproduced = np.asarray(restored.transform(frame, return_array=True), dtype=float)

    assert np.array_equal(reference, reproduced, equal_nan=True)
    np.testing.assert_array_equal(restored.get_feature_names_out(), names)
    assert reference.shape == reproduced.shape == (len(frame), len(names))
    assert sum(dimensions.values()) == reference.shape[1]
    assert restored.output_dims_ == dimensions
    assert restored.get_feature_info(verbose=False) == feature_info
    assert p.fingerprint_ == restored.fingerprint_ == fingerprint


@pytest.mark.parametrize("restore_format", ["pickle", "spec"])
@pytest.mark.parametrize("helper_type", [NoTransformer, ToFloatTransformer])
@pytest.mark.parametrize("add_missing_indicator", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_restore_legacy_sparse_one_hot_feature_count(restore_format, helper_type, add_missing_indicator, frozen):
    frame = pd.DataFrame({"city": ["Rome", None, "Berlin", "Rome"]})
    fitted = Preprocessor(
        numerical_method="none", categorical_method="one-hot", add_missing_indicator=add_missing_indicator
    ).fit(frame)
    fitted.set_output(transform="default")
    if frozen:
        fitted.freeze()
    expected = fitted.transform(frame)
    expected_names = fitted.get_feature_names_out()
    expected_info = fitted.get_feature_info(verbose=False)
    spec = fitted.to_spec()
    pipeline = fitted.column_transformer_.named_transformers_["cat_city"]
    if add_missing_indicator:
        pipeline = dict(pipeline.transformer_list)["representation"]
    helper = helper_type()
    helper.__dict__.update(vars(pipeline.named_steps["to_float"]))
    helper.n_features_in_ = 1
    pipeline.steps[-1] = ("to_float", helper)

    if restore_format == "pickle":
        restored = pickle.loads(pickle.dumps(fitted))  # noqa: S301 - only locally generated test data
    else:
        spec["state"] = _encode_mapping(vars(fitted))
        restored = Preprocessor.from_spec(spec)

    np.testing.assert_array_equal(restored.transform(frame), expected)
    np.testing.assert_array_equal(restored.get_feature_names_out(), expected_names)
    assert restored.get_feature_info(verbose=False) == expected_info
    assert restored.is_frozen() == frozen
    assert vars(restored)["_sklearn_output_config"] == vars(fitted)["_sklearn_output_config"]
    restored_pipeline = restored.column_transformer_.named_transformers_["cat_city"]
    if add_missing_indicator:
        restored_pipeline = dict(restored_pipeline.transformer_list)["representation"]
    restored_helper = restored_pipeline.named_steps["to_float"]
    assert type(restored_helper) is helper_type
    assert restored_helper.n_features_in_ == 2
    rewritten = Preprocessor.from_spec(restored.to_spec())
    assert rewritten.fingerprint_ == restored.fingerprint_
    np.testing.assert_array_equal(rewritten.transform(frame), expected)


@pytest.mark.parametrize("restore_format", ["pickle", "spec"])
@pytest.mark.parametrize("helper_type", [NoTransformer, ToFloatTransformer])
@pytest.mark.parametrize("state_case", ["healthy", "unexpected_width", "dense_input", "named_input"])
def test_restore_preserves_nonlegacy_helper_metadata(restore_format, helper_type, state_case):
    frame = pd.DataFrame({"city": ["Rome", "Berlin", "Rome"]})
    fitted = Preprocessor(numerical_method="none", categorical_method="one-hot").fit(frame)
    spec = fitted.to_spec()
    pipeline = fitted.column_transformer_.named_transformers_["cat_city"]
    helper = helper_type()
    helper.__dict__.update(vars(pipeline.named_steps["to_float"]))
    expected_count = {"healthy": 2, "unexpected_width": 3, "dense_input": 1, "named_input": 1}[state_case]
    helper.n_features_in_ = expected_count
    if state_case == "dense_input":
        pipeline.named_steps["onehot"].sparse_output = False
    if state_case == "named_input":
        helper.feature_names_in_ = np.array(["fitted_name"], dtype=object)
    pipeline.steps[-1] = ("to_float", helper)

    if restore_format == "pickle":
        restored = pickle.loads(pickle.dumps(fitted))  # noqa: S301 - only locally generated test data
    else:
        spec["state"] = _encode_mapping(vars(fitted))
        restored = Preprocessor.from_spec(spec)

    restored_helper = restored.column_transformer_.named_transformers_["cat_city"].named_steps["to_float"]
    assert restored_helper.n_features_in_ == expected_count
    if state_case == "named_input":
        np.testing.assert_array_equal(restored_helper.feature_names_in_, ["fitted_name"])
    if state_case != "healthy":
        with pytest.raises(ValueError):
            restored.get_feature_names_out()


def test_pickle_unfitted_preprocessor_preserves_params():
    original = Preprocessor(numerical_method="none", categorical_method="one-hot")
    restored = pickle.loads(pickle.dumps(original))  # noqa: S301 - only locally generated test data
    assert restored.get_params() == original.get_params()
    assert not hasattr(restored, "column_transformer_")


def test_spec_is_json_serializable_and_versioned(frame, target):
    p = Preprocessor(numerical_method="bspline", target_aware=False, placement_strategy="quantile").fit(frame, target)
    spec = p.to_spec()

    # The whole envelope must survive a JSON dumps/loads cycle unchanged.
    reparsed = json.loads(json.dumps(spec))
    assert reparsed["schema_version"] == SCHEMA_VERSION
    assert reparsed["pretab_version"] == spec["pretab_version"]
    assert set(reparsed["library_versions"]) == {"numpy", "scipy", "scikit_learn"}
    assert reparsed["feature_names_out"] == list(p.get_feature_names_out())


def test_file_round_trip(tmp_path, frame, target):
    p = Preprocessor(
        numerical_method="rbf", categorical_method="one-hot", target_aware=False, placement_strategy="quantile"
    ).fit(frame, target)
    path = tmp_path / "rep.json"

    returned = p.to_spec(path)
    assert path.exists()
    assert returned["schema_version"] == SCHEMA_VERSION  # to_spec still returns the dict

    restored = Preprocessor.from_spec(str(path))
    assert np.array_equal(
        np.asarray(p.transform(frame, return_array=True), dtype=float),
        np.asarray(restored.transform(frame, return_array=True), dtype=float),
        equal_nan=True,
    )


def test_representation_summary_present(frame, target):
    p = Preprocessor(numerical_method="rbf", target_aware=False, placement_strategy="quantile").fit(frame, target)
    spec = p.to_spec()
    families = {entry["family"] for entry in spec["representations"]}
    assert "rbf" in families


@pytest.mark.parametrize("leaf_input", ["imputed", "frame"])
def test_representation_summary_names_each_block_after_its_column(frame, target, leaf_input):
    """Without an imputer or scaler each representation is fitted on its DataFrame
    column and records ``feature_names_in_``, so a spec built from ``x0`` raised
    and the summary silently came back empty. The entries are named after the
    block's column instead of x0."""
    params = {"numerical_method": "rbf", "target_aware": False, "placement_strategy": "quantile"}
    if leaf_input == "frame":
        params.update(numerical_imputation=None, scaling=None)
    p = Preprocessor(**params).fit(frame, target)
    entries = {entry["columns"][0]: entry for entry in p.to_spec()["representations"]}
    inputs = {column: entry["input_features"] for column, entry in entries.items()}

    assert p.reproducibility_report()["representations"] == {"a": "rbf", "b": "rbf", "c": "ordinal"}
    assert inputs == {"a": ["a"], "b": ["b"], "c": ["c"]}
    assert entries["a"]["output_features"] == [f"a_rbf{i}" for i in range(entries["a"]["output_dim"])]


def test_representation_summary_names_the_representation_beside_the_missing_indicator(frame, target):
    """The summary names the representation's input, not the separate indicator."""
    X = frame.copy()
    X.loc[::7, "a"] = np.nan
    p = Preprocessor(
        numerical_method="rbf", target_aware=False, placement_strategy="quantile", add_missing_indicator=True
    ).fit(X, target)

    entries = {entry["columns"][0]: entry for entry in p.to_spec()["representations"]}

    assert entries["a"]["input_features"] == ["a"]
    assert entries["b"]["input_features"] == ["b"]
    assert p.reproducibility_report()["representations"] == {"a": "rbf", "b": "rbf", "c": "ordinal"}
    assert isinstance(p.fingerprint_, str)


def test_round_trip_preserves_dtype_and_output_format(frame, target):
    p = Preprocessor(
        numerical_method="bspline",
        target_aware=False,
        placement_strategy="quantile",
        dtype="float32",
        output_format="dense",
    ).fit(frame, target)
    restored = Preprocessor.from_spec(p.to_spec())

    out = restored.transform(frame, return_array=True)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert restored.dtype == "float32"
    assert restored.output_format == "dense"


@pytest.mark.parametrize("transform", [None, "default", "pandas"], ids=["unset", "default", "pandas"])
def test_portable_spec_restores_without_polars(frame, monkeypatch, transform):
    fitted = Preprocessor(numerical_method="minmax").fit(frame).set_output(transform=transform)
    expected = fitted.transform(frame)
    spec = fitted.to_spec()
    with monkeypatch.context() as missing:
        missing.setitem(sys.modules, "polars", None)
        with config_context(transform_output="polars"):
            restored = Preprocessor.from_spec(spec)

    actual = restored.transform(frame)
    assert type(actual) is type(expected)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(restored.get_feature_names_out(), fitted.get_feature_names_out())


@pytest.mark.parametrize("as_file", [False, True], ids=["dict", "file"])
def test_spec_with_polars_output_requires_polars_at_restore(frame, monkeypatch, tmp_path, as_file):
    fitted = Preprocessor(numerical_method="minmax").fit(frame).set_output(transform="polars")
    path = tmp_path / "preprocessor.json"
    spec = fitted.to_spec(path if as_file else None)
    monkeypatch.setitem(sys.modules, "polars", None)

    with pytest.raises(OptionalDependencyError, match=r"set_output.*requires.*polars.*pip install polars"):
        Preprocessor.from_spec(path if as_file else spec)


@pytest.mark.parametrize("transform", ["default", "polars"])
def test_spec_round_trip_after_fit_on_polars(frame, monkeypatch, transform):
    polars = pytest.importorskip("polars")
    polars_frame = polars.DataFrame(frame.to_dict(orient="list"))
    fitted = Preprocessor(numerical_method="minmax").fit(polars_frame).set_output(transform=transform)
    expected = fitted.transform(frame)
    spec = fitted.to_spec()
    with monkeypatch.context() as missing:
        if transform == "default":
            missing.setitem(sys.modules, "polars", None)
        restored = Preprocessor.from_spec(spec)

    actual = restored.transform(frame)
    assert isinstance(actual, polars.DataFrame if transform == "polars" else np.ndarray)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(restored.get_feature_names_out(), fitted.get_feature_names_out())


def test_round_trip_preserves_policy(frame, target):
    p = Preprocessor(
        numerical_method="bspline",
        target_aware=False,
        placement_strategy="quantile",
        policy={"constant": "error"},
    ).fit(frame, target)
    restored = Preprocessor.from_spec(p.to_spec())
    assert isinstance(restored.policy_, RepresentationPolicy)
    assert restored.policy_.constant == "error"


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(config, id=config["numerical_method"])
        for config in _CONFIGS
        if config["numerical_method"] in {"bspline", "ple", "rbf"}
    ],
)
@pytest.mark.parametrize("categorical_method", ["int", "one-hot"])
@pytest.mark.parametrize("missing_policy", ["impute", "impute_with_indicator", "separate_state"])
@pytest.mark.parametrize("restore_format", ["pickle", "spec"])
def test_round_trip_with_missing_values(params, categorical_method, missing_policy, restore_format):
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0] * 5, "c": ["x", "y", None, "x", "y", "z"] * 5})
    y = np.arange(len(frame), dtype=float)
    p = Preprocessor(
        output_dim=6,
        categorical_method=categorical_method,
        missing_policy=missing_policy,
        random_state=0,
        **params,
    ).fit(frame, y)
    fingerprint = p.fingerprint_
    names = list(p.get_feature_names_out())
    dimensions = p.output_dims_
    feature_info = p.get_feature_info(verbose=False)
    reference = np.asarray(p.transform(frame, return_array=True), dtype=float)
    if restore_format == "pickle":
        restored = pickle.loads(pickle.dumps(p))  # noqa: S301 - only locally generated test data
    else:
        restored = Preprocessor.from_spec(json.loads(json.dumps(p.to_spec(), allow_nan=False)))

    np.testing.assert_array_equal(restored.transform(frame, return_array=True), reference)
    assert reference.shape == (len(frame), len(names))
    assert np.isfinite(reference).all()
    assert sum(dimensions.values()) == reference.shape[1]
    assert list(restored.get_feature_names_out()) == names
    assert restored.output_dims_ == dimensions
    assert restored.get_feature_info(verbose=False) == feature_info
    assert feature_info[0]["a"]["dimension"] == dimensions["a"]
    assert feature_info[1]["c"]["dimension"] == dimensions["c"]
    assert feature_info[2] == {}

    for candidate in (p, restored):
        for column, prefix in (("a", "num"), ("c", "cat")):
            if missing_policy != "impute":
                suffix = (
                    f"missingindicator_{column}"
                    if missing_policy == "impute_with_indicator"
                    else f"missing__{column}__missing"
                )
                indicator_name = f"{prefix}_{column}__{suffix}"
                output = np.asarray(candidate.transform(frame, return_array=True), dtype=float)
                np.testing.assert_array_equal(
                    output[:, names.index(indicator_name)], frame[column].isna().astype(float)
                )
            with pytest.raises(ValueError, match="columns are missing"):
                candidate.transform(frame.drop(columns=[column]))
        assert candidate.fingerprint_ == fingerprint


def test_round_trip_reproduces_unseen_category_encoding(frame, target):
    p = Preprocessor(
        numerical_method="rbf", categorical_method="one-hot", target_aware=False, placement_strategy="quantile"
    ).fit(frame, target)
    restored = Preprocessor.from_spec(p.to_spec())

    unseen = frame.copy()
    unseen.loc[unseen.index[:5], "c"] = "brand_new"
    assert np.array_equal(
        np.asarray(p.transform(unseen, return_array=True), dtype=float),
        np.asarray(restored.transform(unseen, return_array=True), dtype=float),
        equal_nan=True,
    )


@pytest.mark.parametrize("categorical_method", ["int", "one-hot"])
@pytest.mark.parametrize("impute", [True, False])
@pytest.mark.parametrize("add_missing_indicator", [True, False])
def test_round_trip_with_nullable_categorical_missing(categorical_method, impute, add_missing_indicator):
    frame = pd.DataFrame({"c": ["a", None, "b", "a"]}).convert_dtypes()
    options = {"categorical_method": categorical_method}
    if not impute:
        options["categorical_imputation"] = None
    fitted = Preprocessor(add_missing_indicator=add_missing_indicator, **options).fit(frame)
    original_fingerprint = fitted.fingerprint_
    spec = json.loads(json.dumps(fitted.to_spec()))
    restored = Preprocessor.from_spec(spec)

    np.testing.assert_array_equal(restored.transform(frame), fitted.transform(frame))
    assert restored.get_feature_names_out().tolist() == fitted.get_feature_names_out().tolist()
    assert restored.fingerprint_ == original_fingerprint


def test_to_spec_requires_fitted():
    from sklearn.exceptions import NotFittedError

    p = Preprocessor(numerical_method="rbf", target_aware=False, placement_strategy="quantile")
    with pytest.raises(NotFittedError):
        p.to_spec()


def test_from_spec_rejects_unknown_schema_version(frame, target):
    p = Preprocessor(numerical_method="rbf", target_aware=False, placement_strategy="quantile").fit(frame, target)
    spec = p.to_spec()
    spec["schema_version"] = SCHEMA_VERSION + 999
    with pytest.raises(PretabSerializationError, match="schema_version"):
        Preprocessor.from_spec(spec)


def test_from_spec_rejects_missing_schema_version():
    with pytest.raises(PretabSerializationError, match="schema_version"):
        Preprocessor.from_spec({"state": {}})


def test_from_spec_refuses_disallowed_module(frame, target):
    p = Preprocessor(numerical_method="rbf", target_aware=False, placement_strategy="quantile").fit(frame, target)
    spec = p.to_spec()
    # Simulate a tampered spec that tries to import an arbitrary class on load.
    spec["state"]["column_transformer_"] = {"__estimator__": {"class": "os:system", "state": {}}}
    with pytest.raises(PretabSerializationError, match="disallowed module"):
        Preprocessor.from_spec(spec)


def test_from_spec_refuses_builtins_module():
    """``builtins`` must not be resolvable at all (closes the arbitrary-call hole).

    Previously ``builtins`` was an allowed top-level module, so a crafted
    ``__dataclass__``/``__estimator__`` tag naming ``builtins:open`` could call
    ``open(**attacker_fields)`` with attacker-controlled keyword arguments,
    including creating/truncating an arbitrary file. This asserts the module is
    refused outright, and that no file is created as a side effect of the
    refused load.
    """
    import os
    import tempfile

    target_path = os.path.join(tempfile.gettempdir(), "pretab_test_should_not_exist.txt")
    if os.path.exists(target_path):
        os.remove(target_path)

    malicious_spec = {
        "schema_version": SCHEMA_VERSION,
        "state": {
            "__dict__": [
                [
                    "evil",
                    {
                        "__dataclass__": {
                            "class": "builtins:open",
                            "fields": {"file": target_path, "mode": "w"},
                        }
                    },
                ]
            ]
        },
    }
    with pytest.raises(PretabSerializationError, match="disallowed module"):
        Preprocessor.from_spec(malicious_spec)
    assert not os.path.exists(target_path)


def test_from_spec_refuses_estimator_not_a_base_estimator():
    """An allowed-module class that isn't a ``BaseEstimator`` must still be refused.

    Proves the ``__estimator__`` check is a structural ``issubclass`` check, not
    just a module-prefix check: ``numpy`` is an allowed module, but
    ``numpy.ndarray`` is not a scikit-learn estimator.
    """
    spec = {
        "schema_version": SCHEMA_VERSION,
        "state": {"__dict__": [["evil", {"__estimator__": {"class": "numpy:ndarray", "state": {}}}]]},
    }
    with pytest.raises(PretabSerializationError, match="not a scikit-learn BaseEstimator"):
        Preprocessor.from_spec(spec)


@pytest.mark.parametrize("class_name", ["PreprocessorConfig", "_FrozenParams"])
def test_from_spec_refuses_dataclass_not_in_allowlist(class_name):
    """An allowed-module, genuinely-a-dataclass class must still be refused unless
    it is one of the specifically approved dataclasses.

    ``PreprocessorConfig`` is a real, internal PreTab dataclass (module "pretab",
    would pass a module-prefix check) that is deliberately not part of a fitted
    ``Preprocessor``'s serialized state, so it must not be reconstructable via a
    spec either.
    """
    spec = {
        "schema_version": SCHEMA_VERSION,
        "state": {
            "__dict__": [["evil", {"__dataclass__": {"class": f"pretab.compose.config:{class_name}", "fields": {}}}]]
        },
    }
    with pytest.raises(PretabSerializationError, match="disallowed dataclass"):
        Preprocessor.from_spec(spec)


@pytest.mark.parametrize(
    "settings",
    [
        {"method": "RBF"},
        {"method": "rbf", "output_dim": 3, "adaptive": False, "name": "local"},
        {
            "method": "rbf",
            "adaptive": True,
            "min_output_dim": None,
            "max_output_dim": 7,
            "params": {"random_state": None, "nested": {"levels": [True, 2, 0.5, None]}},
        },
    ],
)
def test_representation_config_codec_preserves_portable_values(settings):
    value = RepresentationConfig.from_dict(settings)
    text = json.dumps(_encode(value), allow_nan=False)
    assert "_FrozenParams" not in text
    restored = _decode(json.loads(text))
    assert type(restored) is RepresentationConfig
    assert restored == value
    assert restored.to_dict() == value.to_dict()
    for field in ("output_dim", "target_aware", "placement_strategy", "adaptive", "min_output_dim", "max_output_dim"):
        if field not in settings:
            assert getattr(restored, field) is UNSET
    detached = restored.to_dict()
    detached["params"]["changed"] = True
    assert restored == value


@pytest.mark.parametrize(
    ("fields", "constraint"),
    [
        ({"output_dim": 0}, "must be a positive integer"),
        ({"adaptive": "yes"}, "must be a boolean"),
        ({"params": {"gamma": math.inf}}, "finite JSON-compatible values"),
    ],
)
def test_representation_config_decoder_runs_value_validation(fields, constraint):
    payload = {
        "__dataclass__": {
            "class": "pretab.compose.config:RepresentationConfig",
            "fields": _encode_mapping({"method": "rbf", **fields}),
        }
    }
    with pytest.raises(InvalidParamError, match=constraint):
        _decode(payload)


@pytest.mark.parametrize(
    ("group_fields", "expected"),
    [
        ({}, (False, 1, 1)),
        ({"group_compatible": False, "min_arity": 1, "max_arity": 1}, (False, 1, 1)),
        ({"group_compatible": True, "min_arity": 2, "max_arity": None}, (True, 2, None)),
    ],
    ids=["legacy", "standalone", "group"],
)
def test_registry_spec_decoder_uses_additive_group_metadata(group_fields, expected):
    payload = {
        "__dataclass__": {
            "class": "pretab.compose.registry:TransformerSpec",
            "fields": {"name": "custom", "transformer_cls": _encode(NoTransformer), **group_fields},
        }
    }

    restored = _decode(json.loads(json.dumps(payload, allow_nan=False)))

    assert isinstance(restored, TransformerSpec)
    assert restored.name == "custom"
    assert restored.transformer_cls is NoTransformer
    assert (restored.group_compatible, restored.min_arity, restored.max_arity) == expected


def test_from_spec_rejects_bad_source_type():
    with pytest.raises(PretabSerializationError):
        Preprocessor.from_spec(12345)


# --- strict JSON: no NaN / Infinity tokens (issue #80) -----------------------------


def _strict_loads(text):
    def reject(token):
        raise ValueError(f"non-standard JSON token {token}")

    return json.loads(text, parse_constant=reject)


@pytest.fixture
def frame_with_missing():
    rng = np.random.default_rng(0)
    a = rng.normal(size=80)
    a[:6] = np.nan
    X = pd.DataFrame({"a": a, "b": rng.exponential(size=80), "c": rng.choice(["x", "y"], 80)})
    return X, rng.normal(size=80)


def test_default_spec_file_is_strict_json(frame_with_missing, tmp_path):
    """The imputers' NaN missing-value marker was written as a bare NaN token,
    which strict JSON parsers (JavaScript, Go, Rust, ...) reject."""
    X, y = frame_with_missing
    pre = Preprocessor(random_state=0).fit(X, y)
    path = tmp_path / "spec.json"
    fingerprint = pre.fingerprint_
    pre.to_spec(path)
    text = path.read_text(encoding="utf-8")
    pre.to_spec(path)
    assert path.read_text(encoding="utf-8") == text
    assert pre.fingerprint_ == fingerprint
    assert "NaN" not in text and "Infinity" not in text
    loaded = Preprocessor.from_spec(_strict_loads(text))
    np.testing.assert_array_equal(np.asarray(loaded.transform(X)), np.asarray(pre.transform(X)))
    assert loaded.fingerprint_ == pre.fingerprint_


def test_nan_category_in_an_object_array_is_strict_json():
    """A NaN category of an un-imputed one-hot encoder sits in an object array."""
    rng = np.random.default_rng(0)
    city = rng.choice(["x", "y", "z"], 80).astype(object)
    city[:5] = np.nan
    X = pd.DataFrame({"a": rng.normal(size=80), "city": city})
    pre = Preprocessor(categorical_method="one-hot", categorical_imputation=None, random_state=0).fit(X, X["a"])
    text = json.dumps(pre.to_spec(), allow_nan=False)
    assert '{"__float__": "nan"}' in text
    loaded = Preprocessor.from_spec(_strict_loads(text))
    np.testing.assert_array_equal(np.asarray(loaded.transform(X)), np.asarray(pre.transform(X)))
    assert loaded.fingerprint_ == pre.fingerprint_


def test_non_finite_floats_round_trip():
    from pretab.compose.serialize import _decode, _encode

    array = np.array([[1.0, np.nan], [np.inf, -np.inf]])
    decoded = _decode(_strict_loads(json.dumps(_encode(array), allow_nan=False)))
    assert isinstance(decoded, np.ndarray)
    np.testing.assert_array_equal(decoded, array)
    assert decoded.dtype == array.dtype
    nan, neg_inf = (
        float(cast(float, _decode(_strict_loads(json.dumps(_encode(value), allow_nan=False)))))
        for value in (np.nan, -np.inf)
    )
    assert math.isnan(nan) and neg_inf == -math.inf


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(), (4,), (2, 2), (0, 2)])
def test_strict_json_float_array_shapes_and_dtypes_round_trip(dtype, shape):
    from pretab.compose.serialize import _decode, _encode

    array = np.resize(np.array([np.nan, np.inf, -np.inf, 1.0], dtype=dtype), shape)
    decoded = _decode(_strict_loads(json.dumps(_encode(array), allow_nan=False)))

    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == shape
    assert decoded.dtype == array.dtype
    np.testing.assert_array_equal(decoded, array)


@pytest.mark.parametrize("shape", [(), (6,), (2, 3), (0, 2)])
def test_strict_json_object_arrays_preserve_nested_elements(shape):
    from pretab.compose.serialize import _decode, _encode

    elements = [
        [np.bool_(True), np.float32(np.nan)],
        (np.int64(7), np.float64(np.inf)),
        np.array([np.nan, -np.inf], dtype=np.float32),
        {"missing": np.float32(np.nan)},
        np.bool_(False),
        np.int64(3),
    ]
    array = np.empty(shape, dtype=object)
    for index in range(array.size):
        array.flat[index] = elements[index]
    decoded = _decode(_strict_loads(json.dumps(_encode(array), allow_nan=False)))

    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == shape
    assert decoded.dtype == object
    for actual, expected in zip(decoded.flat, array.flat, strict=True):
        assert type(actual) is type(expected) or isinstance(expected, np.generic)
        np.testing.assert_equal(actual, expected)
        if isinstance(expected, np.ndarray):
            assert actual.dtype == expected.dtype


def test_specs_with_bare_nan_tokens_still_load(frame_with_missing):
    X, y = frame_with_missing
    pre = Preprocessor(random_state=0).fit(X, y)
    spec = json.loads(json.dumps(pre.to_spec()))
    # Rewrite the tagged floats the way older versions wrote them: as bare NaN.
    legacy_text = json.dumps(spec).replace('{"__float__": "nan"}', "NaN")
    assert "NaN" in legacy_text
    loaded = Preprocessor.from_spec(json.loads(legacy_text))
    np.testing.assert_array_equal(np.asarray(loaded.transform(X)), np.asarray(pre.transform(X)))


# --- object arrays and builtin dtypes (issue #66) ---------------------------------


@pytest.fixture
def numpy_scalar_frame():
    rng = np.random.default_rng(0)
    n = 80
    income = np.where(rng.random(n) < 0.1, np.nan, rng.normal(50, 10, n))
    frame = pd.DataFrame({"income": income, "age": rng.normal(40, 10, n)})
    # An object column of np.bool_ / None, as produced by a row-wise apply.
    frame["high_income"] = frame.apply(lambda r: r.income > 50 if pd.notna(r.income) else None, axis=1)
    return frame, rng.normal(size=n)


def test_object_arrays_of_numpy_scalars_serialize(numpy_scalar_frame, tmp_path):
    """Regression guard for issue #66: numpy scalars in object arrays leaked into the spec."""
    frame, y = numpy_scalar_frame
    pre = Preprocessor(random_state=0).fit(frame, y)
    json.dumps(pre.to_spec())
    assert len(pre.fingerprint_) == 64
    path = tmp_path / "spec.json"
    pre.to_spec(path)
    np.testing.assert_array_equal(Preprocessor.from_spec(path).transform(frame), pre.transform(frame))


@pytest.mark.parametrize("dtype", [float, int, np.float32])
def test_builtin_and_numpy_dtypes_round_trip(dtype):
    frame = pd.DataFrame({"age": np.linspace(20.0, 60.0, 30)})
    pre = Preprocessor(numerical_method="minmax", dtype=dtype).fit(frame)
    loaded = Preprocessor.from_spec(json.loads(json.dumps(pre.to_spec())))
    np.testing.assert_array_equal(loaded.transform(frame), pre.transform(frame))
    assert loaded.fingerprint_ == pre.fingerprint_


@pytest.mark.parametrize("unsupported", [object(), np.nan, np.inf, -np.inf], ids=["object", "nan", "inf", "neg_inf"])
def test_failed_save_leaves_an_existing_spec_untouched(tmp_path, monkeypatch, unsupported):
    import pretab.preprocessor as preprocessor_module

    frame = pd.DataFrame({"age": np.linspace(20.0, 60.0, 30)})
    pre = Preprocessor(numerical_method="minmax").fit(frame)
    path = tmp_path / "spec.json"
    pre.to_spec(path)
    saved = path.read_text(encoding="utf-8")

    # A spec the JSON writer rejects half-way: a streaming write into the
    # already-opened file used to leave a truncated, invalid spec behind.
    spec = pre.to_spec()
    monkeypatch.setattr(
        preprocessor_module, "preprocessor_to_spec", lambda _: {**spec, "state": {"unsupported_": unsupported}}
    )
    with pytest.raises(PretabSerializationError, match="cannot be written as JSON"):
        pre.to_spec(path)
    assert path.read_text(encoding="utf-8") == saved
    with pytest.raises(PretabSerializationError, match="cannot be written as JSON"):
        _ = pre.fingerprint_


def test_types_a_spec_cannot_load_are_rejected_at_save_time():
    from pretab.compose.serialize import _encode

    with pytest.raises(PretabSerializationError, match="can only reference types"):
        _encode(list)


def test_object_arrays_written_before_element_encoding_still_load():
    from pretab.compose.serialize import _decode

    payload = {"__ndarray__": {"dtype": "|O", "shape": [2, 2], "data": [["a", None], [True, 1.5]]}}
    decoded = _decode(payload)
    assert isinstance(decoded, np.ndarray)
    assert decoded.dtype == object
    assert decoded.tolist() == [["a", None], [True, 1.5]]
