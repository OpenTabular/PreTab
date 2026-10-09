"""Tests for portable serialization: ``to_spec`` / ``from_spec`` (P9.1).

Covers the versioned JSON envelope, bit-for-bit transform reproduction across
representation families, file round-trips, categorical / missing-value handling,
policy preservation, and the security allow-list that keeps loading a spec safe
(unlike ``pickle``).
"""

import json
import math
from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from pretab import Preprocessor, PretabSerializationError, RepresentationPolicy
from pretab.compose.serialize import SCHEMA_VERSION
from pretab.transformers import RBFExpansionTransformer


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


# Representation configs that round-trip; each is (params, id).
_CONFIGS = [
    {"numerical_method": "rbf", "target_aware": False, "placement_strategy": "quantile"},
    {"numerical_method": "sigmoid", "target_aware": False, "placement_strategy": "quantile"},
    {"numerical_method": "tanh", "target_aware": False, "placement_strategy": "quantile"},
    {"numerical_method": "relu", "target_aware": False, "placement_strategy": "quantile"},
    {"numerical_method": "bspline", "target_aware": False, "placement_strategy": "quantile"},
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
def test_round_trip_reproduces_transform_bit_for_bit(frame, target, params, categorical_method):
    p = Preprocessor(output_dim=6, categorical_method=categorical_method, **params).fit(frame, target)
    reference = np.asarray(p.transform(frame, return_array=True), dtype=float)

    restored = Preprocessor.from_spec(p.to_spec())
    reproduced = np.asarray(restored.transform(frame, return_array=True), dtype=float)

    assert np.array_equal(reference, reproduced, equal_nan=True)
    assert list(p.get_feature_names_out()) == list(restored.get_feature_names_out())


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
    column; its spec raised there, and the summary silently came back empty. The
    entries are named after the block's column instead of x0."""
    params = {"numerical_method": "rbf", "target_aware": False, "placement_strategy": "quantile"}
    if leaf_input == "frame":
        params.update(numerical_imputation=None, scaling=None)
    p = Preprocessor(**params).fit(frame, target)
    entries = {entry["columns"][0]: entry for entry in p.to_spec()["representations"]}
    inputs = {column: entry["input_features"] for column, entry in entries.items()}

    assert p.reproducibility_report()["representations"] == {"a": "rbf", "b": "rbf", "c": "ordinal"}
    assert inputs == {"a": ["a"], "b": ["b"], "c": ["c"]}
    assert entries["a"]["output_features"] == [f"a_rbf{i}" for i in range(entries["a"]["output_dim"])]


def test_representation_summary_of_a_leaf_behind_the_imputer_indicator(frame, target):
    """Before the #62 fix, ``add_missing_indicator`` put the imputer's built-in
    indicator in front of the representation, so a preprocessor fitted then has a
    leaf with two inputs for a one-column block. Its summary entry names its own
    inputs instead of failing on the single column name."""
    X = frame.copy()
    X.loc[::7, "a"] = np.nan
    p = Preprocessor(numerical_method="rbf", target_aware=False, placement_strategy="quantile").fit(X, target)
    legacy_block = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("rbf", RBFExpansionTransformer(target_aware=False, placement_strategy="quantile")),
        ]
    ).fit(X[["a"]], target)
    column_transformer = p.column_transformer_
    column_transformer.transformers_ = [
        (name, legacy_block if columns == ["a"] else transformer, columns)
        for name, transformer, columns in column_transformer.transformers_
    ]

    entries = {entry["columns"][0]: entry for entry in p.to_spec()["representations"]}

    assert entries["a"]["input_features"] == ["x0", "x1"]
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


def test_round_trip_with_missing_values(target):
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0] * 5, "c": ["x", "y", None, "x", "y", "z"] * 5})
    y = np.arange(len(frame), dtype=float)
    p = Preprocessor(
        numerical_method="rbf",
        categorical_method="one-hot",
        target_aware=False,
        placement_strategy="quantile",
        numerical_imputation="median",
    ).fit(frame, y)
    restored = Preprocessor.from_spec(p.to_spec())
    assert np.array_equal(
        np.asarray(p.transform(frame, return_array=True), dtype=float),
        np.asarray(restored.transform(frame, return_array=True), dtype=float),
        equal_nan=True,
    )


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


def test_from_spec_refuses_dataclass_not_in_allowlist():
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
            "__dict__": [
                ["evil", {"__dataclass__": {"class": "pretab.compose.config:PreprocessorConfig", "fields": {}}}]
            ]
        },
    }
    with pytest.raises(PretabSerializationError, match="disallowed dataclass"):
        Preprocessor.from_spec(spec)


def test_from_spec_rejects_bad_source_type():
    with pytest.raises(PretabSerializationError):
        Preprocessor.from_spec(12345)


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


def test_failed_save_leaves_an_existing_spec_untouched(tmp_path):
    frame = pd.DataFrame({"age": np.linspace(20.0, 60.0, 30)})
    pre = Preprocessor(numerical_method="minmax").fit(frame)
    path = tmp_path / "spec.json"
    pre.to_spec(path)
    saved = path.read_text(encoding="utf-8")

    vars(pre)["unsupported_"] = lambda value: value
    with pytest.raises(PretabSerializationError):
        pre.to_spec(path)
    assert path.read_text(encoding="utf-8") == saved


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


# --- strict JSON: no NaN / Infinity tokens --------------------------------------------


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
    pre.to_spec(path)
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
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


def test_specs_with_bare_nan_tokens_still_load(frame_with_missing):
    X, y = frame_with_missing
    pre = Preprocessor(random_state=0).fit(X, y)
    spec = json.loads(json.dumps(pre.to_spec()))
    # Rewrite the tagged floats the way older versions wrote them: as bare NaN.
    legacy_text = json.dumps(spec).replace('{"__float__": "nan"}', "NaN")
    assert "NaN" in legacy_text
    loaded = Preprocessor.from_spec(json.loads(legacy_text))
    np.testing.assert_array_equal(np.asarray(loaded.transform(X)), np.asarray(pre.transform(X)))
