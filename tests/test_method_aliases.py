import numpy as np
import pandas as pd
import pytest

from pretab.core.exceptions import InvalidParamError
from pretab.pipeline.registry import (
    CATEGORICAL_ALIASES,
    CATEGORICAL_METHODS,
    NUMERICAL_ALIASES,
    NUMERICAL_METHODS,
    resolve_method,
)
from pretab.preprocessor import Preprocessor


def _resolve_num(name):
    return resolve_method(name, NUMERICAL_METHODS, NUMERICAL_ALIASES)


def _resolve_cat(name):
    return resolve_method(name, CATEGORICAL_METHODS, CATEGORICAL_ALIASES)


@pytest.mark.parametrize(
    "name, expected",
    [
        # case-insensitive
        ("PLE", "ple"),
        ("Ple", "ple"),
        ("STANDARDIZATION", "standardization"),
        # separator-insensitive (no explicit alias needed)
        ("box_cox", "box-cox"),
        ("boxcox", "box-cox"),
        ("yeo johnson", "yeo-johnson"),
        ("cubic-spline", "cubicspline"),
        ("cubic spline", "cubicspline"),
        ("b_spline", "bspline"),
        ("m spline", "mspline"),
        ("i-spline", "ispline"),
        # synonyms / abbreviations
        ("std", "standardization"),
        ("standard", "standardization"),
        ("standard-scaler", "standardization"),
        ("z-score", "standardization"),
        ("poly", "polynomial"),
        ("robustscaler", "robust"),
        ("bin", "custombin"),
        ("thin-plate", "tprs"),
        ("natural-cubic", "naturalspline"),
        ("tensor-product", "tensorspline"),
        ("passthrough", "none"),
    ],
)
def test_numerical_aliases_resolve(name, expected):
    assert _resolve_num(name) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("one-hot", "one-hot"),
        ("onehot", "one-hot"),
        ("one_hot", "one-hot"),
        ("One Hot", "one-hot"),
        ("OHE", "one-hot"),
        ("dummy", "one-hot"),
        ("ordinal", "int"),
        ("integer", "int"),
        ("label", "int"),
        ("onehot_from_ordinal", "onehot_from_ordinal"),
        ("onehot-from-ordinal", "onehot_from_ordinal"),
        ("embedding", "pretrained"),
        ("language", "pretrained"),
        ("passthrough", "none"),
    ],
)
def test_categorical_aliases_resolve(name, expected):
    assert _resolve_cat(name) == expected


def test_every_canonical_name_resolves_to_itself():
    for name in NUMERICAL_METHODS:
        assert _resolve_num(name) == name
    for name in CATEGORICAL_METHODS:
        assert _resolve_cat(name) == name


def test_alias_targets_are_canonical():
    for target in NUMERICAL_ALIASES.values():
        assert target in NUMERICAL_METHODS
    for target in CATEGORICAL_ALIASES.values():
        assert target in CATEGORICAL_METHODS


def test_unknown_method_is_returned_lowercased():
    # An unrecognized name falls through so the caller can raise its own error.
    assert _resolve_num("Nonsense") == "nonsense"
    assert _resolve_cat("Nonsense") == "nonsense"


@pytest.fixture
def sample_data():
    df = pd.DataFrame(
        {
            "num1": np.linspace(0.0, 1.0, 60),
            "cat1": np.array(["A", "B", "C"] * 20),
        }
    )
    y = df["num1"].to_numpy() * 2.0
    return df, y


@pytest.mark.parametrize(
    "alias, canonical",
    [("std", "standardization"), ("min-max", "minmax"), ("robustscaler", "robust")],
)
def test_numerical_alias_matches_canonical_output(sample_data, alias, canonical):
    X, y = sample_data
    out_alias = Preprocessor(numerical_method=alias, categorical_method="int").fit_transform(X, y, return_array=True)
    out_canon = Preprocessor(numerical_method=canonical, categorical_method="int").fit_transform(X, y, return_array=True)
    np.testing.assert_allclose(out_alias, out_canon)


@pytest.mark.parametrize(
    "alias, canonical",
    [("onehot", "one-hot"), ("ordinal", "int")],
)
def test_categorical_alias_matches_canonical_output(sample_data, alias, canonical):
    X, y = sample_data
    out_alias = Preprocessor(numerical_method="minmax", categorical_method=alias).fit_transform(X, y, return_array=True)
    out_canon = Preprocessor(numerical_method="minmax", categorical_method=canonical).fit_transform(X, y, return_array=True)
    np.testing.assert_allclose(out_alias, out_canon)


def test_alias_in_feature_preprocessing(sample_data):
    X, y = sample_data
    pre = Preprocessor(feature_preprocessing={"num1": "STD", "cat1": "OneHot"})
    out = pre.fit_transform(X, y)
    assert "num_num1" in out
    assert "cat_cat1" in out


def test_unknown_method_still_raises(sample_data):
    X, y = sample_data
    with pytest.raises(InvalidParamError):
        Preprocessor(numerical_method="definitely-not-a-method").fit_transform(X, y)
