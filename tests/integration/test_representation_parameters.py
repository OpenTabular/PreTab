"""Representation parameters flow through the public Preprocessor API."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, QuantileTransformer

from pretab import Preprocessor
from pretab.exceptions import InvalidParamError


def test_representation_options_override_shared_defaults_per_method_and_feature():
    x = np.linspace(0.1, 0.9, 24)
    X = pd.DataFrame({"shared": x, "narrow": x, "string_override": x, "rank": x**2})
    pre = Preprocessor(
        numerical_method="polynomial",
        degree=4,
        scaling="none",
        representation_params={
            "poly": {"degree": 3, "include_bias": False},
            "quantile": {"n_quantiles": 8, "output_distribution": "normal"},
        },
        feature_preprocessing={
            "narrow": {"method": "polynomial", "degree": 2},
            "string_override": "poly",
            "rank": {"method": "quantile", "n_quantiles": 4},
        },
    )
    fitted = clone(pre)
    assert isinstance(fitted, Preprocessor)
    fitted.fit(X)
    blocks = fitted.transform(X, return_array=False)
    for feature, degree in [("shared", 3), ("narrow", 2), ("string_override", 3)]:
        expected = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(X[[feature]])
        np.testing.assert_allclose(blocks[f"num_{feature}"], expected)
    expected_rank = QuantileTransformer(n_quantiles=4, output_distribution="normal").fit_transform(X[["rank"]])
    np.testing.assert_allclose(blocks["num_rank"], expected_rank)

    restored = Preprocessor.from_spec(fitted.to_spec())
    assert restored.get_params()["representation_params"] == pre.representation_params
    np.testing.assert_array_equal(restored.transform(X), fitted.transform(X))


def test_one_hot_options_keep_block_width_and_metadata_in_sync():
    X = pd.DataFrame({"city": ["b", "a", "c", "b"], "region": ["y", "z", "x", "y"]})
    pre = Preprocessor(
        categorical_method="one-hot",
        representation_params={"ohe": {"drop": "first", "sparse_output": False}},
        feature_preprocessing={"region": {"method": "one_hot", "drop": None}},
    ).fit(X)

    blocks = pre.transform(X, return_array=False)
    _, metadata, _ = pre.get_feature_info(verbose=False)
    for feature, drop in [("city", "first"), ("region", None)]:
        expected = OneHotEncoder(drop=drop, sparse_output=False).fit_transform(X[[feature]])
        np.testing.assert_array_equal(blocks[f"cat_{feature}"], expected)
        assert metadata[feature]["dimension"] == expected.shape[1]
        assert metadata[feature]["categories"] == 3


@pytest.mark.parametrize(
    "options, message",
    [
        ({"representation_params": []}, "must be a mapping"),
        ({"representation_params": {"quantile": 8}}, "parameter mappings"),
        ({"representation_params": {"unknown": {}}}, "per-column method"),
        ({"representation_params": {"poly": {}, "polynomial": {}}}, "duplicates"),
        ({"representation_params": {"quantile": {"n_quantile": 8}}}, "n_quantile"),
        ({"feature_preprocessing": {"x": {"degree": 2}}}, "requires a string method name"),
        ({"feature_preprocessing": {"x": {"method": "polynomial", "degrees": 2}}}, "degrees"),
        ({"feature_preprocessing": {"x": {"method": "rbf", "target_aware": False}}}, "target_aware"),
    ],
)
def test_invalid_explicit_representation_options_fail_clearly(options, message):
    X = pd.DataFrame({"x": np.linspace(0.1, 0.9, 24)})
    with pytest.raises(InvalidParamError, match=message):
        Preprocessor(numerical_method="polynomial", **options).fit(X)


def test_spec_written_before_representation_options_can_still_be_loaded_and_cloned():
    X = pd.DataFrame({"x": [0.1, 0.2, 0.4]})
    fitted = Preprocessor(numerical_method="minmax").fit(X)
    spec = fitted.to_spec()
    spec["params"].pop("representation_params")
    spec["state"].pop("representation_params")

    restored = Preprocessor.from_spec(spec)
    cloned = clone(restored)
    assert isinstance(cloned, Preprocessor)
    assert cloned.get_params()["representation_params"] is None
    np.testing.assert_array_equal(restored.transform(X), fitted.transform(X))
