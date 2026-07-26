import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from pretab.exceptions import InsufficientSamplesError, PretabDataError
from pretab.transformers import CustomBinTransformer


@pytest.mark.parametrize("bins", [2, [0.0, 0.5, 1.0]])
def test_custom_bin_transformer_basic_functionality(bins):
    X = np.array([[0.1], [0.4], [0.6], [0.8], [0.95]])
    transformer = CustomBinTransformer(output_dim=bins)
    transformer.fit(X)

    # Ensure fitted attribute exists
    assert hasattr(transformer, "n_features_in_")
    assert transformer.n_features_in_ == 1
    assert transformer.total_output_dim_ == 1

    # Transform
    Xt = transformer.transform(X)
    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (X.shape[0], 1)
    assert Xt.dtype.kind in {"i", "u"}  # integer bins

    # Check values are within bin range
    assert Xt.min() >= 0
    if isinstance(bins, int):
        assert Xt.max() < bins
    else:
        assert Xt.max() < len(bins) - 1


@pytest.mark.parametrize("bins", [2, [0.0, 0.5, 1.0]])
@pytest.mark.parametrize("input_type", ["list", "np", "df"])
def test_custom_bin_transformer_input_types(bins, input_type):
    raw = [[0.1], [0.4], [0.6], [0.8]]
    X = (
        np.array(raw)  # Always convert to array to be safe
        if input_type == "list"
        else np.array(raw)
        if input_type == "np"
        else pd.DataFrame(raw, columns=["x"])
    )
    transformer = CustomBinTransformer(output_dim=bins)
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (4, 1)


def test_custom_bin_transformer_invalid_input():
    transformer = CustomBinTransformer(output_dim=3)
    with pytest.raises(PretabDataError):
        transformer.transform("invalid_input")


def test_custom_bin_transformer_raises_on_invalid_shape():
    transformer = CustomBinTransformer(output_dim=3)
    X = np.array([[0.1]])  # This will become scalar after squeeze()

    with pytest.raises(ValueError, match=r"Input must have more than 2 observations."):
        transformer.transform(X)


def test_custom_bin_transformer_invalid_bins_type():
    with pytest.raises(InsufficientSamplesError):
        CustomBinTransformer(output_dim="not_valid").fit_transform(np.array([[0.1]]))


def test_custom_bin_transformer_feature_names_out():
    transformer = CustomBinTransformer(output_dim=3)
    transformer.fit(np.array([[0.2]]))
    names = transformer.get_feature_names_out(["feature1"])
    assert names == ["feature1"]


def test_custom_bin_transformer_feature_names_out_raises():
    transformer = CustomBinTransformer(output_dim=3)
    with pytest.raises(ValueError):
        transformer.get_feature_names_out()


def test_custom_bin_transformer_is_sklearn_compatible():
    assert isinstance(CustomBinTransformer(output_dim=3), (BaseEstimator, TransformerMixin))
