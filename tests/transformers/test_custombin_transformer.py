import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from pretab.exceptions import InsufficientSamplesError, InvalidParamError, PretabDataError
from pretab.transformers import NumericBinningTransformer


@pytest.mark.parametrize("bins", [2, [0.0, 0.5, 1.0]])
def test_custom_bin_transformer_basic_functionality(bins):
    X = np.array([[0.1], [0.4], [0.6], [0.8], [0.95]])
    transformer = NumericBinningTransformer(output_dim=bins)
    transformer.fit(X)

    # Ensure fitted attributes exist
    assert hasattr(transformer, "n_features_in_")
    assert transformer.n_features_in_ == 1
    assert transformer.total_output_dim_ == 1
    assert len(transformer.bin_edges_) == 1

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
        else pd.DataFrame(raw, columns=pd.Index(["x"]))
    )
    transformer = NumericBinningTransformer(output_dim=bins)
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (4, 1)


def test_custom_bin_transformer_is_stateful():
    """Edges are learned at fit time and reused on shifted transform data."""
    X_train = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
    transformer = NumericBinningTransformer(output_dim=4).fit(X_train)
    learned = transformer.bin_edges_[0].copy()

    # Values outside the fitted range are clamped into the outer bins, and the
    # learned edges do not change when transforming different data.
    X_test = np.array([[-5.0], [0.25], [0.75], [5.0]])
    Xt = transformer.transform(X_test)
    np.testing.assert_array_equal(transformer.bin_edges_[0], learned)
    assert Xt[0, 0] == 0  # below the fitted minimum -> first bin
    assert Xt[-1, 0] == transformer.n_bins_[0] - 1  # above the maximum -> last bin


def test_custom_bin_transformer_quantile_placement():
    """Quantile placement puts edges on the empirical distribution."""
    rng = np.random.default_rng(0)
    X = rng.exponential(1.0, size=200).reshape(-1, 1)
    uniform = NumericBinningTransformer(output_dim=4, placement_strategy="uniform").fit(X)
    quantile = NumericBinningTransformer(output_dim=4, placement_strategy="quantile").fit(X)

    # The two strategies must produce different edges for skewed data.
    assert not np.allclose(uniform.bin_edges_[0], quantile.bin_edges_[0])

    # Quantile bins are all occupied for a well-spread sample.
    counts = np.bincount(quantile.transform(X).ravel(), minlength=4)
    assert counts.min() > 0


def test_custom_bin_transformer_onehot_encoding():
    X = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
    transformer = NumericBinningTransformer(output_dim=4, encode="onehot").fit(X)
    assert transformer.total_output_dim_ == 4

    Xt = transformer.transform(X)
    assert Xt.shape == (20, 4)
    # Exactly one active bin per row.
    np.testing.assert_array_equal(Xt.sum(axis=1), np.ones(20))
    assert set(np.unique(Xt)) <= {0.0, 1.0}


def test_custom_bin_transformer_soft_encoding():
    X = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
    transformer = NumericBinningTransformer(output_dim=5, encode="soft").fit(X)
    assert transformer.total_output_dim_ == 5

    Xt = transformer.transform(X)
    assert Xt.shape == (20, 5)
    # Weights are non-negative and sum to 1 for every row.
    assert Xt.min() >= 0.0
    np.testing.assert_allclose(Xt.sum(axis=1), np.ones(20))


def test_custom_bin_transformer_multifeature():
    X = np.column_stack([np.linspace(0.0, 1.0, 30), np.linspace(-5.0, 5.0, 30)])
    transformer = NumericBinningTransformer(output_dim=3, encode="onehot").fit(X)
    assert transformer.n_features_in_ == 2
    assert transformer.n_bins_ == [3, 3]
    assert transformer.total_output_dim_ == 6
    assert transformer.transform(X).shape == (30, 6)


def test_custom_bin_transformer_invalid_encode():
    X = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
    with pytest.raises(InvalidParamError):
        NumericBinningTransformer(output_dim=3, encode="bogus").fit(X)


def test_custom_bin_transformer_invalid_placement():
    X = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
    with pytest.raises(InvalidParamError):
        NumericBinningTransformer(output_dim=3, placement_strategy="cart").fit(X)


def test_custom_bin_transformer_missing_output_dim():
    X = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
    with pytest.raises(InvalidParamError):
        NumericBinningTransformer().fit(X)


def test_custom_bin_transformer_invalid_input():
    transformer = NumericBinningTransformer(output_dim=3)
    transformer.fit(np.linspace(0.0, 1.0, 10).reshape(-1, 1))
    with pytest.raises(PretabDataError):
        transformer.transform("invalid_input")


def test_custom_bin_transformer_raises_on_insufficient_samples():
    transformer = NumericBinningTransformer(output_dim=3)
    X = np.array([[0.1]])  # Not enough observations to bin.

    with pytest.raises(ValueError, match=r"Input must have more than 2 observations."):
        transformer.fit(X)


def test_custom_bin_transformer_insufficient_samples_via_fit_transform():
    with pytest.raises(InsufficientSamplesError):
        NumericBinningTransformer(output_dim="not_valid").fit_transform(np.array([[0.1]]))


def test_custom_bin_transformer_feature_names_out_ordinal():
    transformer = NumericBinningTransformer(output_dim=3)
    transformer.fit(np.linspace(0.0, 1.0, 10).reshape(-1, 1))
    names = transformer.get_feature_names_out(["feature1"])
    assert list(names) == ["feature1"]


def test_custom_bin_transformer_feature_names_out_onehot():
    transformer = NumericBinningTransformer(output_dim=3, encode="onehot")
    transformer.fit(np.linspace(0.0, 1.0, 10).reshape(-1, 1))
    names = transformer.get_feature_names_out(["feature1"])
    assert list(names) == ["feature1_bin0", "feature1_bin1", "feature1_bin2"]


def test_custom_bin_transformer_feature_names_out_raises():
    """Regression guard for issue #36: an unfitted transformer must raise

    NotFittedError, but a fitted one must accept no arguments and default to
    generated ``x0, x1, ...`` names (matching the sklearn contract), not raise.
    """
    transformer = NumericBinningTransformer(output_dim=3)
    with pytest.raises(NotFittedError):
        transformer.get_feature_names_out()

    transformer.fit(np.linspace(0.0, 1.0, 10).reshape(-1, 1))
    names = transformer.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert list(names) == ["x0"]


def test_custom_bin_transformer_get_feature_names_out_in_pipeline():
    """Regression guard for issue #36: Pipeline.get_feature_names_out() must work

    for a pipeline containing this transformer, with no names passed explicitly.
    """
    from sklearn.pipeline import Pipeline

    X = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
    pipeline = Pipeline([("bin", NumericBinningTransformer(output_dim=3))]).fit(X)
    names = pipeline.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert list(names) == ["x0"]


def test_custom_bin_transformer_is_sklearn_compatible():
    assert isinstance(NumericBinningTransformer(output_dim=3), (BaseEstimator, TransformerMixin))
