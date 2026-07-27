import numpy as np
import pandas as pd
import pytest
from sklearn.base import TransformerMixin, BaseEstimator
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
        else np.array(raw) if input_type == "np" else pd.DataFrame(raw, columns=["x"])
    )
    transformer = CustomBinTransformer(output_dim=bins)
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (4, 1)


def test_custom_bin_transformer_invalid_input():
    transformer = CustomBinTransformer(output_dim=3)
    with pytest.raises(Exception):
        transformer.transform("invalid_input")


def test_custom_bin_transformer_raises_on_invalid_shape():
    transformer = CustomBinTransformer(output_dim=3).fit(np.linspace(0, 1, 10).reshape(-1, 1))

    with pytest.raises(ValueError, match=r"2D array with shape \(n_samples, 1\)"):
        transformer.transform(np.zeros((5, 2)))


def test_custom_bin_transformer_raises_before_fit():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        CustomBinTransformer(output_dim=3).transform(np.linspace(0, 1, 10).reshape(-1, 1))


def test_custom_bin_transformer_invalid_bins_type():
    with pytest.raises(Exception):
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


# --------------------------------------------------------------------------- #
# Bin edges must be learned at fit and reused verbatim.
#
# ``fit`` recorded only ``n_features_in_``; ``transform`` re-derived equal-width
# edges from whatever batch it was handed, so the same value received different
# codes depending on which other rows travelled with it.
# --------------------------------------------------------------------------- #
def test_edges_are_learned_at_fit():
    transformer = CustomBinTransformer(output_dim=4).fit(np.linspace(0, 10, 100).reshape(-1, 1))

    assert hasattr(transformer, "bin_edges_")
    assert len(transformer.bin_edges_[0]) == 5  # 4 bins -> 5 edges
    assert transformer.bin_edges_[0][0] <= 0.0
    assert transformer.bin_edges_[0][-1] >= 10.0


def test_codes_do_not_depend_on_the_transform_batch():
    transformer = CustomBinTransformer(output_dim=4).fit(np.linspace(0, 10, 100).reshape(-1, 1))
    rows = np.array([[0.0], [2.5], [5.0], [7.5], [10.0]])

    alone = transformer.transform(rows)
    with_outlier = transformer.transform(np.vstack([rows, [[100.0]]]))

    np.testing.assert_array_equal(alone.ravel(), with_outlier[:-1].ravel())


def test_codes_reflect_the_fitted_range_not_the_batch():
    transformer = CustomBinTransformer(output_dim=4).fit(np.linspace(0, 10, 100).reshape(-1, 1))

    codes = transformer.transform(np.array([[0.0], [2.5], [5.0], [7.5], [10.0]])).ravel()

    # Evenly spaced points across the fitted [0, 10] range must span the bins.
    np.testing.assert_array_equal(codes, [0, 0, 1, 2, 3])


def test_single_row_transform_is_supported():
    transformer = CustomBinTransformer(output_dim=4).fit(np.linspace(0, 10, 100).reshape(-1, 1))

    out = transformer.transform(np.array([[5.0]]))

    assert out.shape == (1, 1)
    assert out[0, 0] == transformer.transform(np.array([[0.0], [5.0], [10.0]]))[1, 0]


def test_out_of_range_values_clip_into_the_end_bins():
    transformer = CustomBinTransformer(output_dim=4).fit(np.linspace(0, 10, 100).reshape(-1, 1))

    codes = transformer.transform(np.array([[-50.0], [5.0], [50.0]])).ravel()

    assert codes[0] == 0
    assert codes[2] == 3
    assert not np.isnan(codes).any()


def test_explicit_edges_are_stored_and_reused():
    edges = [0.0, 0.5, 1.0]
    transformer = CustomBinTransformer(output_dim=edges).fit(np.linspace(0, 1, 20).reshape(-1, 1))

    np.testing.assert_allclose(transformer.bin_edges_[0], edges)
    np.testing.assert_array_equal(
        transformer.transform(np.array([[0.1], [0.9]])).ravel(), [0, 1]
    )


def test_string_input_raises_at_fit():
    from pretab.core.exceptions import PretabDataError

    with pytest.raises(PretabDataError, match="requires numeric input"):
        CustomBinTransformer(output_dim=3).fit(np.array([["a"], ["b"], ["c"]], dtype=object))
