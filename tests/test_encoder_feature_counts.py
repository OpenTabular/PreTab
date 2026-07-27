"""A2: ``n_features_in_`` must reflect the fitted column count, not a hardcoded 1.

Covers ``NoTransformer``, ``ToFloatTransformer`` and ``CustomBinTransformer``.
"""

import numpy as np
import pytest

from pretab.transformers import (
    CustomBinTransformer,
    NoTransformer,
    ToFloatTransformer,
)


@pytest.mark.parametrize("n_cols", [1, 2, 3])
def test_no_transformer_records_feature_count(n_cols):
    X = np.zeros((5, n_cols))
    assert NoTransformer().fit(X).n_features_in_ == n_cols


@pytest.mark.parametrize("n_cols", [1, 2, 3])
def test_to_float_transformer_records_feature_count(n_cols):
    X = np.zeros((5, n_cols))
    assert ToFloatTransformer().fit(X).n_features_in_ == n_cols


def test_custom_bin_transformer_records_single_feature():
    X = np.linspace(0, 1, 10).reshape(-1, 1)
    assert CustomBinTransformer(output_dim=4).fit(X).n_features_in_ == 1


def test_custom_bin_transformer_reads_actual_column_count():
    # Proves the value is derived from X, not hardcoded to 1.
    X = np.zeros((10, 2))
    assert CustomBinTransformer(output_dim=4).fit(X).n_features_in_ == 2


# --------------------------------------------------------------------------- #
# RaiseOnNaNTransformer: the single enforcement point for handle_missing="error"
# --------------------------------------------------------------------------- #
def test_raise_on_nan_passes_clean_data_through():
    from pretab.transformers import RaiseOnNaNTransformer

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = RaiseOnNaNTransformer().fit_transform(X)

    np.testing.assert_array_equal(out, X)


def test_raise_on_nan_rejects_missing_values_at_fit():
    from pretab.core.exceptions import PretabDataError
    from pretab.transformers import RaiseOnNaNTransformer

    with pytest.raises(PretabDataError, match="handle_missing"):
        RaiseOnNaNTransformer().fit(np.array([[1.0], [np.nan]]))


def test_raise_on_nan_rejects_missing_values_at_transform():
    from pretab.core.exceptions import PretabDataError
    from pretab.transformers import RaiseOnNaNTransformer

    transformer = RaiseOnNaNTransformer().fit(np.array([[1.0], [2.0]]))

    with pytest.raises(PretabDataError, match="handle_missing"):
        transformer.transform(np.array([[1.0], [np.nan]]))


@pytest.mark.parametrize("n_cols", [1, 2, 3])
def test_raise_on_nan_records_feature_count(n_cols):
    from pretab.transformers import RaiseOnNaNTransformer

    assert RaiseOnNaNTransformer().fit(np.zeros((5, n_cols))).n_features_in_ == n_cols
