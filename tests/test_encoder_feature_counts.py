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
