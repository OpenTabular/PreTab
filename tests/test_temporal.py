"""Contract tests for the standalone temporal transformers.

The temporal transformers are documented as standalone time-series utilities that
are deliberately *not* wired into the ``Preprocessor`` pipeline:

* ``LagFeatureTransformer`` and ``RollingStatsTransformer`` intentionally change
  the row count (they drop the initial, incomplete windows) and assume the rows
  are ordered in time, so they cannot live inside the ``ColumnTransformer`` the
  preprocessor builds.
* ``CyclicalTimeTransformer`` preserves the row count but requires a per-feature
  ``period`` argument and constrains its inputs, so it is also applied directly.

These tests pin that intended behaviour: the exact output shapes/values, the
row-count semantics, the generated feature names, and the input-range guard.
Error paths (insufficient samples, unsupported stat) are covered in
``tests/test_exceptions.py``.
"""

import numpy as np
import pytest

from pretab.core.exceptions import PretabDataError
from pretab.transformers import (
    CyclicalTimeTransformer,
    LagFeatureTransformer,
    RollingStatsTransformer,
)

# --------------------------------------------------------------------------- #
# LagFeatureTransformer
# --------------------------------------------------------------------------- #


def test_lag_reduces_rows_and_pins_values():
    X = np.arange(6).reshape(-1, 1)
    out = LagFeatureTransformer(n_lags=2).fit_transform(X)
    # n_samples - n_lags rows; columns are (lag-1, lag-2).
    assert out.shape == (4, 2)
    np.testing.assert_array_equal(out, [[1, 0], [2, 1], [3, 2], [4, 3]])


def test_lag_default_single_lag():
    X = np.arange(5).reshape(-1, 1)
    out = LagFeatureTransformer().fit_transform(X)
    assert out.shape == (4, 1)
    np.testing.assert_array_equal(out.ravel(), [0, 1, 2, 3])


def test_lag_feature_names():
    X = np.arange(6).reshape(-1, 1)
    transformer = LagFeatureTransformer(n_lags=2).fit(X)
    np.testing.assert_array_equal(
        transformer.get_feature_names_out(["t"]), ["t_lag0", "t_lag1"]
    )


# --------------------------------------------------------------------------- #
# RollingStatsTransformer
# --------------------------------------------------------------------------- #


def test_rolling_reduces_rows_and_pins_mean():
    X = np.arange(10).reshape(-1, 1).astype(float)
    out = RollingStatsTransformer(window_size=3, stats=("mean",)).fit_transform(X)
    # n_samples - window_size + 1 rows.
    assert out.shape == (8, 1)
    np.testing.assert_allclose(out.ravel(), np.arange(1, 9, dtype=float))


def test_rolling_min_max_columns():
    X = np.arange(10).reshape(-1, 1).astype(float)
    out = RollingStatsTransformer(window_size=3, stats=("min", "max")).fit_transform(X)
    assert out.shape == (8, 2)
    np.testing.assert_allclose(out[:, 0], np.arange(0, 8, dtype=float))  # min
    np.testing.assert_allclose(out[:, 1], np.arange(2, 10, dtype=float))  # max


def test_rolling_feature_names():
    X = np.arange(10).reshape(-1, 1).astype(float)
    transformer = RollingStatsTransformer(window_size=3, stats=("mean", "std")).fit(X)
    np.testing.assert_array_equal(
        transformer.get_feature_names_out(["t"]), ["t_roll0", "t_roll1"]
    )


# --------------------------------------------------------------------------- #
# CyclicalTimeTransformer
# --------------------------------------------------------------------------- #


def test_cyclic_preserves_rows_and_pins_values():
    X = np.array([[0], [6], [12], [18]])
    out = CyclicalTimeTransformer(period=24).fit_transform(X)
    # Row count preserved; columns are (sin, cos).
    assert out.shape == (4, 2)
    expected_angle = 2 * np.pi * X.ravel() / 24
    np.testing.assert_allclose(out[:, 0], np.sin(expected_angle), atol=1e-12)
    np.testing.assert_allclose(out[:, 1], np.cos(expected_angle), atol=1e-12)


def test_cyclic_rejects_out_of_range_input():
    transformer = CyclicalTimeTransformer(period=24)
    with pytest.raises(PretabDataError):
        transformer.fit(np.array([[25]]))
    with pytest.raises(PretabDataError):
        transformer.fit(np.array([[-1]]))


def test_cyclic_requires_period():
    with pytest.raises(TypeError):
        CyclicalTimeTransformer()  # type: ignore[call-arg]


def test_cyclic_feature_names():
    X = np.array([[0], [6], [12], [18]])
    transformer = CyclicalTimeTransformer(period=24).fit(X)
    np.testing.assert_array_equal(
        transformer.get_feature_names_out(["hour"]), ["hour_cyclic0", "hour_cyclic1"]
    )


# --------------------------------------------------------------------------- #
# Output names must line up with the columns ``transform`` actually produces.
#
# All three transformers hstack one (n_rows, n_features) block per lag / stat /
# component, so the columns are block-major. The inherited default in
# ``BasePreTabTransformer`` emits feature-major names, which mislabelled every
# column but the first and last whenever there was more than one input feature.
# --------------------------------------------------------------------------- #
@pytest.fixture
def two_features():
    # B is always 100x A, so each column is identifiable from its value alone.
    return np.column_stack([np.arange(8.0), np.arange(8.0) * 100])


def test_lag_names_match_column_order(two_features):
    transformer = LagFeatureTransformer(n_lags=2).fit(two_features)

    names = list(transformer.get_feature_names_out(["A", "B"]))
    assert names == ["A_lag0", "B_lag0", "A_lag1", "B_lag1"]

    row = transformer.transform(two_features)[0]
    # Column 1 holds B's lag-1 value (100), so it must be named for B.
    assert row[1] == 100.0
    assert names[1].startswith("B")


def test_rolling_names_match_column_order(two_features):
    transformer = RollingStatsTransformer(window_size=3, stats=("mean", "max")).fit(two_features)

    names = list(transformer.get_feature_names_out(["A", "B"]))
    assert names == ["A_roll0", "B_roll0", "A_roll1", "B_roll1"]

    row = transformer.transform(two_features)[0]
    assert row.tolist() == [1.0, 100.0, 2.0, 200.0]


def test_cyclic_names_match_column_order():
    X = np.column_stack([np.arange(8.0), np.arange(8.0)])
    transformer = CyclicalTimeTransformer(period=8).fit(X)

    names = list(transformer.get_feature_names_out(["A", "B"]))
    assert names == ["A_cyclic0", "B_cyclic0", "A_cyclic1", "B_cyclic1"]

    out = transformer.transform(X)
    np.testing.assert_allclose(out[:, :2], np.sin(2 * np.pi * X / 8))
    np.testing.assert_allclose(out[:, 2:], np.cos(2 * np.pi * X / 8))


@pytest.mark.parametrize(
    ("transformer", "expected"),
    [
        (LagFeatureTransformer(n_lags=2), ["x0_lag0", "x0_lag1"]),
        (RollingStatsTransformer(window_size=3, stats=("mean",)), ["x0_roll0"]),
        (CyclicalTimeTransformer(period=8), ["x0_cyclic0", "x0_cyclic1"]),
    ],
)
def test_default_names_are_generated_for_single_feature(transformer, expected):
    X = np.arange(8.0).reshape(-1, 1)
    assert list(transformer.fit(X).get_feature_names_out()) == expected


def test_names_count_matches_transform_width(two_features):
    for transformer in (
        LagFeatureTransformer(n_lags=3),
        RollingStatsTransformer(window_size=3, stats=("mean", "std", "min")),
    ):
        transformer.fit(two_features)
        assert len(transformer.get_feature_names_out()) == transformer.transform(two_features).shape[1]
