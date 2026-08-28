"""Contract tests for the standalone :class:`PeriodicEncodingTransformer`.

``PeriodicEncodingTransformer`` preserves the row count but requires a per-feature
``period`` argument and constrains its inputs to ``[0, period]``, so it is applied
directly rather than wired into the ``Preprocessor`` pipeline.

These tests pin the intended behaviour: the exact output shapes/values, the
generated feature names, and the input-range guard.
"""

import numpy as np
import pytest

from pretab.exceptions import InvalidParamError, PretabDataError
from pretab.transformers import PeriodicEncodingTransformer


def test_cyclic_preserves_rows_and_pins_values():
    X = np.array([[0], [6], [12], [18]])
    out = PeriodicEncodingTransformer(period=24).fit_transform(X)
    # Row count preserved; columns are (sin, cos).
    assert out.shape == (4, 2)
    expected_angle = 2 * np.pi * X.ravel() / 24
    np.testing.assert_allclose(out[:, 0], np.sin(expected_angle), atol=1e-12)
    np.testing.assert_allclose(out[:, 1], np.cos(expected_angle), atol=1e-12)


def test_cyclic_rejects_out_of_range_input():
    transformer = PeriodicEncodingTransformer(period=24)
    with pytest.raises(PretabDataError):
        transformer.fit(np.array([[25]]))
    with pytest.raises(PretabDataError):
        transformer.fit(np.array([[-1]]))


def test_cyclic_requires_period():
    with pytest.raises(TypeError):
        PeriodicEncodingTransformer()  # type: ignore[call-arg]


def test_cyclic_feature_names():
    X = np.array([[0], [6], [12], [18]])
    transformer = PeriodicEncodingTransformer(period=24).fit(X)
    np.testing.assert_array_equal(transformer.get_feature_names_out(["hour"]), ["hour_cyclic0", "hour_cyclic1"])


def test_cyclic_harmonics_expand_columns():
    X = np.array([[0], [6], [12], [18]])
    transformer = PeriodicEncodingTransformer(period=24, harmonics=3).fit(X)
    out = transformer.transform(X)
    # 3 harmonic pairs -> 6 columns; layout is (sin_h, cos_h) in ascending order.
    assert out.shape == (4, 6)
    assert transformer.total_output_dim_ == 6
    for h in range(1, 4):
        angle = 2 * np.pi * h * X.ravel() / 24
        np.testing.assert_allclose(out[:, 2 * (h - 1)], np.sin(angle), atol=1e-12)
        np.testing.assert_allclose(out[:, 2 * (h - 1) + 1], np.cos(angle), atol=1e-12)


def test_cyclic_include_original_prepends_raw_value():
    X = np.array([[0], [6], [12], [18]])
    transformer = PeriodicEncodingTransformer(period=24, harmonics=2, include_original=True).fit(X)
    out = transformer.transform(X)
    # 1 original + 2 harmonic pairs = 5 columns per feature.
    assert out.shape == (4, 5)
    assert transformer.total_output_dim_ == 5
    np.testing.assert_allclose(out[:, 0], X.ravel(), atol=1e-12)


def test_cyclic_rejects_non_positive_harmonics():
    X = np.array([[0], [6], [12], [18]])
    with pytest.raises(InvalidParamError):
        PeriodicEncodingTransformer(period=24, harmonics=0).fit(X)
