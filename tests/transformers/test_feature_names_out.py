"""A3: ``get_feature_names_out(None)`` defaults to generated ``x0, x1, ...`` names
instead of raising, for ``NoTransformer``, ``ToFloatTransformer`` and
``ContinuousOrdinalTransformer``.
"""

import numpy as np
import pytest

from pretab.transformers import (
    ContinuousOrdinalTransformer,
    NoTransformer,
    ToFloatTransformer,
)


def _num():
    return [NoTransformer(), ToFloatTransformer()]


@pytest.mark.parametrize("transformer", _num())
def test_numeric_encoders_default_names(transformer):
    transformer.fit(np.zeros((5, 2)))
    np.testing.assert_array_equal(transformer.get_feature_names_out(), np.asarray(["x0", "x1"], dtype=object))


@pytest.mark.parametrize("transformer", _num())
def test_numeric_encoders_passthrough_names(transformer):
    transformer.fit(np.zeros((5, 2)))
    np.testing.assert_array_equal(
        transformer.get_feature_names_out(["a", "b"]),
        np.asarray(["a", "b"], dtype=object),
    )


def test_continuous_ordinal_default_names():
    X = np.array([["a", "x"], ["b", "y"], ["a", "x"]], dtype=object)
    transformer = ContinuousOrdinalTransformer().fit(X)
    np.testing.assert_array_equal(transformer.get_feature_names_out(), np.asarray(["x0", "x1"], dtype=object))


def test_continuous_ordinal_passthrough_names():
    X = np.array([["a", "x"], ["b", "y"], ["a", "x"]], dtype=object)
    transformer = ContinuousOrdinalTransformer().fit(X)
    np.testing.assert_array_equal(
        transformer.get_feature_names_out(["c1", "c2"]),
        np.asarray(["c1", "c2"], dtype=object),
    )
