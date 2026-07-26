"""Unit tests for :mod:`pretab.compose.feature_detection`."""

import numpy as np
import pandas as pd
import pytest

from pretab.compose.feature_detection import detect_column_types, to_dataframe
from pretab.exceptions import InvalidParamError


def test_to_dataframe_wraps_ndarray_with_feature_names():
    df = to_dataframe(np.zeros((2, 3)))
    assert list(df.columns) == ["feature_0", "feature_1", "feature_2"]


def test_to_dataframe_wraps_dict():
    df = to_dataframe({"a": [1, 2], "b": [3, 4]})
    assert list(df.columns) == ["a", "b"]


def test_to_dataframe_returns_same_object_without_copy():
    df = pd.DataFrame({"a": [1, 2]})
    assert to_dataframe(df) is df
    assert to_dataframe(df, copy=True) is not df


def test_float_cutoff_uses_unique_ratio():
    df = pd.DataFrame({"x": [1, 2, 3, 1, 2, 3]})  # 3 unique of 6 -> ratio 0.5
    num, cat = detect_column_types(df, cat_cutoff=0.6, treat_all_integers_as_numerical=False)
    assert cat == ["x"] and num == []
    num, cat = detect_column_types(df, cat_cutoff=0.4, treat_all_integers_as_numerical=False)
    assert num == ["x"] and cat == []


def test_int_cutoff_uses_absolute_count():
    df = pd.DataFrame({"x": [1, 2, 3, 1, 2, 3]})  # 3 unique
    _, cat = detect_column_types(df, cat_cutoff=4, treat_all_integers_as_numerical=False)
    assert cat == ["x"]
    num, _ = detect_column_types(df, cat_cutoff=2, treat_all_integers_as_numerical=False)
    assert num == ["x"]


def test_treat_all_integers_as_numerical_overrides_cutoff():
    df = pd.DataFrame({"x": [1, 2, 3, 1, 2, 3]})
    num, cat = detect_column_types(df, cat_cutoff=0.9, treat_all_integers_as_numerical=True)
    assert num == ["x"] and cat == []


def test_object_dtype_is_always_categorical():
    df = pd.DataFrame({"c": ["a", "b", "c", "d", "e", "f"]})
    _, cat = detect_column_types(df, cat_cutoff=0.01, treat_all_integers_as_numerical=False)
    assert cat == ["c"]


def test_float_columns_are_numerical():
    df = pd.DataFrame({"f": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    num, _ = detect_column_types(df, cat_cutoff=0.9, treat_all_integers_as_numerical=False)
    assert num == ["f"]


def test_invalid_cat_cutoff_type_raises():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(InvalidParamError):
        detect_column_types(df, cat_cutoff="bad", treat_all_integers_as_numerical=False)
