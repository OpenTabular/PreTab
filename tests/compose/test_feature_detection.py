"""Unit tests for :mod:`pretab.compose.feature_detection`."""

import numpy as np
import pandas as pd
import pytest

from pretab.compose.feature_detection import (
    bool_columns_as_object,
    detect_column_types,
    to_dataframe,
    with_string_labels,
)
from pretab.exceptions import InvalidParamError, PretabDataError


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


def test_to_dataframe_rejects_duplicate_columns():
    """Regression guard for issue #37: a duplicate column label must raise a

    clear PretabDataError instead of an opaque AttributeError deep inside
    column-type detection.
    """
    df = pd.DataFrame(np.column_stack([np.zeros(5), np.ones(5)]), columns=pd.Index(["a", "a"]))
    with pytest.raises(PretabDataError, match=r"Duplicate column names.*\['a'\]"):
        to_dataframe(df)


def test_to_dataframe_converts_polars_frame_with_names_order_and_dtypes():
    """A polars frame used to be treated as pandas and fail on ``columns.duplicated``."""
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame(
        {
            "z_float": [0.5, None, 2.5],
            "count": [3, 1, 2],
            "count_with_null": [3, None, 2],
            "flag": [True, False, True],
            "flag_with_null": [True, None, False],
            "city": ["b", "a", None],
        }
    ).with_columns(pl.col("city").cast(pl.Categorical).alias("city_cat"))

    df = to_dataframe(frame)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == frame.columns
    assert df.dtypes.to_dict() == {
        "z_float": np.float64,
        "count": np.int64,
        "count_with_null": np.float64,
        "flag": bool,
        "flag_with_null": object,
        "city": object,
        "city_cat": object,
    }
    assert df["count"].tolist() == [3, 1, 2]
    assert df["city"].tolist()[:2] == ["b", "a"] and df["city_cat"].tolist()[:2] == ["b", "a"]
    # Polars nulls become NaN, the missing marker pandas and the imputers use.
    assert df.isna().sum().to_dict() == {
        "z_float": 1,
        "count": 0,
        "count_with_null": 1,
        "flag": 0,
        "flag_with_null": 1,
        "city": 1,
        "city_cat": 1,
    }
    assert np.isnan(df["city"][2]) and np.isnan(df["flag_with_null"][1])


def test_detect_column_types_on_polars_frame():
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame(
        {
            "f": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "many": [1, 2, 3, 4, 5, 6],
            "few": [1, 2, 1, 2, 1, 2],
            "s": ["a", "b", "a", "b", "a", "b"],
            "flag": [True, False, True, False, True, False],
        }
    ).with_columns(pl.col("s").cast(pl.Categorical).alias("cat"))
    num, cat = detect_column_types(frame, cat_cutoff=0.5, treat_all_integers_as_numerical=False)
    assert num == ["f", "many"]
    assert cat == ["few", "s", "flag", "cat"]


def test_to_dataframe_reads_a_list_of_rows_as_an_array():
    df = to_dataframe([[1.0, 2.0], [3.0, 4.0]])
    assert list(df.columns) == ["feature_0", "feature_1"]
    np.testing.assert_array_equal(df.to_numpy(), [[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize(
    "X, match",
    [
        (pd.Series([1.0, 2.0]), "got a pandas Series"),
        (np.arange(3.0), "got an array with 1 dimension"),
        ([1.0, 2.0], "got an array with 1 dimension"),
    ],
)
def test_to_dataframe_rejects_input_that_is_not_2d(X, match):
    with pytest.raises(PretabDataError, match=match):
        to_dataframe(X)


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


@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
@pytest.mark.parametrize(
    "cat_cutoff, expected_numerical, expected_categorical",
    [
        (0.6, [], ["x"]),
        (0.4, ["x"], []),
        (4, [], ["x"]),
        (2, ["x"], []),
    ],
)
def test_signed_and_unsigned_integers_follow_same_cutoff(dtype, cat_cutoff, expected_numerical, expected_categorical):
    df = pd.DataFrame({"x": np.array([1, 2, 3, 1, 2, 3], dtype=dtype)})
    numerical, categorical = detect_column_types(
        df,
        cat_cutoff=cat_cutoff,
        treat_all_integers_as_numerical=False,
    )
    assert numerical == expected_numerical
    assert categorical == expected_categorical


def test_treat_all_unsigned_integers_as_numerical_overrides_cutoff():
    df = pd.DataFrame({"x": np.array([0, 1, 0, 1], dtype=np.uint8)})
    numerical, categorical = detect_column_types(
        df,
        cat_cutoff=0.9,
        treat_all_integers_as_numerical=True,
    )
    assert numerical == ["x"]
    assert categorical == []


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


def test_bool_columns_as_object_casts_only_boolean_columns():
    frame = pd.DataFrame({"x": [0.5, 1.5], "flag": [True, False], "nullable": pd.array([True, None], dtype="boolean")})
    cast = bool_columns_as_object(frame)
    assert cast["x"].dtype == np.float64
    assert cast["flag"].dtype == object and cast["flag"].tolist() == [True, False]
    assert cast["nullable"].dtype == object
    assert cast["nullable"][0] is True and np.isnan(cast["nullable"][1])
    # The caller's frame keeps its dtypes.
    assert frame["flag"].dtype == bool and frame["nullable"].dtype == "boolean"


def test_bool_columns_as_object_returns_frame_without_bool_columns_unchanged():
    frame = pd.DataFrame({"x": [0.5, 1.5]})
    assert bool_columns_as_object(frame) is frame


def test_with_string_labels_relabels_without_touching_the_input():
    frame = pd.DataFrame({1: [1.0], "b": [2.0]})
    relabelled = with_string_labels(frame)
    assert list(relabelled.columns) == ["1", "b"]
    assert list(frame.columns) == [1, "b"]
    string_frame = pd.DataFrame({"a": [1.0]})
    assert with_string_labels(string_frame) is string_frame
