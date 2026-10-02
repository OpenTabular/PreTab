"""Pandas missing markers share the existing NaN preprocessing contracts."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from pretab import Preprocessor


def _frame():
    return pd.DataFrame({"x": [0.1, 0.4, 0.2, 0.9, 0.5, 0.7], "c": ["a", np.nan, "b", "a", "b", "a"]})


@pytest.mark.parametrize("categorical_method", ["int", "one-hot"])
@pytest.mark.parametrize("impute", [True, False])
def test_nullable_categorical_fit_matches_nan(categorical_method, impute):
    frame = _frame()
    options: dict[str, Any] = {"numerical_method": "minmax", "categorical_method": categorical_method}
    if not impute:
        options.update(numerical_imputation=None, categorical_imputation=None)
    expected = Preprocessor(**options).fit(frame)
    actual = Preprocessor(**options).fit(frame.convert_dtypes())

    assert actual.get_feature_names_out().tolist() == expected.get_feature_names_out().tolist()
    np.testing.assert_allclose(
        np.asarray(actual.transform(frame.convert_dtypes()), dtype=float),
        np.asarray(expected.transform(frame), dtype=float),
    )


@pytest.mark.parametrize("missing_options", [{}, {"add_missing_indicator": True}, {"missing_policy": "separate_state"}])
def test_object_none_imputation_matches_nan(missing_options):
    reference = _frame()
    frame = reference.copy()
    frame.loc[1, "c"] = None
    options: dict[str, Any] = {"numerical_method": "minmax", "categorical_method": "one-hot", **missing_options}
    expected = Preprocessor(**options).fit(reference)
    actual = Preprocessor(**options).fit(frame)

    assert actual.get_feature_names_out().tolist() == expected.get_feature_names_out().tolist()
    np.testing.assert_allclose(
        np.asarray(actual.transform(frame), dtype=float), np.asarray(expected.transform(reference), dtype=float)
    )


@pytest.mark.parametrize("categorical_method", ["int", "one-hot"])
@pytest.mark.parametrize("missing_policy", [None, "propagate", "separate_state"])
def test_mixed_categorical_missing_matches_nan(categorical_method, missing_policy):
    frame = pd.DataFrame({"c": ["b", None, "a", pd.NA, np.nan, "a"]}, index=pd.Index([7, 2, 19, 4, 30, 1]))
    original = frame.copy(deep=True)
    reference = frame.copy()
    reference.loc[[2, 4, 30], "c"] = np.nan
    options: dict[str, Any] = {"categorical_method": categorical_method, "missing_policy": missing_policy}
    expected = Preprocessor(**options).fit(reference)
    actual = Preprocessor(**options).fit(frame)

    assert actual.get_feature_names_out().tolist() == expected.get_feature_names_out().tolist()
    np.testing.assert_allclose(
        np.asarray(actual.transform(frame), dtype=float), np.asarray(expected.transform(reference), dtype=float)
    )
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("dtype", ["Float64", "Int64"])
def test_nullable_numerical_imputation_matches_nan(dtype):
    reference = pd.DataFrame({"x": [1.0, np.nan, 3.0, 5.0], "c": ["a", "b", "a", "b"]}, index=pd.Index([8, 2, 6, 1]))
    frame = reference.copy()
    frame["x"] = frame["x"].astype(dtype)
    original = frame.copy(deep=True)
    options: dict[str, Any] = {"numerical_method": "minmax", "missing_policy": "separate_state"}
    expected = Preprocessor(**options).fit(reference)
    actual = Preprocessor(**options).fit(frame)

    assert list(actual.get_feature_info(verbose=False)[0]) == ["x"]
    assert actual.get_feature_names_out().tolist() == expected.get_feature_names_out().tolist()
    np.testing.assert_allclose(
        np.asarray(actual.transform(frame), dtype=float), np.asarray(expected.transform(reference), dtype=float)
    )
    pd.testing.assert_frame_equal(frame, original)


def test_nullable_one_hot_clone_names_and_pandas_rows():
    frame = pd.DataFrame({"c": ["a", None, "b", "a"]}, index=pd.Index([17, 3, 9, 6])).convert_dtypes()
    original = frame.copy(deep=True)
    cloned = clone(Preprocessor(categorical_method="one-hot", categorical_imputation=None))
    assert isinstance(cloned, Preprocessor)
    fitted = cloned.fit(frame)

    assert fitted.get_feature_names_out().tolist() == ["cat_c_a", "cat_c_b", "cat_c_nan"]
    expected = fitted.transform(frame)
    fitted.set_output(transform="pandas")
    output = fitted.transform(frame)
    assert isinstance(output, pd.DataFrame)
    np.testing.assert_array_equal(output.to_numpy(), expected)
    assert output.columns.tolist() == fitted.get_feature_names_out().tolist()
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("marker", [None, pd.NA, np.nan], ids=["None", "pd.NA", "NaN"])
@pytest.mark.parametrize("return_array", [True, False])
def test_propagate_none_retains_raw_marker(marker, return_array):
    frame = pd.DataFrame({"c": ["a", marker, "b"]}, dtype=object)
    original = frame.copy(deep=True)
    fitted = Preprocessor(categorical_method="none", missing_policy="propagate").fit(frame)
    transformed = fitted.transform(frame, return_array=return_array)
    result = transformed if return_array else transformed["cat_c"]

    assert result[0, 0] == "a"
    assert result[2, 0] == "b"
    if marker is None or marker is pd.NA:
        assert result[1, 0] is marker
    else:
        assert np.isnan(result[1, 0])
    pd.testing.assert_frame_equal(frame, original)
