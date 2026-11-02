import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp
from sklearn.pipeline import Pipeline

from pretab.compose.factory import get_categorical_transformer_steps


def _build(method, **kwargs):
    return Pipeline(get_categorical_transformer_steps(method, add_imputer=False, **kwargs))


def _feature_names(pipe):
    names = pipe.get_feature_names_out()
    assert names is not None
    return names.tolist()


def test_one_hot_ignores_unseen_categories():
    pipe = _build("one-hot", sparse_output=False)
    pipe.fit(np.array([["A"], ["B"], ["C"]]))

    # A category absent at fit time must not crash; it yields an all-zero row.
    Xt = pipe.transform(np.array([["A"], ["D"]]))

    assert Xt.shape == (2, 3)
    np.testing.assert_array_equal(Xt[1], np.zeros(3))
    assert Xt[0].sum() == 1


def test_one_hot_handle_unknown_override():
    pipe = _build("one-hot", handle_unknown="error")

    pipe.fit(np.array([["A"], ["B"]]))
    with pytest.raises(ValueError):
        pipe.transform(np.array([["C"]]))


@pytest.mark.parametrize("marker", [None, pd.NA, np.nan], ids=["None", "pd.NA", "NaN"])
def test_one_hot_missing_after_clean_fit_matches_unseen(marker):
    pipe = _build("one-hot", sparse_output=False).fit(pd.DataFrame({"c": ["a", "b"]}))

    result = pipe.transform(pd.DataFrame({"c": [marker, "unseen"]}))

    np.testing.assert_array_equal(result, np.zeros((2, 2)))


@pytest.mark.parametrize("marker", [None, pd.NA, np.nan], ids=["None", "pd.NA", "NaN"])
def test_one_hot_all_missing_matches_nan(marker):
    frame = pd.DataFrame({"c": [marker, marker, marker]}, dtype=object)
    reference = pd.DataFrame({"c": [np.nan, np.nan, np.nan]}, dtype=object)
    expected = _build("one-hot", sparse_output=False).fit(reference)
    actual = _build("one-hot", sparse_output=False).fit(frame)

    assert _feature_names(actual) == _feature_names(expected)
    np.testing.assert_array_equal(actual.transform(frame), expected.transform(reference))


def test_one_hot_explicit_none_category_is_preserved():
    categories = [["a", "b", None]]
    pipe = _build("one-hot", categories=categories, sparse_output=False)
    frame = pd.DataFrame({"c": ["a", None, "b"]})

    result = pipe.fit_transform(frame)

    np.testing.assert_array_equal(result, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    assert _feature_names(pipe) == ["c_a", "c_b", "c_None"]
    assert categories == [["a", "b", None]]


def test_one_hot_explicit_none_drop_is_preserved():
    drop = [None]
    pipe = _build("one-hot", drop=drop, sparse_output=False)

    result = pipe.fit_transform(pd.DataFrame({"c": ["a", None, "b"]}))

    np.testing.assert_array_equal(result, [[1, 0], [0, 0], [0, 1]])
    assert _feature_names(pipe) == ["c_a", "c_b"]
    assert drop == [None]


@pytest.mark.parametrize("add_imputer", [True, False])
@pytest.mark.parametrize("nullable", [True, False])
def test_one_hot_unsorted_numeric_categories_are_rejected(add_imputer, nullable):
    frame = pd.DataFrame({"c": pd.Series([1, 2], dtype="Int64")}) if nullable else np.array([[1], [2]])
    pipe = Pipeline(
        get_categorical_transformer_steps("one-hot", add_imputer=add_imputer, categories=[[2, 1]], sparse_output=False)
    )

    with pytest.raises(ValueError, match="Unsorted categories"):
        pipe.fit(frame)


@pytest.mark.parametrize("add_imputer", [True, False])
def test_one_hot_clean_numeric_fit_with_missing_at_transform(add_imputer):
    reference = pd.DataFrame({"c": [1.0, 2.0]})
    frame = pd.DataFrame({"c": [1, 2]})
    options = {"add_imputer": add_imputer, "sparse_output": False}
    expected = Pipeline(get_categorical_transformer_steps("one-hot", **options)).fit(reference)
    actual = Pipeline(get_categorical_transformer_steps("one-hot", **options)).fit(frame)

    np.testing.assert_array_equal(
        actual.transform(pd.DataFrame({"c": [None, 2]})),
        expected.transform(pd.DataFrame({"c": [np.nan, 2.0]})),
    )


def test_one_hot_pandas_sparse_columns_keep_native_support():
    frame = pd.DataFrame({"c": pd.arrays.SparseArray([1, 0, 2], fill_value=0)})
    original = frame.copy(deep=True)
    dense = pd.DataFrame({"c": [1, 0, 2]})
    actual = _build("one-hot", sparse_output=False).fit(frame)
    expected = _build("one-hot", sparse_output=False).fit(dense)

    np.testing.assert_array_equal(actual.transform(frame), expected.transform(dense))
    assert _feature_names(actual) == _feature_names(expected)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("fill_value", [0.0, 1.0, np.nan], ids=["zero", "nonzero", "NaN"])
def test_one_hot_pandas_sparse_missing_keeps_native_support(fill_value):
    frame = pd.DataFrame({"c": pd.arrays.SparseArray([1.0, np.nan, 2.0], fill_value=fill_value)})
    original = frame.copy(deep=True)
    dense = pd.DataFrame({"c": [1.0, np.nan, 2.0]})
    actual = _build("one-hot", sparse_output=False).fit(frame)
    expected = _build("one-hot", sparse_output=False).fit(dense)

    np.testing.assert_array_equal(actual.transform(frame), expected.transform(dense))
    assert _feature_names(actual) == _feature_names(expected)
    pd.testing.assert_frame_equal(frame, original)


def test_categorical_imputation_preserves_sparse_input_support():
    frame = sp.csr_matrix([[1.0], [np.nan], [2.0]])
    original = frame.copy()
    pipe = Pipeline(get_categorical_transformer_steps("none"))

    result = pipe.fit_transform(frame)

    assert sp.issparse(result)
    np.testing.assert_array_equal(result.toarray(), [[1.0], [1.0], [2.0]])
    np.testing.assert_array_equal(frame.toarray(), original.toarray())
