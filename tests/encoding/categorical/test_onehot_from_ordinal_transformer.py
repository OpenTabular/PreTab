import numpy as np
import pandas as pd
import pytest

from pretab import Preprocessor
from pretab.exceptions import PretabDataError
from pretab.transformers import OneHotFromOrdinalTransformer


def test_onehot_from_ordinal_single_feature():
    X = np.array([[0], [1], [2], [1]])
    transformer = OneHotFromOrdinalTransformer()
    transformer.fit(X)
    Xt = transformer.transform(X)

    expected = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
        ]
    )
    assert Xt.shape == (4, 3)
    np.testing.assert_array_equal(Xt, expected)


def test_onehot_from_ordinal_multi_feature():
    X = np.array(
        [
            [0, 1],
            [1, 0],
            [2, 1],
            [1, 2],
        ]
    )
    transformer = OneHotFromOrdinalTransformer()
    transformer.fit(X)
    Xt = transformer.transform(X)

    assert Xt.shape == (4, 3 + 3)  # 3 bins for each feature
    assert np.all((Xt == 0) | (Xt == 1))
    assert np.all(Xt.sum(axis=1) == 2)


def test_onehot_from_ordinal_consistent_output_shape():
    X = np.random.randint(0, 4, size=(10, 5))
    transformer = OneHotFromOrdinalTransformer()
    transformer.fit(X)
    Xt = transformer.transform(X)

    expected_dim = sum(transformer.max_bins_)
    assert Xt.shape == (10, expected_dim)


def test_onehot_get_feature_names():
    X = np.array([[0, 1], [2, 0]])
    transformer = OneHotFromOrdinalTransformer()
    transformer.fit(X)
    names = transformer.get_feature_names_out(["a", "b"])

    expected = np.array(["a_bin_0", "a_bin_1", "a_bin_2", "b_bin_0", "b_bin_1"])
    assert names.shape == (5,)
    np.testing.assert_array_equal(names, expected)


def test_onehot_transform_raises_if_not_fit():
    X = np.array([[0, 1]])
    transformer = OneHotFromOrdinalTransformer()
    with pytest.raises(AttributeError):
        transformer.transform(X)


def test_onehot_from_ordinal_unseen_larger_code_gives_zero_row():
    transformer = OneHotFromOrdinalTransformer()
    transformer.fit(np.array([[0], [1], [2]]))

    # A code larger than the fitted range must not raise IndexError; it maps to
    # an all-zero row (handle_unknown="ignore" behaviour).
    Xt = transformer.transform(np.array([[1], [3]]))

    expected = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
        ]
    )
    assert Xt.shape == (2, 3)
    np.testing.assert_array_equal(Xt, expected)


def test_onehot_from_ordinal_negative_code_gives_zero_row():
    transformer = OneHotFromOrdinalTransformer()
    transformer.fit(np.array([[0], [1], [2]]))

    # Negative codes must not silently wrap around via numpy indexing.
    Xt = transformer.transform(np.array([[-1], [0]]))

    expected = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
        ]
    )
    np.testing.assert_array_equal(Xt, expected)


def test_onehot_from_ordinal_fit_rejects_non_numeric():
    """Regression guard for issue #17: string input must raise a clear PretabDataError."""
    transformer = OneHotFromOrdinalTransformer()
    with pytest.raises(PretabDataError, match="already ordinal-encoded"):
        transformer.fit(np.array([["a"], ["b"], ["c"]]))


def test_onehot_from_ordinal_transform_rejects_non_numeric():
    """Regression guard for issue #17: same guard applies at transform time."""
    transformer = OneHotFromOrdinalTransformer()
    transformer.fit(np.array([[0], [1], [2]]))
    with pytest.raises(PretabDataError, match="already ordinal-encoded"):
        transformer.transform(np.array([["a"], ["b"]]))


def test_preprocessor_onehot_from_ordinal_rejects_string_column():
    """Regression guard for issue #17: the Preprocessor path raises a typed pretab

    error instead of a bare numpy ValueError with no pointer to the cause.
    """
    df = pd.DataFrame({"c": ["a", "b", "c"] * 30})
    with pytest.raises(PretabDataError, match="already ordinal-encoded"):
        Preprocessor(categorical_method="onehot_from_ordinal").fit(df, None)
