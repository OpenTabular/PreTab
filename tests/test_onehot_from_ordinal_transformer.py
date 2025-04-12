import numpy as np
import pytest
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
