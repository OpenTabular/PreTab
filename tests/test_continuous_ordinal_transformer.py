"""``ContinuousOrdinalTransformer`` must treat a DataFrame like the equivalent array.

``fit`` iterated ``X.T``, which for a DataFrame yields the *column labels of the
transpose* -- i.e. the original row index -- and ``transform`` iterated ``X``,
which yields column names. Frames therefore produced one mapping per row and an
all-zero result of the wrong shape, with no error raised.
"""

import numpy as np
import pandas as pd
import pytest

from pretab.core.exceptions import PretabDataError
from pretab.transformers import ContinuousOrdinalTransformer


@pytest.fixture
def frame():
    return pd.DataFrame({"c1": ["a", "b", "a", "c"], "c2": ["x", "y", "x", "y"]})


def test_fit_maps_one_dict_per_column(frame):
    transformer = ContinuousOrdinalTransformer().fit(frame)

    assert len(transformer.mapping_) == 2
    assert transformer.n_features_in_ == 2
    assert set(transformer.mapping_[0]) == {"a", "b", "c", None}
    assert set(transformer.mapping_[1]) == {"x", "y", None}


def test_transform_of_dataframe_keeps_shape_and_codes(frame):
    out = ContinuousOrdinalTransformer().fit(frame).transform(frame)

    assert out.shape == (4, 2)
    np.testing.assert_array_equal(out[:, 0], [1, 2, 1, 3])
    np.testing.assert_array_equal(out[:, 1], [1, 2, 1, 2])


def test_dataframe_and_ndarray_agree(frame):
    array = frame.to_numpy(dtype=object)

    from_frame = ContinuousOrdinalTransformer().fit(frame).transform(frame)
    from_array = ContinuousOrdinalTransformer().fit(array).transform(array)

    np.testing.assert_array_equal(from_frame, from_array)


def test_list_input_is_accepted():
    rows = [["a", "x"], ["b", "y"], ["a", "x"]]
    out = ContinuousOrdinalTransformer().fit(rows).transform(rows)

    assert out.shape == (3, 2)


def test_empty_transform_keeps_two_dimensions():
    X = np.array([["a", "x"], ["b", "y"]], dtype=object)
    transformer = ContinuousOrdinalTransformer().fit(X)

    assert transformer.transform(np.empty((0, 2), dtype=object)).shape == (0, 2)


def test_transform_rejects_wrong_feature_count():
    X = np.array([["a", "x"], ["b", "y"]], dtype=object)
    transformer = ContinuousOrdinalTransformer().fit(X)

    with pytest.raises(PretabDataError, match="expecting 2 features"):
        transformer.transform(np.array([["a"], ["b"]], dtype=object))


def test_unknown_categories_still_map_to_zero():
    X = np.array([["a"], ["b"]], dtype=object)
    transformer = ContinuousOrdinalTransformer().fit(X)

    np.testing.assert_array_equal(
        transformer.transform(np.array([["a"], ["zzz"]], dtype=object)).ravel(), [1, 0]
    )
