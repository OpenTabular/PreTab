import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from pretab.pipeline import get_categorical_transformer_steps


def _build(method, **kwargs):
    return Pipeline(get_categorical_transformer_steps(method, add_imputer=False, **kwargs))


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

