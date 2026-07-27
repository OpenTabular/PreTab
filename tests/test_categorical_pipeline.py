from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from pretab import Preprocessor
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



# --------------------------------------------------------------------------- #
# ``onehot_from_ordinal`` must accept raw (string) categoricals.
#
# The pipeline appended only ``OneHotFromOrdinalTransformer``, which requires
# already-ordinal input, so ``np.max(X, axis=0).astype(int)`` died with a bare
# ``ValueError: invalid literal for int() with base 10: 'c'``.
# --------------------------------------------------------------------------- #
def test_onehot_from_ordinal_encodes_string_categories():
    frame = pd.DataFrame({"c": ["a", "b", "c"] * 30})
    pre = Preprocessor(categorical_method="onehot_from_ordinal", numerical_method="none")

    out = cast("np.ndarray", pre.fit_transform(frame, return_array=True))

    # 3 categories plus the reserved column 0 for unseen values.
    assert out.shape == (90, 4)
    np.testing.assert_array_equal(out[:3], np.eye(4)[[1, 2, 3]])


def test_onehot_from_ordinal_pipeline_encodes_before_one_hot():
    steps = [name for name, _ in get_categorical_transformer_steps("onehot_from_ordinal")]
    assert steps.index("continuous_ordinal") < steps.index("onehot_from_ordinal")


def test_onehot_from_ordinal_sends_unseen_categories_to_the_reserved_column():
    frame = pd.DataFrame({"c": ["a", "b", "c"] * 30})
    pre = Preprocessor(categorical_method="onehot_from_ordinal", numerical_method="none").fit(frame)

    out = cast("np.ndarray", pre.transform(pd.DataFrame({"c": ["a", "ZZZ", "c"]}), return_array=True))

    np.testing.assert_array_equal(out[1], [1.0, 0.0, 0.0, 0.0])


def test_onehot_from_ordinal_feature_names_match_width():
    frame = pd.DataFrame({"c": ["a", "b", "c"] * 30})
    pre = Preprocessor(categorical_method="onehot_from_ordinal", numerical_method="none").fit(frame)

    transformed = cast("np.ndarray", pre.transform(frame, return_array=True))
    assert len(pre.get_feature_names_out()) == transformed.shape[1]
