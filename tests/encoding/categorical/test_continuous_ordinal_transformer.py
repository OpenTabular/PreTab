import numpy as np
import pandas as pd

from pretab.transformers import ContinuousOrdinalTransformer


def test_continuous_ordinal_fit_transform_handles_nan():
    # Regression test: np.unique() on an object column crashes when it mixes
    # strings with NaN/None, despite the transformer declaring allow_nan=True.
    X = np.array([["a"], ["b"], [np.nan], ["a"]], dtype=object)
    transformer = ContinuousOrdinalTransformer()
    Xt = transformer.fit_transform(X)

    assert Xt.ravel().tolist() == [1, 2, 0, 1]


def test_continuous_ordinal_fit_transform_handles_none():
    X = np.array([["a"], ["b"], [None], ["a"]], dtype=object)
    transformer = ContinuousOrdinalTransformer()
    Xt = transformer.fit_transform(X)

    assert Xt.ravel().tolist() == [1, 2, 0, 1]


def test_continuous_ordinal_handles_pandas_categorical_export():
    # A realistic pandas categorical export: object dtype, mixed strings and a
    # missing marker (NaN), round-tripping through fit/transform without error.
    df = pd.DataFrame({"col": pd.Categorical(["x", "y", None, "x", "z"])})
    X = df["col"].to_numpy(dtype=object).reshape(-1, 1)

    transformer = ContinuousOrdinalTransformer()
    Xt = transformer.fit(X).transform(X)

    assert Xt.shape == (5, 1)
    assert Xt[2, 0] == 0  # the missing row maps to the reserved code


def test_continuous_ordinal_allow_nan_tag_is_exercised():
    tags = ContinuousOrdinalTransformer().__sklearn_tags__()
    assert tags.input_tags.allow_nan is True

    # Not just declared: actually fitting/transforming NaN must not raise.
    X = np.array([["a"], [np.nan], ["b"]], dtype=object)
    ContinuousOrdinalTransformer().fit_transform(X)


def test_continuous_ordinal_unseen_category_maps_to_zero():
    transformer = ContinuousOrdinalTransformer().fit(np.array([["a"], ["b"]], dtype=object))
    Xt = transformer.transform(np.array([["c"]], dtype=object))
    assert Xt.ravel().tolist() == [0]
