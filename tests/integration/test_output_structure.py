"""``output_structure`` controls the top-level shape ``transform`` /
``fit_transform`` return when ``return_array`` is not passed explicitly.

Regression coverage for the original bug: a plain ``Preprocessor`` inside a bare
``sklearn.pipeline.Pipeline`` broke the next step, because ``transform()``
defaulted to a dict and ``Pipeline`` has no way to request ``return_array=True``
for an intermediate step.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from pretab import Preprocessor
from pretab.exceptions import InvalidParamError


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})


@pytest.fixture
def y():
    return np.random.default_rng(1).normal(size=40)


def test_default_output_structure_is_matrix(frame, y):
    pre = Preprocessor().fit(frame, y)
    out = pre.transform(frame)
    assert isinstance(out, np.ndarray)
    assert out.shape == (len(frame), pre.total_output_dim_)


def test_output_structure_blocks_returns_dict(frame, y):
    pre = Preprocessor(output_structure="blocks").fit(frame, y)
    out = pre.transform(frame)
    assert isinstance(out, dict)
    assert all(isinstance(v, np.ndarray) for v in out.values())


def test_explicit_return_array_overrides_output_structure(frame, y):
    # output_structure="matrix" but return_array=False explicitly -> dict.
    matrix_pre = Preprocessor(output_structure="matrix").fit(frame, y)
    assert isinstance(matrix_pre.transform(frame, return_array=False), dict)

    # output_structure="blocks" but return_array=True explicitly -> array.
    blocks_pre = Preprocessor(output_structure="blocks").fit(frame, y)
    assert isinstance(blocks_pre.transform(frame, return_array=True), np.ndarray)


def test_invalid_output_structure_raises(frame, y):
    with pytest.raises(InvalidParamError, match="output_structure"):
        Preprocessor(output_structure="nope").fit(frame, y)


def test_preprocessor_composes_in_a_plain_pipeline(frame, y):
    # Regression test: Pipeline always calls transform(X) with no return_array
    # kwarg, so a dict-by-default Preprocessor broke the next step with
    # "TypeError: float() argument must be a string or a real number, not 'dict'".
    model = Pipeline([("pretab", Preprocessor()), ("model", Ridge())])
    model.fit(frame, y)
    preds = model.predict(frame)
    assert preds.shape == (len(frame),)
