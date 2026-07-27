"""External embedding arrays must be validated against what ``fit`` recorded.

``Preprocessor.fit`` populates ``embedding_dimensions_`` but nothing used to read
it back, so ``transform`` accepted any array -- including one with the wrong row
count, which produced a result dict whose blocks had different heights.
"""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from pretab.core.exceptions import IncompatibleParamsError, PretabDataError
from pretab.preprocessor import Preprocessor


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=100), "c": rng.choice(list("xyz"), 100)})
    return frame, rng.normal(size=100), rng


def _fitted(frame, y, embeddings):
    return Preprocessor(numerical_method="minmax").fit(frame, y, embeddings=embeddings)


def test_matching_embeddings_pass_through(data):
    frame, y, rng = data
    pre = _fitted(frame, y, rng.random((100, 8)))

    out = cast("dict[str, np.ndarray]", pre.transform(frame, embeddings=rng.random((100, 8))))

    assert out["embedding_1"].shape == (100, 8)
    assert out["embedding_1"].dtype == np.float32


def test_fit_records_the_dimensions(data):
    frame, y, rng = data
    assert _fitted(frame, y, rng.random((100, 8))).embedding_dimensions_ == {"embedding_1": 8}


def test_wrong_width_is_rejected(data):
    frame, y, rng = data
    pre = _fitted(frame, y, rng.random((100, 8)))

    with pytest.raises(PretabDataError, match="has 3 columns, but 8 were seen during fit"):
        pre.transform(frame, embeddings=rng.random((100, 3)))


def test_wrong_row_count_is_rejected(data):
    frame, y, rng = data
    pre = _fitted(frame, y, rng.random((100, 8)))

    with pytest.raises(PretabDataError, match="has 7 rows, but X has 100"):
        pre.transform(frame, embeddings=rng.random((7, 8)))


def test_one_dimensional_embedding_is_rejected(data):
    frame, y, rng = data
    pre = _fitted(frame, y, rng.random((100, 8)))

    with pytest.raises(PretabDataError, match="must be a 2D array"):
        pre.transform(frame, embeddings=rng.random(100))


def test_wrong_number_of_arrays_is_rejected(data):
    frame, y, rng = data
    pre = _fitted(frame, y, [rng.random((100, 4)), rng.random((100, 5))])

    with pytest.raises(PretabDataError, match="Expected 2 embedding array"):
        pre.transform(frame, embeddings=[rng.random((100, 4))])


def test_embedding_list_round_trips(data):
    frame, y, rng = data
    pre = _fitted(frame, y, [rng.random((100, 4)), rng.random((100, 5))])

    out = cast(
        "dict[str, np.ndarray]",
        pre.transform(frame, embeddings=[rng.random((100, 4)), rng.random((100, 5))]),
    )

    assert out["embedding_1"].shape == (100, 4)
    assert out["embedding_2"].shape == (100, 5)


def test_unexpected_embeddings_still_rejected(data):
    frame, y, rng = data
    pre = Preprocessor(numerical_method="minmax").fit(frame, y)

    with pytest.raises(IncompatibleParamsError):
        pre.transform(frame, embeddings=rng.random((100, 8)))
