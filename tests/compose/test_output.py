"""Unit tests for :mod:`pretab.compose.output`."""

import numpy as np
import pytest

from pretab.compose.output import attach_embeddings, build_output_dict, format_output
from pretab.exceptions import IncompatibleParamsError, PretabDataError


def test_build_output_dict_slices_by_span():
    arr = np.arange(12).reshape(3, 4)
    out = build_output_dict(arr, [("a", 0, 1), ("b", 1, 3)])
    assert set(out) == {"a", "b"}
    assert out["a"].shape == (3, 1)
    np.testing.assert_array_equal(out["b"], arr[:, 1:4])


def test_attach_embeddings_array_casts_to_float32():
    result = {}
    attach_embeddings(result, np.ones((2, 3)), expected=True)
    assert result["embedding_1"].dtype == np.float32
    assert result["embedding_1"].shape == (2, 3)


def test_attach_embeddings_list_numbers_blocks():
    result = {}
    attach_embeddings(result, [np.ones((2, 2)), np.ones((2, 1))], expected=True)
    assert set(result) == {"embedding_1", "embedding_2"}


def test_attach_embeddings_unexpected_raises():
    with pytest.raises(IncompatibleParamsError):
        attach_embeddings({}, np.ones((2, 3)), expected=False)


def test_attach_embeddings_rejects_wrong_count():
    """Regression guard for issue #34: a mismatched number of arrays must raise."""
    with pytest.raises(PretabDataError, match="Expected 2"):
        attach_embeddings({}, np.ones((2, 3)), expected=True, embedding_dimensions={"embedding_1": 3, "embedding_2": 4})


def test_attach_embeddings_rejects_wrong_width():
    """Regression guard for issue #34: a mismatched embedding width must raise."""
    with pytest.raises(PretabDataError, match="has 3 column"):
        attach_embeddings({}, np.ones((2, 3)), expected=True, embedding_dimensions={"embedding_1": 8})


def test_attach_embeddings_rejects_wrong_row_count():
    """Regression guard for issue #34: a mismatched row count must raise."""
    with pytest.raises(PretabDataError, match="has 2 row"):
        attach_embeddings(
            {}, np.ones((2, 3)), expected=True, embedding_dimensions={"embedding_1": 3}, n_samples=100
        )


def test_format_output_array_returns_input_unchanged():
    arr = np.zeros((2, 2))
    assert format_output(arr, return_array=True) is arr


def test_format_output_dict_builds_blocks():
    arr = np.arange(6).reshape(2, 3)
    out = format_output(arr, return_array=False, slices=[("x", 0, 3)])
    assert isinstance(out, dict)
    assert set(out) == {"x"}
    np.testing.assert_array_equal(out["x"], arr)


def test_format_output_dict_attaches_embeddings():
    arr = np.arange(6).reshape(2, 3)
    out = format_output(
        arr,
        return_array=False,
        slices=[("x", 0, 3)],
        embeddings=np.ones((2, 4)),
        embeddings_expected=True,
    )
    assert "embedding_1" in out
