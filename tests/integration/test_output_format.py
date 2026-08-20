"""Tests for first-class output format control (P8.4).

Covers ``output_format`` (dense/sparse/auto), ``dtype`` casting, the
``output_report_`` memory report, and ``set_output`` pandas / polars DataFrame
wrapping.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

from pretab import Preprocessor
from pretab.exceptions import OptionalDependencyError


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.random(30), "b": rng.random(30)})


@pytest.fixture
def y():
    rng = np.random.default_rng(1)
    return rng.random(30)


def _bspline(**kwargs):
    return Preprocessor(
        numerical_method="bspline",
        output_dim=8,
        target_aware=False,
        placement_strategy="quantile",
        **kwargs,
    )


# --- default (dense) behaviour -------------------------------------------------


def test_default_output_format_is_dense(frame, y):
    p = _bspline().fit(frame, y)
    arr = p.transform(frame, return_array=True)
    assert isinstance(arr, np.ndarray)
    assert p.output_report_["format"] == "dense"


def test_default_dict_blocks_are_dense(frame, y):
    p = _bspline().fit(frame, y)
    out = p.transform(frame)
    assert isinstance(out, dict)
    assert all(isinstance(v, np.ndarray) for v in out.values())


# --- sparse --------------------------------------------------------------------


def test_sparse_return_array_is_csr(frame, y):
    p = _bspline(output_format="sparse").fit(frame, y)
    arr = p.transform(frame, return_array=True)
    assert sp.issparse(arr)
    assert isinstance(arr, sp.csr_matrix)
    assert arr.format == "csr"
    dense = _bspline().fit(frame, y).transform(frame, return_array=True)
    assert isinstance(dense, np.ndarray)
    np.testing.assert_allclose(arr.toarray(), dense)


def test_sparse_dict_blocks_are_csr(frame, y):
    p = _bspline(output_format="sparse").fit(frame, y)
    out = p.transform(frame)
    assert isinstance(out, dict)
    assert all(sp.issparse(v) for v in out.values())


def test_sparse_report_saves_memory(frame, y):
    p = _bspline(output_format="sparse").fit(frame, y)
    p.transform(frame, return_array=True)
    report = p.output_report_
    assert report["format"] == "sparse"
    assert report["actual_bytes"] < report["dense_bytes"]
    assert report["memory_saved_bytes"] == report["dense_bytes"] - report["actual_bytes"]


def test_sparse_intermediate_is_never_densified(monkeypatch):
    """A sparse ColumnTransformer result must remain sparse through formatting."""
    cats = pd.DataFrame({"c": [f"value-{i}" for i in range(100)]})
    p = Preprocessor(categorical_method="one-hot", output_format="sparse").fit(cats)
    raw = p.column_transformer_.transform(cats)
    assert sp.issparse(raw)

    class NoDensifyCSR(sp.csr_matrix):
        def toarray(self, *args, **kwargs):
            raise AssertionError("sparse intermediate was densified")

    guarded = NoDensifyCSR(raw)
    monkeypatch.setattr(p.column_transformer_, "transform", lambda X: guarded)

    out = p.transform(cats, return_array=True)
    assert sp.issparse(out)
    assert out.shape == (100, 100)
    assert out.nnz == 100
    assert p.output_report_["density"] == pytest.approx(0.01)


# --- auto ----------------------------------------------------------------------


def test_auto_picks_sparse_for_low_density(frame, y):
    # One-hot output on a high-cardinality categorical is very sparse.
    cats = pd.DataFrame({"c": [f"v{i % 15}" for i in range(30)]})
    p = Preprocessor(categorical_method="one-hot", output_format="auto").fit(cats)
    p.transform(cats, return_array=True)
    assert p.output_report_["format"] == "sparse"


def test_auto_picks_dense_for_high_density(frame, y):
    p = _bspline(output_format="auto").fit(frame, y)
    p.transform(frame, return_array=True)
    assert p.output_report_["format"] == "dense"


# --- dtype ---------------------------------------------------------------------


def test_dtype_casts_output(frame, y):
    p = _bspline(dtype=np.float32).fit(frame, y)
    arr = p.transform(frame, return_array=True)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32


def test_dtype_none_keeps_float64(frame, y):
    p = _bspline().fit(frame, y)
    arr = p.transform(frame, return_array=True)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64


def test_dtype_with_sparse(frame, y):
    p = _bspline(dtype=np.float32, output_format="sparse").fit(frame, y)
    arr = p.transform(frame, return_array=True)
    assert sp.issparse(arr)
    assert isinstance(arr, sp.csr_matrix)
    assert arr.dtype == np.float32


# --- output_report_ ------------------------------------------------------------


def test_output_report_shape_and_keys(frame, y):
    p = _bspline().fit(frame, y)
    arr = p.transform(frame, return_array=True)
    assert isinstance(arr, np.ndarray)
    report = p.output_report_
    assert set(report) == {
        "format",
        "shape",
        "density",
        "dense_bytes",
        "actual_bytes",
        "memory_saved_bytes",
    }
    assert report["shape"] == arr.shape
    assert 0.0 <= report["density"] <= 1.0


# --- set_output ----------------------------------------------------------------


def test_set_output_pandas_returns_dataframe(frame, y):
    p = _bspline().fit(frame, y).set_output(transform="pandas")
    out = p.transform(frame)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(p.get_feature_names_out())
    assert out.shape == (len(frame), p.total_output_dim_)


def test_set_output_pandas_fit_transform(frame, y):
    p = _bspline().set_output(transform="pandas")
    out = p.fit_transform(frame, y)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[1] == p.total_output_dim_


def test_set_output_default_still_dict(frame, y):
    p = _bspline().fit(frame, y).set_output(transform="default")
    out = p.transform(frame)
    assert isinstance(out, dict)


def test_set_output_polars_without_polars_raises(frame, y):
    import importlib.util

    p = _bspline().fit(frame, y).set_output(transform="polars")
    if importlib.util.find_spec("polars") is None:
        with pytest.raises(OptionalDependencyError):
            p.transform(frame)
    else:
        out = p.transform(frame)
        assert out.shape == (len(frame), p.total_output_dim_)


# --- validation ----------------------------------------------------------------


def test_invalid_output_format_raises(frame, y):
    from pretab.exceptions import InvalidParamError

    with pytest.raises(InvalidParamError):
        _bspline(output_format="nope").fit(frame, y)
