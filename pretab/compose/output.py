"""Format the fitted ColumnTransformer output into the public return shapes.

The Preprocessor returns either a single stacked NumPy array or a dictionary that
keeps each feature's transformed block separate (and any external embedding
blocks alongside them). This module owns that formatting only -- it performs no
fitting and holds no capability logic; the per-block slices it consumes are
computed in :mod:`pretab.compose.inspection`.

It also owns the dense/sparse decision (``output_format``), the dtype-independent
memory report (``output_report_``), and wrapping the stacked array into a pandas
or polars DataFrame for :meth:`~pretab.Preprocessor.set_output`.
"""

import numpy as np
from scipy import sparse as sp

from ..exceptions import IncompatibleParamsError, OptionalDependencyError, PretabDataError

__all__ = [
    "attach_embeddings",
    "build_output_dict",
    "compute_output_report",
    "format_output",
    "to_dataframe_output",
    "validate_embedding_request",
]

# Density at or below which ``output_format="auto"`` switches to a sparse matrix,
# matching scikit-learn's ColumnTransformer ``sparse_threshold`` convention.
_SPARSE_AUTO_THRESHOLD = 0.3

_EMBEDDINGS_NOT_EXPECTED = (
    "Embeddings were not expected, but were provided.\n"
    "Fix: configure an embedding feature in feature_preprocessing before "
    "passing embeddings to transform, or omit the embeddings argument."
)
_EMBEDDINGS_REQUIRED = (
    "External embeddings were supplied during fit and are required during transform.\n"
    "Fix: pass embeddings with the same number of blocks and dimensions used during fit."
)
_EMBEDDINGS_DICT_ONLY = (
    "External embeddings are supported only with dictionary output.\n"
    "Fix: call transform(..., return_array=False) with set_output(transform='default')."
)


def validate_embedding_request(embeddings, *, expected: bool, output_kind: str = "dict") -> None:
    """Validate embedding presence and the requested output container.

    External embeddings are separate named feature blocks, so they are available
    only through dictionary output. Once supplied during fit, they are required on
    every transform to keep the fitted and transformed feature contracts aligned.
    """
    if embeddings is None:
        if expected:
            raise PretabDataError(_EMBEDDINGS_REQUIRED)
        return
    if not expected:
        raise IncompatibleParamsError(_EMBEDDINGS_NOT_EXPECTED)
    if output_kind != "dict":
        raise IncompatibleParamsError(_EMBEDDINGS_DICT_ONLY)


def compute_output_report(array, output_format, *, threshold=_SPARSE_AUTO_THRESHOLD):
    """Resolve the concrete output format and build the memory report.

    Parameters
    ----------
    array : numpy.ndarray or scipy.sparse matrix
        The stacked output. Sparse inputs are inspected through their shape,
        dtype, and stored values without converting them to a dense array.
    output_format : {"auto", "dense", "sparse"}
        Requested format. ``"auto"`` picks ``"sparse"`` when the density is below
        ``threshold``.
    threshold : float, default=0.3
        Density cut-off for the ``"auto"`` decision.

    Returns
    -------
    tuple of (str, dict)
        The resolved format (``"dense"`` or ``"sparse"``) and a report dict with
        ``format``, ``shape``, ``density``, ``dense_bytes``, ``actual_bytes``, and
        ``memory_saved_bytes``.
    """
    size = int(np.prod(array.shape, dtype=np.int64))
    nonzero = int(array.count_nonzero()) if sp.issparse(array) else int(np.count_nonzero(array))
    density = float(nonzero) / size if size else 0.0
    dense_bytes = size * int(array.dtype.itemsize)

    if output_format == "sparse":
        use_sparse = True
    elif output_format == "auto":
        use_sparse = density < threshold
    else:  # "dense"
        use_sparse = False

    if use_sparse:
        csr = array.tocsr(copy=False) if sp.issparse(array) else sp.csr_matrix(array)
        actual_bytes = int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes)
        fmt = "sparse"
    else:
        actual_bytes = dense_bytes
        fmt = "dense"

    report = {
        "format": fmt,
        "shape": tuple(int(s) for s in array.shape),
        "density": density,
        "dense_bytes": dense_bytes,
        "actual_bytes": actual_bytes,
        "memory_saved_bytes": max(0, dense_bytes - actual_bytes),
    }
    return fmt, report


def to_dataframe_output(array, columns, container):
    """Wrap a dense stacked array in a pandas or polars DataFrame.

    Parameters
    ----------
    array : numpy.ndarray or scipy.sparse matrix
        Stacked output. Sparse input is intentionally densified because
        ``set_output`` requests a dense pandas or polars container.
    columns : sequence of str
        One name per output column (from ``get_feature_names_out``).
    container : {"pandas", "polars"}
        Target dataframe library.

    Raises
    ------
    OptionalDependencyError
        If ``container="polars"`` but polars is not installed.
    """
    columns = list(columns)
    if sp.issparse(array):
        array = array.toarray()
    if container == "pandas":
        import pandas as pd

        return pd.DataFrame(array, columns=pd.Index(columns))
    try:
        import polars as pl  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without polars
        raise OptionalDependencyError(
            "set_output(transform='polars') requires the optional 'polars' package. "
            "Install it with `pip install polars`."
        ) from exc
    return pl.from_numpy(array, schema=columns)


def build_output_dict(transformed, slices, *, as_sparse=False) -> dict:
    """Split a stacked dense or sparse array into a name -> block dict.

    ``slices`` is an ordered iterable of ``(name, start, width)`` describing each
    transformer's contiguous span in the stacked output. When ``as_sparse`` is
    True each block is returned as a SciPy CSR matrix.
    """
    result = {}
    for name, start, width in slices:
        block = transformed[:, start : start + width]
        if as_sparse:
            result[name] = block.tocsr(copy=False) if sp.issparse(block) else sp.csr_matrix(block)
        else:
            result[name] = block.toarray() if sp.issparse(block) else block
    return result


def attach_embeddings(result: dict, embeddings, *, expected: bool, embedding_dimensions=None, n_samples=None) -> dict:
    """Attach external embedding blocks to a transformed-output dict.

    Validates the arrays against what ``fit`` recorded in ``embedding_dimensions``
    (a ``name -> width`` mapping): the number of arrays, that each is 2D, that
    each array's width matches its fitted dimension, and that each array's row
    count matches ``n_samples``. Both are optional and skip the corresponding
    check when omitted (``None``), so callers without fit-time metadata keep
    working.

    Raises
    ------
    IncompatibleParamsError
        If ``embeddings`` are provided but none were configured at fit time.
    PretabDataError
        If the number of arrays, an array's shape, or its row count does not
        match what ``fit`` recorded.
    """
    validate_embedding_request(embeddings, expected=expected)
    arrays = [embeddings] if isinstance(embeddings, np.ndarray) else list(embeddings)
    if embedding_dimensions is not None and len(arrays) != len(embedding_dimensions):
        raise PretabDataError(
            f"Expected {len(embedding_dimensions)} embedding array(s) (as fitted) but got {len(arrays)}.\n"
            "Fix: pass the same number of embedding arrays that were passed to fit."
        )
    for idx, arr in enumerate(arrays):
        name = f"embedding_{idx + 1}"
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise PretabDataError(
                f"{name} must be 2D (n_samples, n_dims); got shape {arr.shape}.\n"
                "Fix: reshape the embedding array to 2 dimensions."
            )
        expected_width = embedding_dimensions.get(name) if embedding_dimensions is not None else None
        if expected_width is not None and arr.shape[1] != expected_width:
            raise PretabDataError(
                f"{name} has {arr.shape[1]} column(s) but {expected_width} were fitted.\n"
                "Fix: pass an embedding array with the same width used at fit time."
            )
        if n_samples is not None and arr.shape[0] != n_samples:
            raise PretabDataError(
                f"{name} has {arr.shape[0]} row(s) but X has {n_samples}.\n"
                "Fix: pass an embedding array with one row per sample in X."
            )
        result[name] = arr.astype(np.float32)
    return result


def format_output(
    transformed,
    *,
    return_array,
    slices=None,
    embeddings=None,
    embeddings_expected=False,
    embedding_dimensions=None,
    output_format="dense",
):
    """Return the transformed data as a stacked array or a per-block dict.

    Parameters
    ----------
    transformed : numpy.ndarray or scipy.sparse matrix
        The stacked output produced by the fitted ColumnTransformer.
    return_array : bool
        If True, return the stacked array (dense or CSR); otherwise build the dict.
    slices : iterable of (str, int, int), optional
        Ordered ``(name, start, width)`` spans; required when ``return_array`` is
        False.
    embeddings : numpy.ndarray or list of numpy.ndarray, optional
        External embedding blocks to attach to the dict output.
    embeddings_expected : bool, default=False
        Whether embedding blocks were configured at fit time.
    embedding_dimensions : dict, optional
        ``name -> width`` mapping recorded at fit time, used to validate the
        arrays passed here.
    output_format : {"dense", "sparse"}, default="dense"
        Resolved output format. ``"sparse"`` returns a CSR matrix (array path) or
        CSR blocks (dict path).
    """
    validate_embedding_request(
        embeddings,
        expected=embeddings_expected,
        output_kind="array" if return_array else "dict",
    )

    as_sparse = output_format == "sparse"
    if as_sparse:
        transformed = transformed.tocsr(copy=False) if sp.issparse(transformed) else sp.csr_matrix(transformed)
    elif sp.issparse(transformed):
        transformed = transformed.toarray()

    if return_array:
        return transformed

    result = build_output_dict(transformed, slices or [], as_sparse=as_sparse)
    if embeddings is not None:
        attach_embeddings(
            result,
            embeddings,
            expected=embeddings_expected,
            embedding_dimensions=embedding_dimensions,
            n_samples=transformed.shape[0],
        )
    return result
