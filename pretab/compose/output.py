"""Format the fitted ColumnTransformer output into the public return shapes.

The Preprocessor returns either a single stacked NumPy array or a dictionary that
keeps each feature's transformed block separate (and any external embedding
blocks alongside them). This module owns that formatting only -- it performs no
fitting and holds no capability logic; the per-block slices it consumes are
computed in :mod:`pretab.compose.inspection`.
"""

import numpy as np

from ..exceptions import IncompatibleParamsError

__all__ = ["attach_embeddings", "build_output_dict", "format_output"]

_EMBEDDINGS_NOT_EXPECTED = (
    "Embeddings were not expected, but were provided.\n"
    "Fix: configure an embedding feature in feature_preprocessing before "
    "passing embeddings to transform, or omit the embeddings argument."
)


def build_output_dict(transformed, slices) -> dict:
    """Split a stacked array into a name -> block dict using ``slices``.

    ``slices`` is an ordered iterable of ``(name, start, width)`` describing each
    transformer's contiguous span in the stacked output.
    """
    return {name: transformed[:, start : start + width] for name, start, width in slices}


def attach_embeddings(result: dict, embeddings, *, expected: bool) -> dict:
    """Attach external embedding blocks to a transformed-output dict.

    Raises
    ------
    IncompatibleParamsError
        If ``embeddings`` are provided but none were configured at fit time.
    """
    if not expected:
        raise IncompatibleParamsError(_EMBEDDINGS_NOT_EXPECTED)
    if isinstance(embeddings, np.ndarray):
        result["embedding_1"] = embeddings.astype(np.float32)
    elif isinstance(embeddings, list):
        for idx, e in enumerate(embeddings):
            result[f"embedding_{idx + 1}"] = e.astype(np.float32)
    return result


def format_output(transformed, *, return_array, slices=None, embeddings=None, embeddings_expected=False):
    """Return the transformed data as a stacked array or a per-block dict.

    Parameters
    ----------
    transformed : numpy.ndarray
        The stacked array produced by the fitted ColumnTransformer.
    return_array : bool
        If True, return ``transformed`` unchanged; otherwise build the dict.
    slices : iterable of (str, int, int), optional
        Ordered ``(name, start, width)`` spans; required when ``return_array`` is
        False.
    embeddings : numpy.ndarray or list of numpy.ndarray, optional
        External embedding blocks to attach to the dict output.
    embeddings_expected : bool, default=False
        Whether embedding blocks were configured at fit time.
    """
    if return_array:
        return transformed

    result = build_output_dict(transformed, slices or [])
    if embeddings is not None:
        attach_embeddings(result, embeddings, expected=embeddings_expected)
    return result
