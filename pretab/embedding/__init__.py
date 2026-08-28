"""Embedding representations.

Maps a categorical or text column to a dense vector produced by a pretrained
model, as opposed to :mod:`pretab.encoding`, which recodes categories into small
discrete representations (codes or indicators).

Every class here is also re-exported from :mod:`pretab.transformers`.
"""

from .language import LanguageEmbeddingTransformer

__all__ = [
    "LanguageEmbeddingTransformer",
]
