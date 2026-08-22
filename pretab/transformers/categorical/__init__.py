"""Categorical transformers: ordinal encoding, language embeddings and the
time-boxed legacy one-hot-from-ordinal encoder.
"""

from .language_embedding import LanguageEmbeddingTransformer
from .legacy import OneHotFromOrdinalTransformer
from .ordinal import ContinuousOrdinalTransformer

__all__ = [
    "ContinuousOrdinalTransformer",
    "LanguageEmbeddingTransformer",
    "OneHotFromOrdinalTransformer",
]
