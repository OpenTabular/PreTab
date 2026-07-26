"""Categorical transformers: ordinal encoding, language embeddings and the
time-boxed legacy one-hot-from-ordinal encoder. Modules are moved here during the
1.0.0 restructure (Phase 1).
"""

from .language_embedding import LanguageEmbeddingTransformer
from .legacy import OneHotFromOrdinalTransformer
from .ordinal import ContinuousOrdinalTransformer

__all__ = [
    "ContinuousOrdinalTransformer",
    "LanguageEmbeddingTransformer",
    "OneHotFromOrdinalTransformer",
]
