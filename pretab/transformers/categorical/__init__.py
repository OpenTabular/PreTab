"""Categorical transformers: language embeddings. Ordinal encoding and the
time-boxed legacy one-hot-from-ordinal encoder live in
:mod:`pretab.encoding.categorical`.
"""

from .language_embedding import LanguageEmbeddingTransformer

__all__ = [
    "LanguageEmbeddingTransformer",
]
