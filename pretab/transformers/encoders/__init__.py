"""Categorical / numeric encoders for tabular preprocessing.

These transformers turn raw column values into numeric arrays that downstream
models can consume: ordinal integer encoding, a float cast, and a pass-through.
"""

from .continuous_ordinal import ContinuousOrdinalTransformer
from .floats import NoTransformer, ToFloatTransformer

__all__ = [
    "ContinuousOrdinalTransformer",
    "NoTransformer",
    "ToFloatTransformer",
]
