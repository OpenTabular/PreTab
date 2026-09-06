"""Categorical encoding: map categories to ordinal codes or one-hot indicators
from an already ordinal-encoded input.
"""

from .one_hot import OneHotFromOrdinalTransformer
from .ordinal import ContinuousOrdinalTransformer

__all__ = [
    "ContinuousOrdinalTransformer",
    "OneHotFromOrdinalTransformer",
]
