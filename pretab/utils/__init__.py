"""Backward-compatible shim.

The assembly layer moved to :mod:`pretab.pipeline`; the step factories are
re-exported here so existing ``from pretab.utils import ...`` imports keep
working.
"""

from ..pipeline import (
    get_categorical_transformer_steps,
    get_numerical_transformer_steps,
)

__all__ = [
    "get_categorical_transformer_steps",
    "get_numerical_transformer_steps",
]
