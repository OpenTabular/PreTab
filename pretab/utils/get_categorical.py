"""Backward-compatible shim.

``get_categorical_transformer_steps`` moved to
:mod:`pretab.pipeline.categorical`; it is re-exported here so existing
``from pretab.utils.get_categorical import ...`` imports keep working.
"""

from ..pipeline.categorical import get_categorical_transformer_steps

__all__ = ["get_categorical_transformer_steps"]
