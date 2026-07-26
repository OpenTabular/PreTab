"""Backward-compatible shim.

``get_numerical_transformer_steps`` moved to
:mod:`pretab.pipeline.numerical`; it is re-exported here so existing
``from pretab.utils.get_numerical import ...`` imports keep working.
"""

from ..pipeline.numerical import get_numerical_transformer_steps

__all__ = ["get_numerical_transformer_steps"]
