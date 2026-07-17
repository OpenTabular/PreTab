"""Backward-compatible shim.

``ContinuousOrdinalTransformer`` now lives in
:mod:`pretab.transformers.encoders.continuous_ordinal`; it is re-exported here so
existing ``from pretab.transformers.utils.continuous_ordinal import ...`` imports
keep working.
"""

from ..encoders.continuous_ordinal import ContinuousOrdinalTransformer

__all__ = ["ContinuousOrdinalTransformer"]
