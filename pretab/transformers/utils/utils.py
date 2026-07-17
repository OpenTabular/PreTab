"""Backward-compatible shim.

``center_identification_using_decision_tree`` now lives in
:mod:`pretab.core.centers`; it is re-exported here so existing
``from pretab.transformers.utils.utils import ...`` imports keep working.
"""

from ...core.centers import center_identification_using_decision_tree

__all__ = ["center_identification_using_decision_tree"]
