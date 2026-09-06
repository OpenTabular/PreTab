"""Supporting data-preparation utilities.

These transformers don't expand or recode a feature; they prepare it for the rest
of the pipeline, converting types or flagging missingness before it reaches the
transformer that actually does the work. Distinct from
:mod:`pretab.preprocessor`, which holds the top-level :class:`~pretab.preprocessor.Preprocessor`
facade.

Every class here is also re-exported from :mod:`pretab.transformers`.
"""

from .floats import NoTransformer, ToFloatTransformer
from .missing import MissingStateIndicator

__all__ = [
    "MissingStateIndicator",
    "NoTransformer",
    "ToFloatTransformer",
]
