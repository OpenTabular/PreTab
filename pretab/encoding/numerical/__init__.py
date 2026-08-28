"""Numerical encoding: recode numeric values into bins, target-aware piecewise
linear encodings, or cyclic (sin/cos) representations.
"""

from .binning import NumericBinningTransformer
from .periodic import PeriodicEncodingTransformer
from .ple import PLETransformer

__all__ = [
    "NumericBinningTransformer",
    "PLETransformer",
    "PeriodicEncodingTransformer",
]
