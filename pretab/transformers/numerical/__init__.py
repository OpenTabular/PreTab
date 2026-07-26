"""Numerical single-column transformers: binning, piecewise-linear encoding (PLE),
periodic encoding, Fourier feature maps and kernel-approximation feature maps.
Modules are moved here during the 1.0.0 restructure (Phase 1) and renamed to their
intention-revealing public names in Phase 5.
"""

from .binning import NumericBinningTransformer
from .fourier import FourierFeatureTransformer
from .kernel_approx import NystroemFeaturesTransformer, RandomFourierFeaturesTransformer
from .periodic import PeriodicEncodingTransformer
from .piecewise import PLETransformer

__all__ = [
    "FourierFeatureTransformer",
    "NumericBinningTransformer",
    "NystroemFeaturesTransformer",
    "PLETransformer",
    "PeriodicEncodingTransformer",
    "RandomFourierFeaturesTransformer",
]
