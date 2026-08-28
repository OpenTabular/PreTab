"""Kernel approximation representations.

Each transformer builds an explicit, low-dimensional feature map whose inner
products approximate an implicit kernel, so a linear model downstream can behave
like a kernel method without materializing the full kernel matrix. This mirrors
:mod:`sklearn.kernel_approximation`, which PreTab wraps directly.

Every class here is also re-exported from :mod:`pretab.transformers`.
"""

from .nystroem import NystroemFeaturesTransformer
from .random_fourier import RandomFourierFeaturesTransformer

__all__ = [
    "NystroemFeaturesTransformer",
    "RandomFourierFeaturesTransformer",
]
