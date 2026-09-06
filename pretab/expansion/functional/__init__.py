"""Explicit nonlinear basis-function expansions.

Each transformer maps a numeric feature through a fixed nonlinear function (radial
basis, ReLU, sigmoid, tanh, or a sine/cosine pair) evaluated at a set of centers or
frequencies, producing one output column per basis unit. This is the "functional"
half of :mod:`pretab.expansion`, distinct from spline bases which live in
:mod:`pretab.expansion.spline`.

Every class here is also re-exported from :mod:`pretab.transformers`.
"""

from .base import BaseCenterExpansion
from .fourier import FourierFeatureTransformer
from .rbf import RBFExpansionTransformer
from .relu import ReLUExpansionTransformer
from .sigmoid import SigmoidExpansionTransformer
from .tanh import TanhExpansionTransformer

__all__ = [
    "BaseCenterExpansion",
    "FourierFeatureTransformer",
    "RBFExpansionTransformer",
    "ReLUExpansionTransformer",
    "SigmoidExpansionTransformer",
    "TanhExpansionTransformer",
]
