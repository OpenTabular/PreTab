"""Multivariate spline transformers (tensor-product and thin-plate). These operate
on the numeric block as a whole and are standalone/grouped (excluded from the
per-column ``Preprocessor(numerical_method=...)`` whitelist). Modules are moved
here during the 1.0.0 restructure (Phase 1).
"""

from .tensor_product import TensorProductSplineTransformer
from .thin_plate import ThinPlateSplineTransformer

__all__ = [
    "TensorProductSplineTransformer",
    "ThinPlateSplineTransformer",
]
