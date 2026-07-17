"""Declarative registry of numerical preprocessing strategies.

Each entry maps a strategy name to a ``(transformer_cls, allowed_args)`` pair:
the class to instantiate and the constructor arguments it accepts (used to
filter the shared ``**kwargs``). Adding a new numerical strategy is therefore a
single-line edit here rather than a change to the assembly logic.

A few names (``box-cox`` / ``yeo-johnson`` share ``PowerTransformer``, and the
B/M/I splines need extra knot wiring) require special handling in
:mod:`pretab.pipeline.numerical`; this table still records the class and its
allowed arguments for them.
"""

from sklearn.preprocessing import (
    MinMaxScaler,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from ..transformers.binning.binning import CustomBinTransformer
from ..transformers.encoders.floats import NoTransformer
from ..transformers.feature_maps.rbf import RBFExpansionTransformer
from ..transformers.feature_maps.relu import ReLUExpansionTransformer
from ..transformers.feature_maps.sigmoid import SigmoidExpansionTransformer
from ..transformers.feature_maps.tanh import TanhExpansionTransformer
from ..transformers.ple.ple import PLETransformer
from ..transformers.splines.bspline import BSplineTransformer
from ..transformers.splines.cubic import CubicSplineTransformer
from ..transformers.splines.integrated_spline import ISplineTransformer
from ..transformers.splines.mspline import MSplineTransformer
from ..transformers.splines.natural_cubic import NaturalCubicSplineTransformer
from ..transformers.splines.pspline import PSplineTransformer
from ..transformers.splines.tensor_product import TensorProductSplineTransformer
from ..transformers.splines.thinplate_spline import ThinPlateSplineTransformer

__all__ = ["NUMERICAL_METHODS"]


# name -> (transformer class, constructor arguments it accepts)
NUMERICAL_METHODS = {
    "standardization": (StandardScaler, []),
    "minmax": (MinMaxScaler, []),
    "quantile": (
        QuantileTransformer,
        ["n_quantiles", "output_distribution", "random_state"],
    ),
    "polynomial": (
        PolynomialFeatures,
        ["degree", "interaction_only", "include_bias"],
    ),
    "robust": (RobustScaler, []),
    "box-cox": (PowerTransformer, []),
    "yeo-johnson": (PowerTransformer, []),
    "ple": (PLETransformer, ["n_bins", "task"]),
    "custombin": (CustomBinTransformer, ["bins"]),
    "rbf": (
        RBFExpansionTransformer,
        ["n_centers", "gamma", "use_decision_tree", "task", "strategy"],
    ),
    "relu": (
        ReLUExpansionTransformer,
        ["n_centers", "use_decision_tree", "task", "strategy"],
    ),
    "sigmoid": (
        SigmoidExpansionTransformer,
        ["n_centers", "use_decision_tree", "task", "strategy"],
    ),
    "tanh": (
        TanhExpansionTransformer,
        ["n_centers", "scale", "use_decision_tree", "task", "strategy"],
    ),
    "cubicspline": (CubicSplineTransformer, ["n_knots", "degree", "include_bias"]),
    "naturalspline": (NaturalCubicSplineTransformer, ["n_knots", "include_bias"]),
    "pspline": (PSplineTransformer, ["n_knots", "degree", "diff_order"]),
    "tensorspline": (
        TensorProductSplineTransformer,
        ["n_knots", "degree", "diff_order"],
    ),
    "tprs": (ThinPlateSplineTransformer, ["n_basis"]),
    "bspline": (BSplineTransformer, ["degree", "task"]),
    "mspline": (MSplineTransformer, ["degree", "task"]),
    "ispline": (ISplineTransformer, ["degree", "task"]),
    "none": (NoTransformer, []),
}
