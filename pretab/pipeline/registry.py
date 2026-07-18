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
    "ple": (PLETransformer, ["output_dim", "task", "adaptive", "min_output_dim", "max_output_dim"]),
    "custombin": (CustomBinTransformer, ["output_dim"]),
    "rbf": (
        RBFExpansionTransformer,
        [
            "output_dim",
            "gamma",
            "use_decision_tree",
            "task",
            "strategy",
            "adaptive",
            "min_output_dim",
            "max_output_dim",
        ],
    ),
    "relu": (
        ReLUExpansionTransformer,
        [
            "output_dim",
            "use_decision_tree",
            "task",
            "strategy",
            "adaptive",
            "min_output_dim",
            "max_output_dim",
        ],
    ),
    "sigmoid": (
        SigmoidExpansionTransformer,
        [
            "output_dim",
            "use_decision_tree",
            "task",
            "strategy",
            "adaptive",
            "min_output_dim",
            "max_output_dim",
        ],
    ),
    "tanh": (
        TanhExpansionTransformer,
        [
            "output_dim",
            "scale",
            "use_decision_tree",
            "task",
            "strategy",
            "adaptive",
            "min_output_dim",
            "max_output_dim",
        ],
    ),
    "cubicspline": (CubicSplineTransformer, ["output_dim", "degree", "include_bias"]),
    "naturalspline": (NaturalCubicSplineTransformer, ["output_dim", "include_bias"]),
    "pspline": (PSplineTransformer, ["output_dim", "degree", "diff_order"]),
    "tensorspline": (
        TensorProductSplineTransformer,
        ["output_dim", "degree", "diff_order"],
    ),
    "tprs": (ThinPlateSplineTransformer, ["output_dim"]),
    "bspline": (BSplineTransformer, ["degree", "task"]),
    "mspline": (MSplineTransformer, ["degree", "task"]),
    "ispline": (ISplineTransformer, ["degree", "task"]),
    "none": (NoTransformer, []),
}
