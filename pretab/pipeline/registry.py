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

__all__ = [
    "CATEGORICAL_ALIASES",
    "CATEGORICAL_METHODS",
    "NUMERICAL_ALIASES",
    "NUMERICAL_METHODS",
    "resolve_method",
]


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
    "ple": (
        PLETransformer,
        ["output_dim", "task", "adaptive", "min_output_dim", "max_output_dim", "random_state", "handle_missing"],
    ),
    "custombin": (CustomBinTransformer, ["output_dim"]),
    "rbf": (
        RBFExpansionTransformer,
        [
            "output_dim",
            "gamma",
            "task",
            "adaptive",
            "min_output_dim",
            "max_output_dim",
            "random_state",
        ],
    ),
    "relu": (
        ReLUExpansionTransformer,
        [
            "output_dim",
            "task",
            "adaptive",
            "min_output_dim",
            "max_output_dim",
            "random_state",
        ],
    ),
    "sigmoid": (
        SigmoidExpansionTransformer,
        [
            "output_dim",
            "task",
            "adaptive",
            "min_output_dim",
            "max_output_dim",
            "random_state",
        ],
    ),
    "tanh": (
        TanhExpansionTransformer,
        [
            "output_dim",
            "scale",
            "task",
            "adaptive",
            "min_output_dim",
            "max_output_dim",
            "random_state",
        ],
    ),
    "cubicspline": (
        CubicSplineTransformer,
        [
            "output_dim",
            "degree",
            "include_bias",
            "task",
            "adaptive",
            "min_output_dim",
            "max_output_dim",
            "random_state",
        ],
    ),
    "naturalspline": (
        NaturalCubicSplineTransformer,
        ["output_dim", "include_bias", "task", "adaptive", "min_output_dim", "max_output_dim", "random_state"],
    ),
    # pspline / tensorspline are penalized (difference-penalty) splines that rely
    # on equally-spaced knots, so they are *not* target-aware: no ``task`` here.
    "pspline": (PSplineTransformer, ["output_dim", "degree", "diff_order"]),
    "tensorspline": (
        TensorProductSplineTransformer,
        ["output_dim", "degree", "diff_order"],
    ),
    "tprs": (ThinPlateSplineTransformer, ["output_dim"]),
    "bspline": (BSplineTransformer, ["degree", "task", "adaptive", "min_output_dim", "max_output_dim", "random_state"]),
    "mspline": (MSplineTransformer, ["degree", "task", "adaptive", "min_output_dim", "max_output_dim", "random_state"]),
    "ispline": (ISplineTransformer, ["degree", "task", "adaptive", "min_output_dim", "max_output_dim", "random_state"]),
    "none": (NoTransformer, []),
}


# Canonical categorical method names (numerical ones are the NUMERICAL_METHODS
# keys). Kept here so both pipeline sides resolve names through one module.
CATEGORICAL_METHODS = frozenset({"int", "one-hot", "onehot_from_ordinal", "pretrained", "custombin", "none"})


def _squash(name: str) -> str:
    """Collapse a method name for separator/case-insensitive comparison.

    Lowercases, trims surrounding whitespace, and drops the ``-``, ``_`` and
    space separators so ``"One-Hot"``, ``"one_hot"`` and ``"onehot"`` all map to
    the same key. Canonical names that only differ by a separator (``"box-cox"``
    vs ``"boxcox"``, ``"cubicspline"`` vs ``"cubic spline"``) therefore match
    without needing an explicit alias entry.
    """
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


# Genuine synonyms / abbreviations that are *not* just separator variants of a
# canonical name (those are handled by :func:`_squash`). Keys are already
# squashed; values are canonical numerical method names.
NUMERICAL_ALIASES = {
    "standard": "standardization",
    "standardize": "standardization",
    "standardscaler": "standardization",
    "std": "standardization",
    "zscore": "standardization",
    "minmaxscaler": "minmax",
    "quantiletransformer": "quantile",
    "poly": "polynomial",
    "robustscaler": "robust",
    "piecewiselinear": "ple",
    "bin": "custombin",
    "binning": "custombin",
    "cubic": "cubicspline",
    "natural": "naturalspline",
    "naturalcubic": "naturalspline",
    "tensor": "tensorspline",
    "tensorproduct": "tensorspline",
    "tensorproductspline": "tensorspline",
    "thinplate": "tprs",
    "thinplatespline": "tprs",
    "passthrough": "none",
    "identity": "none",
    "raw": "none",
}

# Genuine synonyms / abbreviations for the categorical methods (keys squashed).
CATEGORICAL_ALIASES = {
    "integer": "int",
    "ordinal": "int",
    "label": "int",
    "labelencoder": "int",
    "ordinalencoder": "int",
    "ohe": "one-hot",
    "dummy": "one-hot",
    "onehotencoder": "one-hot",
    "embedding": "pretrained",
    "embeddings": "pretrained",
    "language": "pretrained",
    "llm": "pretrained",
    "bin": "custombin",
    "binning": "custombin",
    "passthrough": "none",
    "identity": "none",
    "raw": "none",
}


def resolve_method(name, canonical, aliases):
    """Resolve a user-supplied method name to its canonical spelling.

    Matching is case-insensitive, ignores ``-`` / ``_`` / space separators, and
    honours the explicit ``aliases`` map of synonyms and abbreviations. An
    unrecognized name is returned lowercased and stripped so the caller's own
    "unrecognized method" error lists the canonical options.

    Parameters
    ----------
    name : str
        The method name the user supplied.
    canonical : set or dict
        The canonical method names (``NUMERICAL_METHODS`` keys or
        ``CATEGORICAL_METHODS``).
    aliases : dict
        Squashed-alias to canonical-name mapping for this side of the pipeline.
    """
    key = name.strip().lower()
    if key in canonical:
        return key

    squashed = _squash(name)
    for canon in canonical:
        if _squash(canon) == squashed:
            return canon
    if squashed in aliases:
        return aliases[squashed]
    return key
