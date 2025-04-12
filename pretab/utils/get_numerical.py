from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    PolynomialFeatures,
    RobustScaler,
    PowerTransformer,
)
from sklearn.impute import SimpleImputer
from ..transformers.splines.cubic import CubicSplineTransformer
from ..transformers.splines.thinplate_spline import ThinPlateSplineTransformer
from ..transformers.splines.tensor_product import TensorProductSplineTransformer
from ..transformers.splines.natural_cubic import NaturalCubicSplineTransformer
from ..transformers.splines.p_spline import PSplineTransformer
from ..transformers.feature_maps.rbf import RBFExpansionTransformer
from ..transformers.feature_maps.relu import ReLUExpansionTransformer
from ..transformers.feature_maps.sigmoid import SigmoidExpansionTransformer
from ..transformers.feature_maps.tanh import TanhExpansionTransformer
from ..transformers.binning.binning import CustomBinTransformer
from ..transformers.ple.ple import PLETransformer

from ..transformers.utils.floats import NoTransformer


def filter_kwargs(transformer_cls, kwargs, allowed=None):
    if allowed is not None:
        return {k: kwargs[k] for k in allowed if k in kwargs}
    return kwargs


def get_numerical_transformer_steps(
    method: str,
    add_imputer: bool = True,
    imputer_strategy: str = "mean",
    imputer_kwargs: dict = None,
    scaling: str = None,
    **kwargs,
):
    method = method.lower()
    steps = []

    if add_imputer:
        imputer_kwargs = imputer_kwargs or {}
        steps.append(
            ("imputer", SimpleImputer(strategy=imputer_strategy, **imputer_kwargs))
        )

    if scaling == "standardization":
        steps.append(("scaler", StandardScaler()))
    elif scaling == "minmax":
        steps.append(("minmax", MinMaxScaler(feature_range=(-1, 1))))

    method_map = {
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
        "none": (NoTransformer, []),
    }

    if method not in method_map:
        raise ValueError(f"Unknown numerical transformer method: {method}")

    cls, allowed_args = method_map[method]
    filtered = filter_kwargs(cls, kwargs, allowed=allowed_args)

    if method == "box-cox":
        steps.append(("scale_positive", MinMaxScaler(feature_range=(1e-3, 1))))
        steps.append(("boxcox", cls(method="box-cox", **filtered)))
    elif method == "yeo-johnson":
        steps.append(("yeojohnson", cls(method="yeo-johnson", **filtered)))
    else:
        name = method if method != "none" else "noop"
        steps.append((name, cls(**filtered)))

    return steps
