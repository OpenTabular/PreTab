import warnings

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..transformers.splines.knot_selectors import CARTKnotSelector
from .registry import NUMERICAL_METHODS

# Spline basis expansions that share the target-aware knot API.
SPLINE_EXPANSION_METHODS = ("bspline", "mspline", "ispline")

# Valid range for the number of spline basis functions per feature.
_MIN_SPLINE_BASIS = 5
_MAX_SPLINE_BASIS = 50


def filter_kwargs(transformer_cls, kwargs, allowed=None):
    if allowed is not None:
        return {k: kwargs[k] for k in allowed if k in kwargs}
    return kwargs


def _clamp_spline_basis(n_knots):
    """Clamp a requested basis-function count into the supported spline range.

    The Preprocessor shares a single ``n_knots`` setting across every numerical
    strategy (default 64), but the B/M/I spline transformers accept between
    ``5`` and ``50`` basis functions. Values outside that window are clamped so
    switching to a spline strategy keeps working with the shared default.
    """
    clamped = max(_MIN_SPLINE_BASIS, min(int(n_knots), _MAX_SPLINE_BASIS))
    if clamped != n_knots:
        warnings.warn(
            f"n_knots={n_knots} is outside the spline range "
            f"[{_MIN_SPLINE_BASIS}, {_MAX_SPLINE_BASIS}]; using {clamped} basis functions.",
            UserWarning,
            stacklevel=2,
        )
    return clamped


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
        steps.append(("imputer", SimpleImputer(strategy=imputer_strategy, **imputer_kwargs)))

    # Define scalers that could be added independently
    scalers = {
        "standardization": ("scaler", StandardScaler()),
        "minmax": ("minmax", MinMaxScaler(feature_range=(-1, 1))),
    }

    # Add optional scaling step only if not already part of method
    if scaling in scalers and scaling != method:
        steps.append(scalers[scaling])

    if method not in NUMERICAL_METHODS:
        raise ValueError(f"Unknown numerical transformer method: {method}")

    cls, allowed_args = NUMERICAL_METHODS[method]
    filtered = filter_kwargs(cls, kwargs, allowed=allowed_args)

    if method == "box-cox":
        steps.append(("scale_positive", MinMaxScaler(feature_range=(1e-3, 1))))
        steps.append(("boxcox", cls(method="box-cox", **filtered)))
    elif method == "yeo-johnson":
        steps.append(("yeojohnson", cls(method="yeo-johnson", **filtered)))
    elif method in SPLINE_EXPANSION_METHODS:
        spline_kwargs = dict(filtered)

        n_knots = kwargs.get("n_knots")
        if n_knots is not None:
            spline_kwargs["n_knots"] = _clamp_spline_basis(n_knots)

        strategy = kwargs.get("strategy")
        if strategy in ("uniform", "quantile"):
            spline_kwargs["knot_strategy"] = strategy

        if kwargs.get("use_decision_tree"):
            spline_kwargs["knot_selector"] = CARTKnotSelector(
                degree=spline_kwargs.get("degree", 3),
                spline_type=method,
            )

        steps.append((method, cls(**spline_kwargs)))
    else:
        name = method if method != "none" else "noop"
        steps.append((name, cls(**filtered)))

    return steps
