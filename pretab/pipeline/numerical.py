import warnings

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..core.exceptions import ConfigWarning, invalid_param_error
from ..transformers.splines.knot_selectors import CARTKnotSelector
from .registry import NUMERICAL_ALIASES, NUMERICAL_METHODS, resolve_method

# Spline basis expansions that share the target-aware knot API.
SPLINE_EXPANSION_METHODS = ("bspline", "mspline", "ispline")

# Valid range for the number of spline basis functions per feature.
_MIN_SPLINE_BASIS = 5
_MAX_SPLINE_BASIS = 50

# Legacy constructor argument names mapped to the canonical vocabulary. The
# Preprocessor shares a single kwargs bag; every width name now collapses to the
# canonical ``output_dim`` at the transformer boundary, so only the non-count
# alias (``use_decision_tree`` -> ``use_target``) needs translating here to keep
# the feature-map constructors on their canonical spelling and avoid emitting
# deprecation warnings during normal pipeline use.
_ARG_CANONICAL = {
    "use_decision_tree": "use_target",
}


def _canonicalize(filtered):
    """Rename legacy constructor argument keys to their canonical spelling."""
    return {_ARG_CANONICAL.get(key, key): value for key, value in filtered.items()}


def filter_kwargs(transformer_cls, kwargs, allowed=None):
    if allowed is not None:
        return {k: kwargs[k] for k in allowed if k in kwargs}
    return kwargs


def _clamp_spline_basis(output_dim):
    """Clamp a requested output dimension into the supported spline range.

    The Preprocessor shares a single ``output_dim`` setting across every
    numerical strategy (default 64), but the B/M/I spline transformers accept
    between ``5`` and ``50`` basis functions. Values outside that window are
    clamped so switching to a spline strategy keeps working with the shared
    default.
    """
    clamped = max(_MIN_SPLINE_BASIS, min(int(output_dim), _MAX_SPLINE_BASIS))
    if clamped != output_dim:
        warnings.warn(
            f"output_dim={output_dim} is outside the spline range "
            f"[{_MIN_SPLINE_BASIS}, {_MAX_SPLINE_BASIS}]; using {clamped} basis functions.",
            ConfigWarning,
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
    method = resolve_method(method, NUMERICAL_METHODS, NUMERICAL_ALIASES)
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
    if scaling is not None:
        scaling = resolve_method(scaling, NUMERICAL_METHODS, NUMERICAL_ALIASES)
    if scaling in scalers and scaling != method:
        steps.append(scalers[scaling])

    if method not in NUMERICAL_METHODS:
        raise invalid_param_error(
            "get_numerical_transformer_steps", "method", method,
            "unrecognized numerical preprocessing method",
            valid=set(NUMERICAL_METHODS),
        )

    cls, allowed_args = NUMERICAL_METHODS[method]
    filtered = _canonicalize(filter_kwargs(cls, kwargs, allowed=allowed_args))

    if method == "box-cox":
        steps.append(("scale_positive", MinMaxScaler(feature_range=(1e-3, 1))))
        steps.append(("boxcox", cls(method="box-cox", **filtered)))
    elif method == "yeo-johnson":
        steps.append(("yeojohnson", cls(method="yeo-johnson", **filtered)))
    elif method in SPLINE_EXPANSION_METHODS:
        spline_kwargs = dict(filtered)

        output_dim = kwargs.get("output_dim")
        if output_dim is not None:
            spline_kwargs["output_dim"] = _clamp_spline_basis(output_dim)

        strategy = kwargs.get("strategy")
        if strategy in ("uniform", "quantile"):
            spline_kwargs["strategy"] = strategy

        if kwargs.get("use_decision_tree"):
            selector_kwargs = {
                "degree": spline_kwargs.get("degree", 3),
                "spline_type": method,
            }
            if kwargs.get("random_state") is not None:
                selector_kwargs["random_state"] = kwargs["random_state"]
            spline_kwargs["selector"] = CARTKnotSelector(**selector_kwargs)

        steps.append((method, cls(**spline_kwargs)))
    else:
        name = method if method != "none" else "noop"
        steps.append((name, cls(**filtered)))

    return steps
