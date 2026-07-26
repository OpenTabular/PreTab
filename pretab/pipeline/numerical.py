import warnings

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..core.exceptions import ConfigWarning, invalid_param_error
from .registry import NUMERICAL_ALIASES, NUMERICAL_METHODS, resolve_method

# Spline basis expansions that share the target-aware knot API.
SPLINE_EXPANSION_METHODS = ("bspline", "mspline", "ispline")

# Legacy knot-based spline families that also support target-aware placement.
# They use freely-placed knots (cubic / natural-cubic regression splines), so the
# selector / task / strategy / adaptive knobs apply; their knot selector uses the
# ``"bspline"`` spline_type. The penalized families (``pspline``, ``tensorspline``)
# assume equally-spaced knots for their difference penalty, and the thin-plate
# spline (``tprs``) is kernel-based (knot-free): none of those three are
# target-aware, so they stay on the generic fixed construction path.
LEGACY_SPLINE_METHODS = ("cubicspline", "naturalspline")

# Every spline family for which target-aware (data-driven) knot placement is
# meaningful. Exposed via :func:`supports_target_aware` so callers can query it.
TARGET_AWARE_SPLINE_METHODS = SPLINE_EXPANSION_METHODS + LEGACY_SPLINE_METHODS


def supports_target_aware(method: str) -> bool:
    """Return whether a spline ``method`` supports target-aware knot placement.

    Only freely-placed knot splines qualify: ``bspline``, ``mspline``,
    ``ispline``, ``cubicspline`` and ``naturalspline``. The penalized splines
    (``pspline``, ``tensorspline``) require equally-spaced knots for their
    difference penalty, and the kernel-based ``tprs`` has no knots, so those
    three always use fixed knot placement regardless of ``target_aware`` /
    ``placement_strategy`` / ``adaptive``.
    """
    resolved = resolve_method(method, NUMERICAL_METHODS, NUMERICAL_ALIASES)
    return resolved in TARGET_AWARE_SPLINE_METHODS


# Valid range for the number of spline basis functions per feature.
_MIN_SPLINE_BASIS = 5
_MAX_SPLINE_BASIS = 50


def filter_kwargs(transformer_cls, kwargs, allowed=None):
    if allowed is not None:
        return {k: kwargs[k] for k in allowed if k in kwargs}
    return kwargs


# Method families grouped by which placement modes they support. The Preprocessor
# shares a single ``target_aware`` / ``placement_strategy`` pair; each family only
# receives the placement kwargs it can honor.
BOTH_MODE_METHODS = frozenset(
    {
        "rbf",
        "relu",
        "sigmoid",
        "tanh",
        "bspline",
        "mspline",
        "ispline",
        "cubicspline",
        "naturalspline",
    }
)
# PLE is inherently target-aware: only the supervised selectors apply.
TARGET_AWARE_ONLY_METHODS = frozenset({"ple"})
# Penalized splines assume equally-spaced knots: only the spacing rules apply.
UNSUPERVISED_ONLY_METHODS = frozenset({"pspline", "tensorspline"})


def _placement_kwargs(method, kwargs):
    """Return the placement kwargs to inject for ``method``.

    Honors each family's applicability: both-mode families receive
    ``target_aware`` + ``placement_strategy``; PLE receives a supervised
    ``placement_strategy`` only when target-aware; the penalized splines receive
    an unsupervised ``placement_strategy`` only when not target-aware. Anything
    else receives nothing.
    """
    target_aware = bool(kwargs.get("target_aware", False))
    placement_strategy = kwargs.get("placement_strategy")
    if method in BOTH_MODE_METHODS:
        out = {"target_aware": target_aware}
        if placement_strategy is not None:
            out["placement_strategy"] = placement_strategy
        return out
    if method in TARGET_AWARE_ONLY_METHODS:
        if target_aware and placement_strategy in ("cart", "lightgbm"):
            return {"placement_strategy": placement_strategy}
        return {}
    if method in UNSUPERVISED_ONLY_METHODS:
        if not target_aware and placement_strategy in ("uniform", "quantile"):
            return {"placement_strategy": placement_strategy}
        return {}
    return {}


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
    imputer_kwargs: dict | None = None,
    scaling: str | None = None,
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
            "get_numerical_transformer_steps",
            "method",
            method,
            "unrecognized numerical preprocessing method",
            valid=set(NUMERICAL_METHODS),
        )

    cls, allowed_args = NUMERICAL_METHODS[method]
    filtered = filter_kwargs(cls, kwargs, allowed=allowed_args)
    placement = _placement_kwargs(method, kwargs)

    if method == "box-cox":
        steps.append(("scale_positive", MinMaxScaler(feature_range=(1e-3, 1))))
        steps.append(("boxcox", cls(method="box-cox", **filtered)))
    elif method == "yeo-johnson":
        steps.append(("yeojohnson", cls(method="yeo-johnson", **filtered)))
    elif method in SPLINE_EXPANSION_METHODS or method in LEGACY_SPLINE_METHODS:
        spline_kwargs = dict(filtered)
        spline_kwargs.update(placement)

        # The B/M/I splines share the Preprocessor's default ``output_dim`` (which
        # can sit outside their [5, 50] basis range); the legacy families keep
        # their own wider bounds, so only clamp for B/M/I.
        if method in SPLINE_EXPANSION_METHODS:
            output_dim = kwargs.get("output_dim")
            if output_dim is not None:
                spline_kwargs["output_dim"] = _clamp_spline_basis(output_dim)

        steps.append((method, cls(**spline_kwargs)))
    else:
        name = method if method != "none" else "noop"
        call_kwargs = dict(filtered)
        call_kwargs.update(placement)
        steps.append((name, cls(**call_kwargs)))

    return steps
