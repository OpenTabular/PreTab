import warnings

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..core.exceptions import ConfigWarning, invalid_param_error
from ..transformers.splines.knot_selectors import CARTKnotSelector, LightGBMKnotSelector
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

# spline_type passed to the knot selector for each target-aware spline family.
_SELECTOR_SPLINE_TYPE = {
    "bspline": "bspline",
    "mspline": "mspline",
    "ispline": "ispline",
    "cubicspline": "bspline",
    "naturalspline": "bspline",
}


def supports_target_aware(method: str) -> bool:
    """Return whether a spline ``method`` supports target-aware knot placement.

    Only freely-placed knot splines qualify: ``bspline``, ``mspline``,
    ``ispline``, ``cubicspline`` and ``naturalspline``. The penalized splines
    (``pspline``, ``tensorspline``) require equally-spaced knots for their
    difference penalty, and the kernel-based ``tprs`` has no knots, so those
    three always use fixed knot placement regardless of ``use_target`` /
    ``selector`` / ``adaptive``.
    """
    resolved = resolve_method(method, NUMERICAL_METHODS, NUMERICAL_ALIASES)
    return resolved in TARGET_AWARE_SPLINE_METHODS

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


def _wants_target(kwargs):
    """Whether the caller asked for target-aware (selector-driven) placement."""
    value = kwargs.get("use_target")
    if value is None:
        value = kwargs.get("use_decision_tree")
    return bool(value)


def _build_knot_selector(spline_type, degree, kwargs):
    """Build a target-aware knot selector for a spline family.

    ``spline_type`` is the selector's basis type (``"bspline"`` / ``"mspline"`` /
    ``"ispline"``); the ``selector`` kwarg picks ``"cart"`` (default) or
    ``"lightgbm"``. ``random_state`` is forwarded only when the caller set one.
    """
    selector_name = kwargs.get("selector") or "cart"
    selector_kwargs = {"degree": degree, "spline_type": spline_type}
    if kwargs.get("random_state") is not None:
        selector_kwargs["random_state"] = kwargs["random_state"]
    if selector_name == "cart":
        return CARTKnotSelector(**selector_kwargs)
    if selector_name == "lightgbm":
        return LightGBMKnotSelector(**selector_kwargs)
    raise invalid_param_error(
        "get_numerical_transformer_steps", "selector", selector_name,
        "must be 'cart' or 'lightgbm'", valid={"cart", "lightgbm"},
    )


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
    elif method in SPLINE_EXPANSION_METHODS or method in LEGACY_SPLINE_METHODS:
        spline_kwargs = dict(filtered)

        # The B/M/I splines share the Preprocessor's default ``output_dim`` (which
        # can sit outside their [5, 50] basis range); the legacy families keep
        # their own wider bounds, so only clamp for B/M/I.
        if method in SPLINE_EXPANSION_METHODS:
            output_dim = kwargs.get("output_dim")
            if output_dim is not None:
                spline_kwargs["output_dim"] = _clamp_spline_basis(output_dim)

        strategy = kwargs.get("strategy")
        if strategy in ("uniform", "quantile"):
            spline_kwargs["strategy"] = strategy

        if _wants_target(kwargs):
            spline_kwargs["selector"] = _build_knot_selector(
                _SELECTOR_SPLINE_TYPE[method], spline_kwargs.get("degree", 3), kwargs
            )
            # The adaptive window only takes effect on the target-aware
            # (selector) path, so forward it here: each feature is then sized
            # within [min_output_dim, max_output_dim] instead of the fixed
            # output_dim.
            if kwargs.get("adaptive"):
                spline_kwargs["adaptive"] = True
                for bound in ("min_output_dim", "max_output_dim"):
                    if kwargs.get(bound) is not None:
                        spline_kwargs[bound] = kwargs[bound]

        steps.append((method, cls(**spline_kwargs)))
    else:
        name = method if method != "none" else "noop"
        steps.append((name, cls(**filtered)))

    return steps
