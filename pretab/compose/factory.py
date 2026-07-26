"""Build the per-column transformer pipelines and combine them.

This module turns a resolved method name plus a :class:`PreprocessorConfig` into
scikit-learn transformer steps, wraps them in a per-column
:class:`~sklearn.pipeline.Pipeline`, and assembles every column into the final
:class:`~sklearn.compose.ColumnTransformer`. The class to instantiate, the
constructor arguments it accepts, and which placement keyword arguments apply are
all taken from :data:`~pretab.compose.registry.TRANSFORMER_REGISTRY`.
"""

import warnings

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..core.parameters import UNSET
from ..exceptions import ConfigWarning, invalid_param_error
from ..transformers.encoders.floats import ToFloatTransformer
from .config import PreprocessorConfig
from .registry import (
    CATEGORICAL_ALIASES,
    CATEGORICAL_METHODS,
    NUMERICAL_ALIASES,
    NUMERICAL_METHODS,
    TransformerSpec,
    get_spec,
    resolve_method,
)

__all__ = [
    "build_column_transformer",
    "create_transformer",
    "get_categorical_transformer_steps",
    "get_numerical_transformer_steps",
]

# Valid range for the number of B/M/I spline basis functions per feature. The
# Preprocessor shares a single ``output_dim`` across every numerical strategy
# (default 7); values outside this window are clamped for the B/M/I splines.
_MIN_SPLINE_BASIS = 5
_MAX_SPLINE_BASIS = 50

# B/M/I spline bases whose shared ``output_dim`` is clamped into the basis range.
_BMI_SPLINE_METHODS = frozenset({"bspline", "mspline", "ispline"})
# Freely-placed knot splines built through the knot-wiring construction path
# (B/M/I plus the legacy cubic / natural-cubic regression splines).
_KNOT_SPLINE_METHODS = _BMI_SPLINE_METHODS | frozenset({"cubicspline", "naturalspline"})


def _filter_kwargs(allowed, kwargs):
    """Keep only the ``allowed`` keyword arguments that are present in ``kwargs``."""
    return {key: kwargs[key] for key in allowed if key in kwargs}


def _clamp_spline_basis(output_dim):
    """Clamp a requested output dimension into the supported B/M/I spline range.

    Values outside ``[5, 50]`` are clamped and a :class:`ConfigWarning` is
    emitted so switching to a B/M/I spline keeps working with the shared default.
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


def _placement_kwargs(spec: TransformerSpec, kwargs):
    """Return the placement kwargs to inject for a method, honouring its capability.

    Mirrors the shared-placement contract: methods with optional target awareness
    (feature maps and freely-placed knot splines) receive ``target_aware`` plus
    the ``placement_strategy`` (when set); the always-target-aware ``ple`` receives
    a supervised ``placement_strategy`` only when target-aware; the unsupervised-only
    penalized splines receive an unsupervised ``placement_strategy`` only when not
    target-aware. Methods without data-driven placement receive nothing.
    """
    if not spec.placement_strategies:
        return {}

    target_aware = bool(kwargs.get("target_aware", False))
    placement_strategy = kwargs.get("placement_strategy")

    if spec.target_usage == "optional":
        out = {"target_aware": target_aware}
        if placement_strategy is not None:
            out["placement_strategy"] = placement_strategy
        return out
    if spec.target_usage == "required":
        if target_aware and placement_strategy in ("cart", "lightgbm"):
            return {"placement_strategy": placement_strategy}
        return {}
    # target_usage == "forbidden" but with unsupervised placement (pspline / tensorspline).
    if not target_aware and placement_strategy in ("uniform", "quantile"):
        return {"placement_strategy": placement_strategy}
    return {}


def get_numerical_transformer_steps(
    method: str,
    add_imputer: bool = True,
    imputer_strategy: str = "mean",
    imputer_kwargs: dict | None = None,
    scaling: str | None = None,
    **kwargs,
):
    """Return the ordered ``(name, transformer)`` steps for a numerical ``method``."""
    method = resolve_method(method, NUMERICAL_METHODS, NUMERICAL_ALIASES)
    steps = []

    if add_imputer:
        imputer_kwargs = imputer_kwargs or {}
        steps.append(("imputer", SimpleImputer(strategy=imputer_strategy, **imputer_kwargs)))

    # Optional scaling step, added only when it is not already the chosen method.
    scalers = {
        "standardization": ("scaler", StandardScaler()),
        "minmax": ("minmax", MinMaxScaler(feature_range=(-1, 1))),
    }
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

    spec = get_spec(method)
    cls = spec.transformer_cls
    filtered = _filter_kwargs(spec.allowed_args, kwargs)
    placement = _placement_kwargs(spec, kwargs)

    if method == "box-cox":
        steps.append(("scale_positive", MinMaxScaler(feature_range=(1e-3, 1))))
        steps.append(("boxcox", cls(method="box-cox", **filtered)))
    elif method == "yeo-johnson":
        steps.append(("yeojohnson", cls(method="yeo-johnson", **filtered)))
    elif method in _KNOT_SPLINE_METHODS:
        spline_kwargs = dict(filtered)
        spline_kwargs.update(placement)

        # The B/M/I splines share the Preprocessor's default ``output_dim`` (which
        # can sit outside their [5, 50] basis range); the legacy families keep
        # their own wider bounds, so only clamp for B/M/I.
        if method in _BMI_SPLINE_METHODS:
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


def get_categorical_transformer_steps(
    method: str,
    add_imputer: bool = True,
    imputer_strategy: str = "most_frequent",
    imputer_kwargs: dict | None = None,
    output_dim=UNSET,
    **kwargs,
):
    """Return the ordered ``(name, transformer)`` steps for a categorical ``method``."""
    method = resolve_method(method, CATEGORICAL_METHODS, CATEGORICAL_ALIASES)
    steps = []

    if add_imputer:
        imputer_kwargs = imputer_kwargs or {}
        steps.append(("imputer", SimpleImputer(strategy=imputer_strategy, **imputer_kwargs)))

    if method not in CATEGORICAL_METHODS:
        raise invalid_param_error(
            "get_categorical_transformer_steps",
            "method",
            method,
            "unrecognized categorical preprocessing method",
            valid=set(CATEGORICAL_METHODS),
        )

    cls = get_spec(method).transformer_cls

    if method == "int":
        steps.append(("continuous_ordinal", cls()))
    elif method == "one-hot":
        # Default to ignoring unseen categories so transform never crashes on
        # categories absent at fit time; callers can override via kwargs.
        onehot_kwargs = {"handle_unknown": "ignore", **kwargs}
        steps.append(("onehot", cls(**onehot_kwargs)))
        steps.append(("to_float", ToFloatTransformer()))
    elif method == "pretrained":
        steps.append(("pretrained", cls()))
    elif method == "none":
        steps.append(("none", cls()))
    elif method == "custombin":
        bin_kwargs = dict(kwargs)
        if output_dim is not UNSET:
            bin_kwargs.setdefault("output_dim", output_dim)
        steps.append(("custombin", cls(**bin_kwargs)))
    elif method == "onehot_from_ordinal":
        steps.append(("onehot_from_ordinal", cls()))

    return steps


def create_transformer(method: str, *, is_numerical: bool, config: PreprocessorConfig) -> Pipeline:
    """Build the per-column :class:`~sklearn.pipeline.Pipeline` for one feature.

    ``method`` is the resolved method name; ``is_numerical`` selects the numerical
    or categorical construction path. All width / placement / seeding knobs are
    taken from ``config``.
    """
    if is_numerical:
        steps = get_numerical_transformer_steps(
            method=method,
            task=config.task,
            target_aware=config.target_aware,
            add_imputer=config.handle_missing != "error",
            imputer_strategy="mean",
            output_dim=config.output_dim,
            adaptive=config.adaptive,
            min_output_dim=config.min_output_dim if config.adaptive else None,
            max_output_dim=config.max_output_dim if config.adaptive else None,
            degree=config.degree,
            scaling=config.scaling,
            placement_strategy=config.placement_strategy,
            handle_missing=config.handle_missing,
            **config.seed_kwargs,
        )
    else:
        steps = get_categorical_transformer_steps(method, output_dim=config.output_dim)
    return Pipeline(steps)


def build_column_transformer(config: PreprocessorConfig, numerical_features, categorical_features) -> ColumnTransformer:
    """Assemble the per-column pipelines into the final ColumnTransformer.

    Numerical features are prefixed ``num_`` and categorical features ``cat_`` to
    match the transformer names the Preprocessor exposes; untransformed columns
    pass through via ``remainder="passthrough"``.
    """
    transformers = []
    for feature in numerical_features:
        method = config.method_for(feature, is_numerical=True)
        pipeline = create_transformer(method, is_numerical=True, config=config)
        transformers.append((f"num_{feature}", pipeline, [feature]))
    for feature in categorical_features:
        method = config.method_for(feature, is_numerical=False)
        pipeline = create_transformer(method, is_numerical=False, config=config)
        transformers.append((f"cat_{feature}", pipeline, [feature]))
    return ColumnTransformer(transformers=transformers, remainder="passthrough")
