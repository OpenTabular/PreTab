"""Normalized, validated configuration for a :class:`Preprocessor` run.

:class:`PreprocessorConfig` is the frozen, canonical view of the user-supplied
Preprocessor parameters. It normalizes the global method names (resolving
aliases and separator/case variants, mapping ``None`` to ``"none"``) and
validates the global ``target_aware`` / ``placement_strategy`` contract and the
``task`` up front. The user's original constructor arguments stay untouched on
the estimator (as scikit-learn requires); this object is the internal, normalized
counterpart the composition layer consumes.

Per-column overrides in ``feature_preprocessing`` are kept verbatim because the
namespace they resolve in (numerical vs categorical) depends on the column type,
which is only known after feature detection; :meth:`PreprocessorConfig.method_for`
resolves them in the correct namespace at build time.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from math import isfinite
from typing import Any

from ..core.parameters import UNSET, _Unset, validate_placement, validate_task
from ..exceptions import invalid_param_error
from .registry import (
    CATEGORICAL_ALIASES,
    CATEGORICAL_METHODS,
    NUMERICAL_ALIASES,
    NUMERICAL_METHODS,
    resolve_method,
)

__all__ = ["PreprocessorConfig", "RepresentationConfig"]

# Valid values for the high-level ``missing_policy`` orchestration knob. ``None``
# keeps the explicit imputation parameters authoritative (historical behaviour).
MISSING_POLICIES = frozenset({"error", "propagate", "impute", "impute_with_indicator", "separate_state"})


@dataclass(frozen=True, slots=True, eq=False)
class _FrozenParams(Mapping[str, Any]):
    _entries: tuple[tuple[str, Any], ...]

    def __getitem__(self, key: str) -> Any:
        for stored_key, value in self._entries:
            if stored_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return repr(dict(self))


def _freeze_config_value(value: Any, path: str, active: set[int]) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and isfinite(value):
        return value
    if not isinstance(value, (Mapping, Sequence)) or isinstance(value, (str, bytes, bytearray, memoryview)):
        raise invalid_param_error(
            "RepresentationConfig", path, value, "must contain only finite JSON-compatible values"
        )
    identity = id(value)
    if identity in active:
        raise invalid_param_error("RepresentationConfig", path, value, "must not contain circular references")
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            entries = []
            for key, item in value.items():
                if type(key) is not str:
                    raise invalid_param_error("RepresentationConfig", path, key, "mapping keys must be strings")
                entries.append((key, _freeze_config_value(item, f"{path}[{key!r}]", active)))
            return _FrozenParams(tuple(entries))
        return tuple(_freeze_config_value(item, f"{path}[{index}]", active) for index, item in enumerate(value))
    finally:
        active.remove(identity)


def _config_value_to_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _config_value_to_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_config_value_to_json(item) for item in value]
    return value


@dataclass(frozen=True)
class RepresentationConfig:
    """Immutable, JSON-compatible settings for one representation.

    Parameters
    ----------
    method : str
        Requested method name. Alias resolution and parameter applicability
        require a numerical or categorical context and are checked separately.
    output_dim : int, optional
        Positive requested width, subject to the selected family's constraints.
    target_aware : bool, optional
        Whether to use the target for placement.
    placement_strategy : str, optional
        Requested nonempty placement-strategy name.
    adaptive : bool, optional
        Whether to use adaptive resolution.
    min_output_dim, max_output_dim : int or None, optional
        Positive adaptive bounds, or ``None`` where the method permits it.
    params : mapping, optional
        Family-specific options. Nested mappings and sequences are copied into
        read-only storage. Only finite JSON-compatible values are accepted.
    name : str or None, default=None
        Optional metadata label; it does not rename input or output columns.

    Notes
    -----
    Omitted common settings retain ``UNSET`` rather than adopting method defaults.
    This value is not an estimator; it has no fitted state or ``set_params``.
    """

    method: str
    output_dim: int | _Unset = UNSET
    target_aware: bool | _Unset = UNSET
    placement_strategy: str | _Unset = UNSET
    adaptive: bool | _Unset = UNSET
    min_output_dim: int | _Unset | None = UNSET
    max_output_dim: int | _Unset | None = UNSET
    params: Mapping[str, Any] = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self) -> None:
        if self.method is UNSET:
            raise invalid_param_error("RepresentationConfig", "method", self.method, "is required")
        for key in ("method", "placement_strategy", "name"):
            value = getattr(self, key)
            if (key == "placement_strategy" and value is UNSET) or (key == "name" and value is None):
                continue
            if type(value) is not str or not value.strip():
                raise invalid_param_error("RepresentationConfig", key, value, "must be a nonempty string")
        for key in ("output_dim", "min_output_dim", "max_output_dim"):
            value = getattr(self, key)
            if value is UNSET or (key != "output_dim" and value is None):
                continue
            if type(value) is not int or value < 1:
                raise invalid_param_error("RepresentationConfig", key, value, "must be a positive integer")
        for key in ("target_aware", "adaptive"):
            value = getattr(self, key)
            if value is not UNSET and type(value) is not bool:
                raise invalid_param_error("RepresentationConfig", key, value, "must be a boolean")
        if not isinstance(self.params, Mapping):
            raise invalid_param_error("RepresentationConfig", "params", self.params, "must be a mapping")
        reserved = {item.name for item in fields(self)}
        for key in self.params:
            if key in reserved:
                raise invalid_param_error(
                    "RepresentationConfig",
                    f"params[{key!r}]",
                    self.params[key],
                    "must be supplied as a top-level field",
                )
        object.__setattr__(self, "params", _freeze_config_value(self.params, "params", set()))

    def to_dict(self) -> dict[str, Any]:
        """Return detached JSON-compatible settings, omitting ``UNSET`` fields."""
        return {
            item.name: _config_value_to_json(value)
            for item in fields(self)
            if (value := getattr(self, item.name)) is not UNSET
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RepresentationConfig:
        """Build a value from a mapping, collecting family options into ``params``.

        An option supplied both at the top level and in ``params`` is an error,
        even if the two values agree. Method-specific validation requires the
        feature-kind context and is not performed by this constructor.
        """
        if not isinstance(data, Mapping):
            raise invalid_param_error("RepresentationConfig", "data", data, "must be a mapping")
        requested = dict(data)
        if any(type(key) is not str for key in requested):
            raise invalid_param_error("RepresentationConfig", "data", data, "mapping keys must be strings")
        if "method" not in requested:
            raise invalid_param_error("RepresentationConfig", "method", UNSET, "is required")
        params = requested.pop("params", {})
        if not isinstance(params, Mapping):
            raise invalid_param_error("RepresentationConfig", "params", params, "must be a mapping")
        params = dict(params)
        common = {item.name for item in fields(cls)}
        for key in list(requested):
            if key not in common:
                if key in params:
                    raise invalid_param_error(
                        "RepresentationConfig", key, requested[key], "must not also be supplied inside params"
                    )
                params[key] = requested.pop(key)
        return cls(**requested, params=params)


def _normalize_representation_config(
    value: str | Mapping[str, Any] | RepresentationConfig, *, is_numerical: bool
) -> RepresentationConfig:
    """Normalize one request in its known feature-kind namespace.

    Raw mappings detach through ``from_dict``. Typed values retain their
    requested spelling; normalization replaces rather than mutates an alias.
    Omitted fields remain ``UNSET``. Method applicability, capability checks,
    and effective defaults are not resolved here.
    """
    if isinstance(value, RepresentationConfig):
        config = value
    elif isinstance(value, str):
        config = RepresentationConfig(method=value)
    elif isinstance(value, Mapping):
        config = RepresentationConfig.from_dict(value)
    else:
        raise invalid_param_error(
            "RepresentationConfig", "value", value, "must be a string, mapping, or RepresentationConfig"
        )
    canonical = NUMERICAL_METHODS if is_numerical else CATEGORICAL_METHODS
    aliases = NUMERICAL_ALIASES if is_numerical else CATEGORICAL_ALIASES
    method = resolve_method(config.method, canonical, aliases)
    return config if method == config.method else replace(config, method=method)


def _normalize_method(method, canonical, aliases) -> str:
    """Resolve a global method name to canonical form, mapping ``None`` to ``"none"``."""
    if method is None:
        return "none"
    return resolve_method(method, canonical, aliases)


@dataclass(frozen=True)
class PreprocessorConfig:
    """Frozen, normalized configuration derived from Preprocessor parameters.

    Built via :meth:`from_params`, which normalizes the global method names and
    validates the placement contract and the task. All other knobs are carried
    through as-is for the factory and orchestration layers.
    """

    numerical_method: str
    categorical_method: str
    feature_preprocessing: dict
    output_dim: int
    degree: int
    target_aware: bool
    placement_strategy: str
    task: str
    adaptive: bool
    min_output_dim: int
    max_output_dim: int
    random_state: int | None
    scaling: str | None
    cat_cutoff: float | int
    treat_all_integers_as_numerical: bool
    numerical_imputation: str | None
    categorical_imputation: str | None
    add_missing_indicator: bool
    missing_policy: str | None
    verbose: int

    @classmethod
    def from_params(
        cls,
        *,
        numerical_method,
        categorical_method,
        feature_preprocessing,
        output_dim,
        degree,
        target_aware,
        placement_strategy,
        task,
        adaptive,
        min_output_dim,
        max_output_dim,
        random_state,
        scaling,
        cat_cutoff,
        treat_all_integers_as_numerical,
        numerical_imputation,
        categorical_imputation,
        add_missing_indicator,
        missing_policy,
        verbose,
    ) -> PreprocessorConfig:
        """Normalize and validate raw Preprocessor parameters into a config.

        Raises
        ------
        InvalidParamError
            If the ``target_aware`` / ``placement_strategy`` combination is
            invalid (via :func:`~pretab.core.parameters.validate_placement`), or
            ``task`` is not ``"regression"`` / ``"classification"`` (via
            :func:`~pretab.core.parameters.validate_task`).
        """
        validate_placement(target_aware, placement_strategy)
        validate_task(task, "Preprocessor")
        if missing_policy is not None and missing_policy not in MISSING_POLICIES:
            raise invalid_param_error(
                "Preprocessor",
                "missing_policy",
                missing_policy,
                "must be None or one of 'error', 'propagate', 'impute', 'impute_with_indicator', 'separate_state'",
                valid=set(MISSING_POLICIES),
            )
        return cls(
            numerical_method=_normalize_method(numerical_method, NUMERICAL_METHODS, NUMERICAL_ALIASES),
            categorical_method=_normalize_method(categorical_method, CATEGORICAL_METHODS, CATEGORICAL_ALIASES),
            feature_preprocessing=dict(feature_preprocessing or {}),
            output_dim=output_dim,
            degree=degree,
            target_aware=target_aware,
            placement_strategy=placement_strategy,
            task=task,
            adaptive=adaptive,
            min_output_dim=min_output_dim,
            max_output_dim=max_output_dim,
            random_state=random_state,
            scaling=scaling,
            cat_cutoff=cat_cutoff,
            treat_all_integers_as_numerical=treat_all_integers_as_numerical,
            numerical_imputation=numerical_imputation,
            categorical_imputation=categorical_imputation,
            add_missing_indicator=add_missing_indicator,
            missing_policy=missing_policy,
            verbose=verbose,
        )

    @staticmethod
    def resolve_numerical(method) -> str:
        """Resolve a numerical method name to its canonical spelling."""
        return resolve_method(method, NUMERICAL_METHODS, NUMERICAL_ALIASES)

    @staticmethod
    def resolve_categorical(method) -> str:
        """Resolve a categorical method name to its canonical spelling."""
        return resolve_method(method, CATEGORICAL_METHODS, CATEGORICAL_ALIASES)

    def method_for(self, feature, *, is_numerical: bool) -> str:
        """Return the resolved method for ``feature`` given its detected kind.

        A per-column override in ``feature_preprocessing`` wins over the global
        default; the chosen name is resolved in the numerical or categorical
        namespace according to ``is_numerical``.
        """
        default = self.numerical_method if is_numerical else self.categorical_method
        raw = self.feature_preprocessing.get(feature, default)
        return self.resolve_numerical(raw) if is_numerical else self.resolve_categorical(raw)

    @property
    def seed_kwargs(self) -> dict:
        """Return ``{"random_state": ...}`` only when a seed was explicitly set.

        Leaving it empty when ``random_state`` is ``None`` preserves each
        transformer's own default seed, matching the historical behaviour where a
        seed is forwarded only when the user pins one.
        """
        return {} if self.random_state is None else {"random_state": self.random_state}

    def imputation_plan(self, *, is_numerical: bool) -> dict:
        """Resolve how missing values are handled for one column kind.

        Returns a dict with ``add_imputer`` / ``add_indicator`` / ``separate_state``
        booleans and the imputer ``strategy``. When ``missing_policy`` is ``None``
        the explicit ``*_imputation`` / ``add_missing_indicator`` parameters stay
        authoritative (historical behaviour); otherwise ``missing_policy`` decides.
        ``add_indicator`` appends a missing indicator built on the raw input next
        to the representation (see :func:`~pretab.compose.factory.create_transformer`).
        ``add_missing_indicator=True`` with imputation disabled for this column
        kind routes through the standalone ``MissingStateIndicator`` (via
        ``separate_state``) instead, preserving its one-column-per-feature output.
        """
        strategy = (
            (self.numerical_imputation or "median")
            if is_numerical
            else (self.categorical_imputation or "most_frequent")
        )
        if self.missing_policy is None:
            configured = self.numerical_imputation if is_numerical else self.categorical_imputation
            add_imputer = configured is not None
            return {
                "add_imputer": add_imputer,
                "add_indicator": self.add_missing_indicator and add_imputer,
                "separate_state": self.add_missing_indicator and not add_imputer,
                "strategy": strategy,
            }
        if self.missing_policy in ("error", "propagate"):
            return {"add_imputer": False, "add_indicator": False, "separate_state": False, "strategy": strategy}
        if self.missing_policy == "impute":
            return {"add_imputer": True, "add_indicator": False, "separate_state": False, "strategy": strategy}
        if self.missing_policy == "impute_with_indicator":
            return {"add_imputer": True, "add_indicator": True, "separate_state": False, "strategy": strategy}
        # "separate_state": impute for the basis, emit a dedicated __missing column
        # (added by the factory as a separate branch that bypasses the basis).
        return {"add_imputer": True, "add_indicator": False, "separate_state": True, "strategy": strategy}
