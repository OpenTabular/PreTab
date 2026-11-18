"""Normalized, validated configuration for a :class:`Preprocessor` run.

:class:`PreprocessorConfig` is the frozen, canonical view of the user-supplied
Preprocessor parameters. It normalizes the global method names (resolving
aliases and separator/case variants, mapping ``None`` to ``"none"``) and
validates the global ``target_aware`` / ``placement_strategy`` contract and the
``task`` up front. The user's original constructor arguments stay untouched on
the estimator (as scikit-learn requires); this object is the internal, normalized
counterpart the composition layer consumes.

Structured per-column overrides detach at normalization. Their method namespace
(numerical vs categorical) depends on feature detection; ``method_for`` keeps
returning a string, while ``representation_for`` resolves applicable settings.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, replace
from difflib import get_close_matches
from math import isfinite
from re import fullmatch
from typing import Any

from ..core.dependencies import _check_optional_dependency
from ..core.parameters import UNSET, _Unset, validate_placement, validate_task
from ..exceptions import InvalidParamError, invalid_param_error
from .registry import (
    CATEGORICAL_ALIASES,
    CATEGORICAL_METHODS,
    NUMERICAL_ALIASES,
    NUMERICAL_METHODS,
    TRANSFORMER_REGISTRY,
    resolve_method,
)

__all__ = ["FeatureGroup", "PreprocessorConfig", "RepresentationConfig"]

# Valid values for the high-level ``missing_policy`` orchestration knob. ``None``
# keeps the explicit imputation parameters authoritative (historical behaviour).
MISSING_POLICIES = frozenset({"error", "propagate", "impute", "impute_with_indicator", "separate_state"})


_REPRESENTATION_PATH = ContextVar("representation_config_path", default="RepresentationConfig")
_FEATURE_GROUP_PATH = ContextVar("feature_group_path", default="FeatureGroup")


def _representation_error(param, value, constraint, *, valid=None):
    """Use the standalone value name or the current normalization path."""
    return invalid_param_error(_REPRESENTATION_PATH.get(), param, value, constraint, valid=valid)


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
        raise _representation_error(path, value, "must contain only finite JSON-compatible values")
    identity = id(value)
    if identity in active:
        raise _representation_error(path, value, "must not contain circular references")
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            entries = []
            for key, item in value.items():
                if type(key) is not str:
                    raise _representation_error(path, key, "mapping keys must be strings")
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
            raise _representation_error("method", self.method, "is required")
        for key in ("method", "placement_strategy", "name"):
            value = getattr(self, key)
            if (key == "placement_strategy" and value is UNSET) or (key == "name" and value is None):
                continue
            if type(value) is not str or not value.strip():
                raise _representation_error(key, value, "must be a nonempty string")
        for key in ("output_dim", "min_output_dim", "max_output_dim"):
            value = getattr(self, key)
            if value is UNSET or (key != "output_dim" and value is None):
                continue
            if type(value) is not int or value < 1:
                raise _representation_error(key, value, "must be a positive integer")
        for key in ("target_aware", "adaptive"):
            value = getattr(self, key)
            if value is not UNSET and type(value) is not bool:
                raise _representation_error(key, value, "must be a boolean")
        if not isinstance(self.params, Mapping):
            raise _representation_error("params", self.params, "must be a mapping")
        reserved = {item.name for item in fields(self)}
        for key in self.params:
            if key in reserved:
                raise _representation_error(
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
            raise _representation_error("data", data, "must be a mapping")
        requested = dict(data)
        if any(type(key) is not str for key in requested):
            raise _representation_error("data", data, "mapping keys must be strings")
        if "method" not in requested:
            raise _representation_error("method", UNSET, "is required")
        params = requested.pop("params", {})
        if not isinstance(params, Mapping):
            raise _representation_error("params", params, "must be a mapping")
        params = dict(params)
        common = {item.name for item in fields(cls)}
        for key in list(requested):
            if key not in common:
                if key in params:
                    raise _representation_error(key, requested[key], "must not also be supplied inside params")
                params[key] = requested.pop(key)
        return cls(**requested, params=params)


def _validate_representation_capabilities(
    config: RepresentationConfig,
    *,
    is_numerical: bool,
    source: str | Mapping[str, Any] | RepresentationConfig,
    is_group: bool = False,
) -> None:
    """Check explicit settings against the live spec without choosing defaults."""
    spec = TRANSFORMER_REGISTRY[config.method]
    kind = "numerical" if is_numerical else "categorical"
    if kind not in spec.feature_kind:
        raise _representation_error("method", config.method, f"does not support {kind} features")
    if is_group and not spec.group_compatible:
        raise _representation_error("method", config.method, "does not support explicit feature groups")
    if not is_group and not spec.preprocessor_compatible:
        raise _representation_error("method", config.method, "is not selectable through Preprocessor")
    for key, value in config.params.items():
        if key not in spec.allowed_args:
            location = key if isinstance(source, Mapping) and key in source else f"params[{key!r}]"
            raise _representation_error(
                location, value, f"is not applicable to method {config.method!r}", valid=spec.allowed_args
            )
    if (
        config.output_dim is not UNSET
        and "output_dim" not in spec.allowed_args
        and not spec.supports_adaptive_resolution
    ):
        raise _representation_error("output_dim", config.output_dim, f"is not applicable to method {config.method!r}")
    for key in ("adaptive", "min_output_dim", "max_output_dim"):
        value = getattr(config, key)
        if value is not UNSET and not spec.supports_adaptive_resolution:
            raise _representation_error(key, value, f"method {config.method!r} does not support adaptive resolution")
    if config.target_aware is True and spec.target_usage == "forbidden":
        raise _representation_error("target_aware", True, f"method {config.method!r} cannot use the target")
    if config.target_aware is False and spec.target_usage == "required":
        raise _representation_error("target_aware", False, f"method {config.method!r} requires the target")
    if isinstance(config.placement_strategy, str):
        if config.placement_strategy not in spec.placement_strategies:
            raise _representation_error(
                "placement_strategy",
                config.placement_strategy,
                f"is not supported by method {config.method!r}",
                valid=spec.placement_strategies,
            )
        target_aware = config.target_aware
        if target_aware is UNSET and spec.target_usage != "optional":
            target_aware = spec.target_usage == "required"
        if isinstance(target_aware, bool):
            try:
                validate_placement(target_aware, config.placement_strategy)
            except InvalidParamError as error:
                raise _representation_error("placement_strategy", config.placement_strategy, str(error)) from error
    for key in ("min_output_dim", "max_output_dim"):
        value = getattr(config, key)
        if config.adaptive is False and value is not UNSET:
            raise _representation_error(key, value, "must not be supplied with adaptive=False")
    if config.adaptive is True and type(config.min_output_dim) is int and type(config.max_output_dim) is int:
        if config.min_output_dim > config.max_output_dim:
            raise _representation_error("max_output_dim", config.max_output_dim, "must be >= min_output_dim")
    path = _REPRESENTATION_PATH.get()
    if spec.optional_dependency:
        _check_optional_dependency(spec.optional_dependency, f"{path}.method = {config.method!r}")
    if config.placement_strategy == "lightgbm":
        _check_optional_dependency("lightgbm", f"{path}.placement_strategy = 'lightgbm'")


def _representation_request(value: str | Mapping[str, Any] | RepresentationConfig) -> RepresentationConfig:
    if isinstance(value, RepresentationConfig):
        return value
    if isinstance(value, str):
        return RepresentationConfig(method=value)
    if isinstance(value, Mapping):
        return RepresentationConfig.from_dict(value)
    raise _representation_error("value", value, "must be a string, mapping, or RepresentationConfig")


@dataclass(frozen=True)
class FeatureGroup:
    """Immutable declaration of ordered source columns and their representation.

    Parameters
    ----------
    name : str
        ASCII identifier beginning with a letter, without double underscores.
    columns : sequence
        Nonempty ordered source labels. Labels retain their original types;
        integer labels identify named columns rather than positions.
    representation : str, mapping, or RepresentationConfig
        Representation request. Mappings use the same common fields and family
        options as :class:`RepresentationConfig`.

    Notes
    -----
    Columns become a tuple and representation settings detach into an immutable
    value. This declaration has no fitted state and does not resolve an input
    schema or perform a transformation.
    """

    name: str
    columns: Sequence[Any]
    representation: str | Mapping[str, Any] | RepresentationConfig

    def __post_init__(self) -> None:
        path = _FEATURE_GROUP_PATH.get()
        if type(self.name) is not str or fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.name) is None or "__" in self.name:
            raise invalid_param_error(path, "name", self.name, "must be an ASCII identifier without double underscores")
        if not isinstance(self.columns, Sequence) or isinstance(self.columns, (str, bytes, bytearray, memoryview)):
            raise invalid_param_error(path, "columns", self.columns, "must be a nonempty ordered sequence of labels")
        if not self.columns:
            raise invalid_param_error(path, "columns", self.columns, "must be a nonempty ordered sequence of labels")
        columns = tuple(self.columns)
        for index, label in enumerate(columns):
            try:
                hash(label)
            except TypeError as error:
                raise invalid_param_error(
                    path, f"columns[{index}]", label, "must be a hashable column label"
                ) from error
        token = _REPRESENTATION_PATH.set(f"{path}.representation")
        try:
            representation = _representation_request(self.representation)
            if representation.name is not None and representation.name != self.name:
                raise _representation_error("name", representation.name, "must match the group name")
        finally:
            _REPRESENTATION_PATH.reset(token)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "representation", representation)


def _normalize_representation_config(
    value: str | Mapping[str, Any] | RepresentationConfig,
    *,
    is_numerical: bool,
    path: str = "RepresentationConfig",
    is_group: bool = False,
) -> RepresentationConfig:
    """Normalize one request in its known feature-kind namespace.

    Raw mappings detach through ``from_dict``. Typed values retain their
    requested spelling; normalization replaces rather than mutates an alias.
    Omitted fields remain ``UNSET``. Unknown methods/options and invalid value
    shapes raise path-qualified errors. Explicit options are checked against
    live registry capabilities; omitted effective defaults remain unresolved.
    """
    token = _REPRESENTATION_PATH.set(path)
    try:
        config = _representation_request(value)
        canonical = TRANSFORMER_REGISTRY if is_group else (NUMERICAL_METHODS if is_numerical else CATEGORICAL_METHODS)
        aliases = NUMERICAL_ALIASES if is_numerical else CATEGORICAL_ALIASES
        method = resolve_method(config.method, canonical, aliases)
        if method not in canonical:
            kind = "numerical" if is_numerical else "categorical"
            raise _representation_error(
                "method", method, f"unrecognized {kind} preprocessing method", valid=set(canonical)
            )
        common = {item.name for item in fields(RepresentationConfig)}
        known = common | {key for spec in TRANSFORMER_REGISTRY.values() for key in spec.allowed_args}
        applicable = common | set(TRANSFORMER_REGISTRY[method].allowed_args)
        for key in config.params:
            if key not in known:
                location = key if isinstance(value, Mapping) and key in value else f"params[{key!r}]"
                matches = get_close_matches(key, sorted(applicable), n=3)
                constraint = "unrecognized representation option"
                if matches:
                    constraint += f". Did you mean {', '.join(repr(match) for match in matches)}?"
                raise _representation_error(location, config.params[key], constraint, valid=applicable)
        normalized = config if method == config.method else replace(config, method=method)
        _validate_representation_capabilities(normalized, is_numerical=is_numerical, source=value, is_group=is_group)
        return normalized
    finally:
        _REPRESENTATION_PATH.reset(token)


def _normalize_feature_groups(value) -> tuple[FeatureGroup, ...]:
    if value is None:
        return ()
    mapping_form = isinstance(value, Mapping)
    if mapping_form:
        entries = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        entries = enumerate(value)
    else:
        raise invalid_param_error("Preprocessor", "feature_groups", value, "must be an ordered mapping or sequence")
    groups = []
    names = set()
    for key, entry in entries:
        path = f"feature_groups[{key!r}]"
        token = _FEATURE_GROUP_PATH.set(path)
        try:
            if mapping_form:
                if not isinstance(key, str):
                    raise invalid_param_error(path, "name", key, "must be a string")
                if not isinstance(entry, Mapping) or set(entry) != {"columns", "representation"}:
                    raise invalid_param_error(
                        "Preprocessor", path, entry, "mapping entries must contain exactly columns and representation"
                    )
                group = FeatureGroup(name=key, columns=entry["columns"], representation=entry["representation"])
            elif isinstance(entry, FeatureGroup):
                group = entry
            elif isinstance(entry, Mapping) and set(entry) == {"name", "columns", "representation"}:
                group = FeatureGroup(
                    name=entry["name"], columns=entry["columns"], representation=entry["representation"]
                )
            else:
                raise invalid_param_error(
                    "Preprocessor",
                    path,
                    entry,
                    "must be a FeatureGroup or a mapping with name, columns and representation",
                )
            if group.name in names:
                raise invalid_param_error(path, "name", group.name, "must be unique among feature groups")
            representation = _normalize_representation_config(
                group.representation,
                is_numerical=True,
                is_group=True,
                path=f"feature_groups[{group.name!r}].representation",
            )
            groups.append(replace(group, representation=representation))
            names.add(group.name)
        finally:
            _FEATURE_GROUP_PATH.reset(token)
    return tuple(groups)


def _normalize_method(method, canonical, aliases) -> str:
    """Resolve a global method name to canonical form, mapping ``None`` to ``"none"``."""
    if method is None:
        return "none"
    return resolve_method(method, canonical, aliases)


@dataclass(frozen=True)
class PreprocessorConfig:
    """Frozen, normalized configuration derived from Preprocessor parameters.

    Built via :meth:`from_params`, which normalizes the global method names and
    validates the placement contract and the task. Structured overrides detach
    into immutable requests; their feature-kind resolution is deferred until
    ``method_for`` or ``representation_for`` is called.
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
    feature_groups: tuple[FeatureGroup, ...] = ()

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
        feature_groups=None,
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
        overrides = dict(feature_preprocessing or {})
        for feature, value in overrides.items():
            if isinstance(value, str):
                continue
            token = _REPRESENTATION_PATH.set(f"feature_preprocessing[{feature!r}]")
            try:
                overrides[feature] = _representation_request(value)
            finally:
                _REPRESENTATION_PATH.reset(token)
        return cls(
            numerical_method=_normalize_method(numerical_method, NUMERICAL_METHODS, NUMERICAL_ALIASES),
            categorical_method=_normalize_method(categorical_method, CATEGORICAL_METHODS, CATEGORICAL_ALIASES),
            feature_preprocessing=overrides,
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
            feature_groups=_normalize_feature_groups(feature_groups),
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
        if isinstance(raw, (Mapping, RepresentationConfig)):
            return _normalize_representation_config(
                raw, is_numerical=is_numerical, path=f"feature_preprocessing[{feature!r}]"
            ).method
        return self.resolve_numerical(raw) if is_numerical else self.resolve_categorical(raw)

    def representation_for(self, feature, *, is_numerical: bool | None = None) -> RepresentationConfig:
        """Resolve one feature request over applicable constructor-wide controls.

        Method defaults remain omitted when the existing factory would not
        forward a constructor-wide value. Inherited inactive adaptive bounds
        are dropped, but contradictory explicit local bounds are rejected.
        Without a detected kind, resolve the declared method's namespace only;
        compatibility with the actual input still belongs to fitting.
        """
        default = self.categorical_method if is_numerical is False else self.numerical_method
        raw = self.feature_preprocessing.get(feature, default)
        path = f"feature_preprocessing[{feature!r}]"
        if is_numerical is None:
            token = _REPRESENTATION_PATH.set(path)
            try:
                method = _representation_request(raw).method
            finally:
                _REPRESENTATION_PATH.reset(token)
            is_numerical = self.resolve_categorical(method) not in CATEGORICAL_METHODS
        return self._resolve_representation(raw, is_numerical=is_numerical, path=path)

    def group_representation_for(self, group: FeatureGroup) -> RepresentationConfig:
        """Resolve a group's representation over applicable constructor controls."""
        return self._resolve_representation(
            group.representation,
            is_numerical=True,
            is_group=True,
            path=f"feature_groups[{group.name!r}].representation",
        )

    def _resolve_representation(self, raw, *, is_numerical: bool, path: str, is_group: bool = False):
        requested = _normalize_representation_config(raw, is_numerical=is_numerical, path=path, is_group=is_group)
        spec = TRANSFORMER_REGISTRY[requested.method]
        inherited = {}
        if "output_dim" in spec.allowed_args or spec.supports_adaptive_resolution:
            inherited["output_dim"] = self.output_dim
        if spec.target_usage in ("optional", "required"):
            inherited["target_aware"] = self.target_aware
        if spec.placement_strategies and (spec.target_usage != "forbidden" or not self.target_aware):
            inherited["placement_strategy"] = self.placement_strategy
        if spec.supports_adaptive_resolution:
            adaptive = self.adaptive if requested.adaptive is UNSET else requested.adaptive
            if requested.output_dim is not UNSET and requested.adaptive is UNSET:
                adaptive = False
            inherited["adaptive"] = adaptive
            if adaptive:
                inherited["min_output_dim"] = self.min_output_dim
                inherited["max_output_dim"] = self.max_output_dim
        shared = {"degree": self.degree, "task": self.task, **self.seed_kwargs}
        params = {key: value for key, value in shared.items() if key in spec.allowed_args}
        params.update(requested.params)
        settings = {key: value for key, value in inherited.items() if getattr(requested, key) is UNSET}
        resolved = replace(requested, **settings, params=params)
        return _normalize_representation_config(resolved, is_numerical=is_numerical, path=path, is_group=is_group)

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
