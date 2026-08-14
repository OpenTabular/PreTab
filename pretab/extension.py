"""Public extensibility surface for PreTab representations.

This module is the supported way third parties add, register, discover, and
validate their own representations so they behave like the built-ins:

- :class:`BaseRepresentation` -- the public base class to subclass. It inherits
  the shared scikit-learn contract (NaN-aware validation, estimator tags,
  ``get_feature_names_out``, and a typed :class:`~pretab.RepresentationSpec`) and
  exposes a small declarative surface (``representation_name`` / ``feature_kind``
  / ``scope`` / ``supervision``).
- :func:`register_representation` -- add a class to the capability registry under
  a name so it is selectable via ``Preprocessor(numerical_method=<name>)``.
- :func:`load_entry_point_representations` -- register representations advertised
  by installed packages through the ``pretab.representations`` entry-point group.
- :func:`list_representations` -- query the registry by capability.
- :func:`check_representation` -- a conformance suite that verifies a class obeys
  the representation contract.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from .compose.registry import (
    CATEGORICAL,
    NUMERICAL,
    TRANSFORMER_REGISTRY,
    TransformerSpec,
    register_spec,
)
from .core.base import BasePreTabTransformer
from .core.representation import RepresentationSpec
from .exceptions import ConfigWarning, RepresentationConformanceError

__all__ = [
    "BaseRepresentation",
    "check_representation",
    "list_representations",
    "load_entry_point_representations",
    "register_representation",
]

#: Entry-point group installed packages use to advertise representations.
ENTRY_POINT_GROUP = "pretab.representations"

_SUPERVISION_TO_TARGET_USAGE = {
    "unsupervised": "forbidden",
    "optional": "optional",
    "supervised": "required",
}
_VALID_FEATURE_KINDS = frozenset({NUMERICAL, CATEGORICAL})
_VALID_SCOPES = frozenset({"univariate", "multivariate"})
_VALID_SUPERVISION = frozenset(_SUPERVISION_TO_TARGET_USAGE)


class BaseRepresentation(BasePreTabTransformer):
    """Public base class for third-party PreTab representations.

    Subclass this to add a custom representation that behaves like a built-in: it
    inherits NaN-aware validation, the estimator tags, ``get_feature_names_out``,
    and a typed :class:`~pretab.RepresentationSpec`. Implement ``fit`` /
    ``transform`` and either ``_output_sizes`` (the number of output columns each
    input feature contributes) or ``get_feature_names_out`` directly; then call
    :func:`register_representation` to make it selectable by name.

    Class attributes
    ----------------
    representation_name : str or None
        Canonical registry name -- the value passed as ``numerical_method=`` /
        ``categorical_method=``. Must be set before registration.
    feature_kind : {"numerical", "categorical"}
        The column kind the representation applies to.
    scope : {"univariate", "multivariate"}
        Whether each input feature is expanded independently or several columns
        are modelled jointly.
    supervision : {"unsupervised", "optional", "supervised"}
        How the representation uses the target ``y``. ``"supervised"`` mandates
        ``y`` at fit time; ``"optional"`` consumes it only when ``target_aware``
        is enabled.
    """

    representation_name: str | None = None
    feature_kind: str = NUMERICAL
    scope: str = "univariate"
    supervision: str = "unsupervised"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.feature_kind not in _VALID_FEATURE_KINDS:
            raise ValueError(
                f"{cls.__name__}.feature_kind must be one of {sorted(_VALID_FEATURE_KINDS)}, got {cls.feature_kind!r}"
            )
        if cls.scope not in _VALID_SCOPES:
            raise ValueError(f"{cls.__name__}.scope must be one of {sorted(_VALID_SCOPES)}, got {cls.scope!r}")
        if cls.supervision not in _VALID_SUPERVISION:
            raise ValueError(
                f"{cls.__name__}.supervision must be one of {sorted(_VALID_SUPERVISION)}, got {cls.supervision!r}"
            )
        # Sync the public contract onto the internal representation hooks so the
        # inherited RepresentationSpec and estimator tags reflect the declared
        # metadata without the subclass having to set the private attributes.
        if cls.representation_name is not None:
            cls._representation_family = cls.representation_name
        cls._representation_scope = cls.scope
        cls._representation_supervision = cls.supervision
        cls._requires_y = cls.supervision == "supervised"


def register_representation(
    name,
    cls,
    *,
    feature_kind=None,
    scope=None,
    supervision=None,
    allowed_args=(),
    placement_strategies=(),
    supports_adaptive_resolution=False,
    preprocessor_compatible=True,
    optional_dependency=None,
    periodic=False,
    sparse_output=False,
    override=False,
):
    """Register a representation class under ``name`` so it is selectable by name.

    The capability metadata (``feature_kind`` / ``scope`` / ``supervision``) is
    inferred from the class when it subclasses :class:`BaseRepresentation` and can
    be overridden through the keyword arguments. After registration the method is
    usable as ``Preprocessor(numerical_method=name)`` (or ``categorical_method``)
    and appears in :func:`list_representations`.

    Parameters
    ----------
    name : str
        Canonical method name to register under.
    cls : type
        The scikit-learn-compatible transformer class.
    feature_kind : {"numerical", "categorical"}, optional
        Column kind the method applies to. Inferred from ``cls`` when omitted.
    scope : {"univariate", "multivariate"}, optional
        Inferred from ``cls`` when omitted.
    supervision : {"unsupervised", "optional", "supervised"}, optional
        Inferred from ``cls`` when omitted.
    allowed_args : iterable of str, default=()
        Constructor argument names the shared Preprocessor keyword arguments are
        filtered down to for this method.
    placement_strategies : iterable of str, default=()
        Placement strategies the method honours (empty for methods without
        data-driven placement).
    supports_adaptive_resolution : bool, default=False
        Whether the method can size its output dimension from the data.
    preprocessor_compatible : bool, default=True
        Whether the method can be selected per column through ``Preprocessor``.
        Set False for standalone / multivariate-only methods.
    optional_dependency : str or None, default=None
        Optional extra required for the method to run.
    periodic : bool, default=False
        Whether the representation encodes a periodic signal.
    sparse_output : bool, default=False
        Whether the method can emit a sparse matrix.
    override : bool, default=False
        Whether replacing an already-registered ``name`` is allowed.

    Returns
    -------
    TransformerSpec
        The registered capability record.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    if not isinstance(cls, type):
        raise TypeError(f"cls must be a class, got {type(cls).__name__}")

    feature_kind = feature_kind if feature_kind is not None else getattr(cls, "feature_kind", NUMERICAL)
    scope = scope if scope is not None else getattr(cls, "scope", "univariate")
    supervision = supervision if supervision is not None else getattr(cls, "supervision", "unsupervised")

    if feature_kind not in _VALID_FEATURE_KINDS:
        raise ValueError(f"feature_kind must be one of {sorted(_VALID_FEATURE_KINDS)}, got {feature_kind!r}")
    if scope not in _VALID_SCOPES:
        raise ValueError(f"scope must be one of {sorted(_VALID_SCOPES)}, got {scope!r}")
    if supervision not in _VALID_SUPERVISION:
        raise ValueError(f"supervision must be one of {sorted(_VALID_SUPERVISION)}, got {supervision!r}")

    spec = TransformerSpec(
        name=name,
        transformer_cls=cls,
        allowed_args=tuple(allowed_args),
        feature_kind=frozenset({feature_kind}),
        arity="multivariate" if scope == "multivariate" else "univariate",
        target_usage=_SUPERVISION_TO_TARGET_USAGE[supervision],
        placement_strategies=frozenset(placement_strategies),
        supports_adaptive_resolution=bool(supports_adaptive_resolution),
        preprocessor_compatible=bool(preprocessor_compatible),
        optional_dependency=optional_dependency,
        periodic=bool(periodic),
        sparse_output=bool(sparse_output),
    )
    return register_spec(spec, override=override)


def load_entry_point_representations(group=ENTRY_POINT_GROUP, *, override=False):
    """Register representations advertised by installed packages.

    Iterates the ``group`` entry points (default ``"pretab.representations"``);
    each entry point is expected to load to a representation class. The class is
    registered under its ``representation_name`` attribute (falling back to the
    entry-point name). A broken plugin emits a :class:`ConfigWarning` and is
    skipped rather than breaking discovery for the others.

    This is opt-in (never called automatically at import) so importing ``pretab``
    stays fast and side-effect free.

    Returns
    -------
    list of str
        The names successfully registered, sorted.
    """
    from importlib.metadata import entry_points

    try:
        eps = entry_points(group=group)
    except TypeError:  # pragma: no cover - Python < 3.10 selection fallback
        eps = entry_points().get(group, [])

    loaded = []
    for ep in eps:
        try:
            obj = ep.load()
            reg_name = getattr(obj, "representation_name", None) or ep.name
            register_representation(reg_name, obj, override=override)
            loaded.append(reg_name)
        except Exception as exc:
            warnings.warn(
                f"skipping representation entry point {ep.name!r}: {exc}",
                ConfigWarning,
                stacklevel=2,
            )
    return sorted(loaded)


def list_representations(
    *,
    feature_kind=None,
    scope=None,
    supervised=None,
    periodic=None,
    sparse_output=None,
    adaptive=None,
    include_optional=True,
):
    """Return the registered method names matching every supplied filter.

    All filters are optional and combined with AND. ``None`` means "don't filter
    on this capability".

    Parameters
    ----------
    feature_kind : {"numerical", "categorical"}, optional
        Keep methods that apply to this column kind.
    scope : {"univariate", "multivariate"}, optional
        Keep methods with this arity.
    supervised : bool, optional
        Keep methods that can (``True``) or cannot (``False``) consume ``y``.
    periodic : bool, optional
        Keep methods whose ``periodic`` flag matches.
    sparse_output : bool, optional
        Keep methods whose ``sparse_output`` flag matches.
    adaptive : bool, optional
        Keep methods whose adaptive-resolution support matches.
    include_optional : bool, default=True
        When False, drop methods that need an optional dependency.

    Returns
    -------
    list of str
        Matching canonical method names, sorted.
    """
    result = []
    for spec_name, spec in TRANSFORMER_REGISTRY.items():
        if feature_kind is not None and feature_kind not in spec.feature_kind:
            continue
        if scope is not None and spec.arity != scope:
            continue
        if supervised is not None and spec.is_supervised != bool(supervised):
            continue
        if periodic is not None and spec.periodic != bool(periodic):
            continue
        if sparse_output is not None and spec.sparse_output != bool(sparse_output):
            continue
        if adaptive is not None and spec.supports_adaptive_resolution != bool(adaptive):
            continue
        if not include_optional and spec.optional_dependency is not None:
            continue
        result.append(spec_name)
    return sorted(result)


def _densify(array):
    """Return a dense 2D ndarray view of a (possibly sparse) transform output."""
    if hasattr(array, "toarray"):
        return array.toarray()
    return np.asarray(array)


def check_representation(cls, *, X=None, y=None):
    """Run the representation conformance suite on a class.

    Verifies the contract a well-behaved representation must obey: constructible
    with defaults; ``transform`` before ``fit`` raises ``NotFittedError``; ``fit``
    returns ``self`` and does not mutate its input; ``transform`` yields a 2D
    array with one row per sample; ``get_feature_names_out`` matches the output
    width and is unique; the result is deterministic across ``clone`` + refit; the
    typed :class:`~pretab.RepresentationSpec` agrees with the declared ``scope``
    and output width; and a ``"supervised"`` class refuses to fit without ``y``.

    Parameters
    ----------
    cls : type
        The representation class to validate.
    X : array-like, optional
        Sample input used for the checks. Defaults to a small numeric matrix.
    y : array-like, optional
        Sample target. Generated automatically for supervised classes when the
        class declares ``supervision="supervised"``.

    Returns
    -------
    list of str
        The names of the checks that passed.

    Raises
    ------
    RepresentationConformanceError
        On the first failed check, with a message identifying the violation.
    """
    rng = np.random.RandomState(0)
    if X is None:
        X = rng.uniform(-2.0, 2.0, size=(40, 1)).astype(float)
    X = np.asarray(X)
    n_samples = X.shape[0]

    supervision = getattr(cls, "supervision", "unsupervised")
    needs_y = supervision == "supervised"
    if needs_y and y is None:
        y = rng.uniform(size=n_samples)

    def _make():
        try:
            return cls()
        except TypeError as exc:
            raise RepresentationConformanceError(
                f"{cls.__name__} must be constructible with no required arguments: {exc}"
            ) from exc

    def _fit(est):
        return est.fit(X, y) if needs_y else est.fit(X)

    passed = []

    # 1. transform before fit must raise NotFittedError.
    est = _make()
    try:
        est.transform(X)
    except NotFittedError:
        pass
    except Exception as exc:
        raise RepresentationConformanceError(
            f"{cls.__name__}.transform before fit should raise NotFittedError, got {type(exc).__name__}"
        ) from exc
    else:
        raise RepresentationConformanceError(f"{cls.__name__}.transform before fit should raise NotFittedError")
    passed.append("unfitted_transform_raises")

    # 2. fit returns self and does not mutate X.
    est = _make()
    X_before = X.copy()
    fitted = _fit(est)
    if fitted is not est:
        raise RepresentationConformanceError(f"{cls.__name__}.fit must return self")
    if not np.array_equal(X, X_before, equal_nan=True):
        raise RepresentationConformanceError(f"{cls.__name__}.fit must not mutate its input X")
    passed.append("fit_returns_self_no_mutation")

    # 3. transform is a 2D array with one row per sample.
    out = _densify(fitted.transform(X))
    if out.ndim != 2 or out.shape[0] != n_samples:
        raise RepresentationConformanceError(
            f"{cls.__name__}.transform must return a 2D array with {n_samples} rows, "
            f"got shape {getattr(out, 'shape', None)}"
        )
    width = out.shape[1]
    passed.append("transform_shape")

    # 4. feature names match the output width and are unique.
    names = [str(name) for name in fitted.get_feature_names_out()]
    if len(names) != width:
        raise RepresentationConformanceError(
            f"{cls.__name__}.get_feature_names_out length {len(names)} != output width {width}"
        )
    if len(set(names)) != len(names):
        raise RepresentationConformanceError(f"{cls.__name__}.get_feature_names_out must be unique")
    passed.append("feature_names_match")

    # 5. deterministic across clone + refit.
    clone_out = _densify(_fit(clone(fitted)).transform(X))
    if clone_out.shape != out.shape or not np.allclose(clone_out, out, equal_nan=True):
        raise RepresentationConformanceError(f"{cls.__name__} is not deterministic across clone + refit")
    passed.append("deterministic")

    # 6. typed representation spec agrees with the declared metadata.
    spec = fitted.get_representation_spec()
    if not isinstance(spec, RepresentationSpec):
        raise RepresentationConformanceError(f"{cls.__name__}.get_representation_spec must return a RepresentationSpec")
    declared_scope = getattr(cls, "scope", "univariate")
    if spec.scope != declared_scope:
        raise RepresentationConformanceError(
            f"{cls.__name__} spec.scope {spec.scope!r} != declared scope {declared_scope!r}"
        )
    if spec.output_dim != width:
        raise RepresentationConformanceError(
            f"{cls.__name__} spec.output_dim {spec.output_dim} != output width {width}"
        )
    passed.append("spec_consistent")

    # 7. a supervised class must refuse to fit without y.
    if needs_y:
        est = _make()
        try:
            est.fit(X)
        except Exception:
            passed.append("supervised_requires_y")
        else:
            raise RepresentationConformanceError(
                f"{cls.__name__} declares supervision='supervised' but fit succeeded without y"
            )

    return passed
