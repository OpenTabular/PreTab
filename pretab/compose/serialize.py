"""Portable, versioned JSON (de)serialization for the fitted preprocessor.

:func:`preprocessor_to_spec` / :func:`preprocessor_from_spec` capture a fitted
:class:`~pretab.preprocessor.Preprocessor` as a self-describing JSON document --
schema-versioned, dependency-versioned, and human-inspectable -- that
reconstructs the estimator bit-for-bit. It is an explicit, auditable alternative
to :mod:`pickle`: loading a spec only ever imports an allow-listed set of library
namespaces and never runs estimator ``__init__`` / ``__reduce__`` /
``__setstate__`` code, so a spec cannot execute arbitrary code the way
``pickle.load`` can.

The document has a small declarative envelope (schema/library versions, resolved
constructor params, per-representation summary, output column order) plus a
``state`` payload that JSON-encodes the fitted object graph (numpy arrays, knot /
center / bin locations, scalers, nested estimators) for exact reconstruction.
"""

import dataclasses
import importlib
from typing import Any, cast

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .._version import __version__ as _PRETAB_VERSION
from ..core.parameters import UNSET
from ..core.policy import RepresentationPolicy
from ..core.representation import FeatureLineage, RepresentationSpec
from ..exceptions import PretabError, PretabSerializationError
from ..placement.base import PlacementResult
from .registry import TransformerSpec

SCHEMA_VERSION = 1

# Top-level packages that a spec is allowed to *import a module from*. This is
# only the first line of defense (see ``_ALLOWED_DATACLASSES`` and the
# ``BaseEstimator`` check below for the checks that actually gate what gets
# called/instantiated). ``builtins`` is deliberately excluded: nothing PreTab
# ever needs to reconstruct from a spec lives there, and allowing it is what let
# a crafted spec call ``builtins.open`` with attacker-controlled arguments.
_ALLOWED_TOP_LEVEL = frozenset({"pretab", "sklearn", "numpy", "scipy"})

# The exact, closed set of dataclasses a spec is allowed to reconstruct via the
# ``__dataclass__`` tag. Reconstruction calls ``cls(**fields)`` (i.e. runs the
# dataclass's ``__init__``), so this must be an exact allow-list, not merely
# "any dataclass importable under an allowed module" -- the latter would still
# let a spec construct an arbitrary sklearn/numpy/scipy dataclass with
# attacker-controlled fields.
_ALLOWED_DATACLASSES = frozenset(
    {RepresentationPolicy, RepresentationSpec, FeatureLineage, PlacementResult, TransformerSpec}
)


# --- helpers -------------------------------------------------------------
def _qualname(obj) -> str:
    cls = type(obj)
    return f"{cls.__module__}:{cls.__qualname__}"


def _module_allowed(module: str) -> bool:
    return module.partition(".")[0] in _ALLOWED_TOP_LEVEL


def _resolve(dotted: str):
    """Import and return the class/type named ``module:qualname`` from the allow-list."""
    module, _, qualname = dotted.partition(":")
    if not _module_allowed(module):
        raise PretabSerializationError(
            f"Refusing to import disallowed module {module!r} while loading a spec. "
            f"Only {sorted(_ALLOWED_TOP_LEVEL)} are permitted."
        )
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


# --- encoding ------------------------------------------------------------
def _encode(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if obj is UNSET:
        return {"__unset__": True}
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": {"dtype": obj.dtype.str, "shape": list(obj.shape), "data": obj.tolist()}}
    if isinstance(obj, np.dtype):
        return {"__npdtype__": obj.str}
    if isinstance(obj, type):
        return {"__type__": f"{obj.__module__}:{obj.__qualname__}"}
    if isinstance(obj, slice):
        return {"__slice__": [obj.start, obj.stop, obj.step]}
    if isinstance(obj, tuple):
        return {"__tuple__": [_encode(v) for v in obj]}
    if isinstance(obj, list):
        return [_encode(v) for v in obj]
    if isinstance(obj, BaseEstimator):
        return {"__estimator__": {"class": _qualname(obj), "state": _encode_mapping(vars(obj))}}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = {f.name: _encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        return {"__dataclass__": {"class": _qualname(obj), "fields": fields}}
    if isinstance(obj, dict):
        return {"__dict__": [[_encode(k), _encode(v)] for k, v in obj.items()]}
    raise PretabSerializationError(f"Cannot serialize value of type {type(obj).__module__}.{type(obj).__name__!r}.")


def _encode_mapping(mapping: dict) -> dict:
    """Encode a ``str``-keyed attribute mapping (an estimator/object ``__dict__``)."""
    return {str(k): _encode(v) for k, v in mapping.items()}


def _json_safe(obj):
    """Best-effort readable encoding used for the declarative ``params`` block.

    Keeps JSON-native values verbatim and falls back to the tagged :func:`_encode`
    form only for exotic values. The result is informational and never decoded.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return _encode(obj)


# --- decoding ------------------------------------------------------------
def _decode_ndarray(payload: dict) -> np.ndarray:
    dtype = np.dtype(payload["dtype"])
    arr = np.array(payload["data"], dtype=dtype)
    return arr.reshape(payload["shape"])


def _decode(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return _decode_ndarray(obj["__ndarray__"])
        if "__unset__" in obj:
            return UNSET
        if "__npdtype__" in obj:
            return np.dtype(obj["__npdtype__"])
        if "__type__" in obj:
            return _resolve(obj["__type__"])
        if "__slice__" in obj:
            start, stop, step = obj["__slice__"]
            return slice(start, stop, step)
        if "__tuple__" in obj:
            return tuple(_decode(v) for v in obj["__tuple__"])
        if "__dict__" in obj:
            return {_decode(k): _decode(v) for k, v in obj["__dict__"]}
        if "__estimator__" in obj:
            return _decode_estimator(obj["__estimator__"])
        if "__dataclass__" in obj:
            return _decode_dataclass(obj["__dataclass__"])
        raise PretabSerializationError(f"Unrecognized encoded object with keys {sorted(obj)}.")
    raise PretabSerializationError(f"Cannot decode value of type {type(obj).__name__!r}.")


def _decode_mapping(mapping: dict) -> dict:
    return {k: _decode(v) for k, v in mapping.items()}


def _decode_estimator(payload: dict):
    """Reconstruct a ``BaseEstimator`` from an ``__estimator__`` tag.

    Bypasses ``__init__``/``__reduce__``/``__setstate__`` by design (a plain
    ``__new__`` plus a ``__dict__`` update), but only for a class that is
    actually a registered scikit-learn estimator -- anything else (a crafted
    ``\"class\"`` naming an unrelated callable) is refused before it is ever
    instantiated.
    """
    cls = cast(Any, _resolve(payload["class"]))
    if not (isinstance(cls, type) and issubclass(cls, BaseEstimator)):
        raise PretabSerializationError(
            f"Refusing to reconstruct {payload['class']!r} as an estimator: not a scikit-learn BaseEstimator subclass."
        )
    obj = cls.__new__(cls)
    obj.__dict__.update(_decode_mapping(payload["state"]))
    return obj


def _decode_dataclass(payload: dict):
    """Reconstruct a dataclass from a ``__dataclass__`` tag.

    Unlike :func:`_decode_estimator`, this calls ``cls(**fields)`` (the
    dataclass's real ``__init__``), so the allow-list must be an *exact* set of
    approved classes, not merely "any dataclass reachable under an allowed
    module" -- refusing anything outside ``_ALLOWED_DATACLASSES`` is what stops
    a crafted spec from constructing an arbitrary callable with
    attacker-controlled keyword arguments.
    """
    cls = cast(Any, _resolve(payload["class"]))
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls) and cls in _ALLOWED_DATACLASSES):
        allowed = sorted(c.__qualname__ for c in _ALLOWED_DATACLASSES)
        raise PretabSerializationError(
            f"Refusing to reconstruct disallowed dataclass {payload['class']!r}. Only {allowed} are permitted."
        )
    fields = {k: _decode(v) for k, v in payload["fields"].items()}
    return cls(**fields)


# --- envelope ------------------------------------------------------------
def _library_versions() -> dict:
    import scipy
    import sklearn

    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _representation_summary(preprocessor) -> list:
    """Best-effort declarative per-representation summary (family/columns/locations)."""
    summary: list = []
    column_transformer = getattr(preprocessor, "column_transformer_", None)
    if column_transformer is None:
        return summary
    for name, transformer, columns in column_transformer.transformers_:
        if name == "remainder":
            continue
        leaf = transformer.steps[-1][1] if hasattr(transformer, "steps") else transformer
        spec_fn = getattr(leaf, "get_representation_spec", None)
        if spec_fn is None:
            continue
        try:
            entry = spec_fn().to_dict()
        except (PretabError, ValueError, AttributeError, TypeError, KeyError):
            continue
        entry["columns"] = [str(col) for col in columns]
        summary.append(entry)
    return summary


def preprocessor_to_spec(preprocessor) -> dict:
    """Serialize a fitted preprocessor into a portable, versioned spec dictionary."""
    check_is_fitted(preprocessor)
    return {
        "schema_version": SCHEMA_VERSION,
        "pretab_version": _PRETAB_VERSION,
        "library_versions": _library_versions(),
        "params": _json_safe(preprocessor.get_params(deep=False)),
        "feature_names_out": [str(name) for name in preprocessor.get_feature_names_out()],
        "representations": _representation_summary(preprocessor),
        "state": _encode_mapping(vars(preprocessor)),
    }


def check_spec_schema(data: dict) -> None:
    """Validate the envelope's schema version before reconstruction."""
    if not isinstance(data, dict) or "schema_version" not in data:
        raise PretabSerializationError("Not a PreTab spec: missing 'schema_version'.")
    version = data["schema_version"]
    if version != SCHEMA_VERSION:
        raise PretabSerializationError(
            f"Unsupported spec schema_version {version!r}; this build of PreTab supports {SCHEMA_VERSION}."
        )


def preprocessor_from_spec(data: dict):
    """Reconstruct a fitted preprocessor from a spec produced by :func:`preprocessor_to_spec`."""
    check_spec_schema(data)
    from ..preprocessor import Preprocessor

    obj = Preprocessor.__new__(Preprocessor)
    obj.__dict__.update(_decode_mapping(data["state"]))
    return obj
