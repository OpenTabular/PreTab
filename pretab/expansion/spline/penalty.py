"""Validated penalty matrices in explicit coefficient coordinates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from numbers import Integral, Real
from struct import pack
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...exceptions import InvalidParamError, invalid_param_error


def _invalid(name: str, value: object, constraint: str) -> InvalidParamError:
    return invalid_param_error("PenaltySpec", name, value, constraint)


def _text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid(name, value, "must be a nonempty string")
    return value


def _items(name: str, values: object) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, (Sequence, np.ndarray)):
        raise _invalid(name, values, "must be a nonempty ordered sequence")
    if (isinstance(values, np.ndarray) and values.ndim != 1) or len(values) == 0:
        raise _invalid(name, values, "must be a nonempty one-dimensional sequence")
    return tuple(values)


def _integer(name: str, value: int, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise _invalid(name, value, f"must be an integer >= {minimum}")
    return int(value)


def _tolerance(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or value < 0:
        raise _invalid(name, value, "must be a finite nonnegative real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise _invalid(name, value, "must be a finite nonnegative real number") from exc
    if not isfinite(result):
        raise _invalid(name, value, "must be a finite nonnegative real number")
    return result


def _freeze_provenance(value: Any, active: set[int]) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and isfinite(value):
        return float(value)
    if isinstance(value, (Mapping, list, tuple)):
        identity = id(value)
        if identity in active:
            raise _invalid("provenance", value, "must not contain circular references")
        active.add(identity)
        try:
            if isinstance(value, Mapping):
                if not all(isinstance(key, str) for key in value):
                    raise _invalid("provenance", value, "mapping keys must be strings")
                return MappingProxyType({key: _freeze_provenance(item, active) for key, item in value.items()})
            return tuple(_freeze_provenance(item, active) for item in value)
        finally:
            active.remove(identity)
    raise _invalid("provenance", value, "must contain only finite JSON-compatible values")


def _copy_provenance(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy_provenance(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_copy_provenance(item) for item in value]
    return value


def _validated_matrix(matrix: ArrayLike, atol: float, rtol: float) -> tuple[NDArray[np.float64], int]:
    try:
        raw = np.asarray(matrix)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _invalid("matrix", matrix, "must be a nonempty real square matrix") from exc
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[0] != raw.shape[1] or raw.dtype.kind not in "iuf":
        raise _invalid("matrix", matrix, "must be a nonempty real square matrix")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(values).all():
        raise _invalid("matrix", matrix, "must contain only finite float64 values")

    scale = float(np.max(np.abs(values))) or 1.0
    scaled = values / scale
    symmetric = (scaled + scaled.T) * 0.5
    try:
        eigenvalues = np.linalg.eigvalsh(symmetric)
    except np.linalg.LinAlgError as exc:
        raise _invalid("matrix", matrix, "could not validate its symmetric eigenvalues") from exc
    threshold = atol / scale + rtol * float(np.max(np.abs(eigenvalues)))
    if (threshold == 0 and not np.array_equal(values, values.T)) or np.max(np.abs(scaled - scaled.T)) > threshold:
        raise _invalid("matrix", matrix, "must be symmetric within the recorded tolerance")
    if eigenvalues[0] < -threshold:
        raise _invalid("matrix", matrix, "must be positive semidefinite within the recorded tolerance")
    return values, int(np.count_nonzero(eigenvalues > threshold))


@dataclass(frozen=True, init=False, eq=False, slots=True)
class PenaltySpec:
    """Immutable quadratic penalty and its ordered output coefficient coordinates.

    ``matrix`` is a detached, read-only float64 array. The unscaled penalty is
    ``beta.T @ matrix @ beta`` in the order of ``coefficient_names`` and
    ``coefficient_indices``. ``axis`` names the source directions. ``order`` is
    a positive integer or ``None`` for a convention without a difference order.

    Symmetry, positive semidefiniteness and numerical rank are checked using
    ``atol + rtol * spectral_norm`` of the symmetric part. Validation never
    repairs the stored matrix. ``rank`` and ``nullity`` are derived, and nested
    JSON-compatible ``provenance`` is detached and read-only.
    """

    _matrix_bytes: bytes = field(repr=False)
    kind: str
    order: int | None
    axis: tuple[str, ...]
    coordinate_system: str
    coefficient_names: tuple[str, ...]
    coefficient_indices: tuple[int, ...]
    rank: int
    nullity: int
    atol: float
    rtol: float
    provenance: Mapping[str, Any]

    def __init__(
        self,
        matrix: ArrayLike,
        kind: str,
        order: int | None,
        axis: Sequence[str],
        coordinate_system: str,
        coefficient_names: Sequence[str],
        coefficient_indices: Sequence[int],
        atol: float = 1e-10,
        rtol: float = 1e-10,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        kind = _text("kind", kind)
        if kind == "coefficient_difference" and order is None:
            raise _invalid("order", order, "coefficient_difference requires a positive integer order")
        order = None if order is None else _integer("order", order, 1)
        axes = tuple(_text("axis", label) for label in _items("axis", axis))
        coordinate_system = _text("coordinate_system", coordinate_system)
        names = tuple(_text("coefficient_names", name) for name in _items("coefficient_names", coefficient_names))
        indices = tuple(
            _integer("coefficient_indices", index, 0) for index in _items("coefficient_indices", coefficient_indices)
        )
        atol, rtol = _tolerance("atol", atol), _tolerance("rtol", rtol)
        values, rank = _validated_matrix(matrix, atol, rtol)
        width = values.shape[0]
        for field_name, coordinates in (("coefficient_names", names), ("coefficient_indices", indices)):
            if len(coordinates) != width or len(set(coordinates)) != width:
                raise _invalid(field_name, coordinates, "must contain one distinct entry per matrix row")
        if len(set(axes)) != len(axes):
            raise _invalid("axis", axis, "must contain distinct source labels")
        if provenance is not None and not isinstance(provenance, Mapping):
            raise _invalid("provenance", provenance, "must be a mapping")

        object.__setattr__(self, "_matrix_bytes", values.tobytes())
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "axis", axes)
        object.__setattr__(self, "coordinate_system", coordinate_system)
        object.__setattr__(self, "coefficient_names", names)
        object.__setattr__(self, "coefficient_indices", indices)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "nullity", width - rank)
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "rtol", rtol)
        object.__setattr__(self, "provenance", _freeze_provenance({} if provenance is None else provenance, set()))

    @property
    def matrix(self) -> NDArray[np.float64]:
        """Read-only matrix with independent array metadata on each access."""
        width = len(self.coefficient_indices)
        return np.frombuffer(self._matrix_bytes, dtype=np.float64).reshape(width, width)

    def to_summary(self) -> dict[str, Any]:
        """Return detached JSON metadata without the dense matrix.

        ``matrix_sha256`` is the hexadecimal SHA-256 digest of the row and
        column counts (two unsigned 64-bit little-endian integers), followed
        by C-order little-endian float64 matrix bytes. The checksum treats
        negative zero as positive zero and does not round any other values.
        It identifies exact matrix content, not tolerance-based equivalence.
        The summary cannot reconstruct the matrix.
        """
        matrix = np.array(self.matrix, dtype="<f8", order="C", copy=True)
        matrix[matrix == 0] = 0.0
        digest = sha256(pack("<QQ", *matrix.shape))
        digest.update(memoryview(matrix).cast("B"))
        return {
            "shape": list(matrix.shape),
            "dtype": "float64",
            "kind": self.kind,
            "order": self.order,
            "axis": list(self.axis),
            "coordinate_system": self.coordinate_system,
            "coefficient_names": list(self.coefficient_names),
            "coefficient_indices": list(self.coefficient_indices),
            "rank": self.rank,
            "nullity": self.nullity,
            "atol": self.atol,
            "rtol": self.rtol,
            "provenance": _copy_provenance(self.provenance),
            "matrix_sha256": digest.hexdigest(),
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PenaltySpec):
            return NotImplemented
        return self._metadata() == other._metadata() and np.array_equal(self.matrix, other.matrix)

    def _metadata(self) -> tuple:
        return (
            self.kind,
            self.order,
            self.axis,
            self.coordinate_system,
            self.coefficient_names,
            self.coefficient_indices,
            self.atol,
            self.rtol,
            self.provenance,
        )

    def __reduce__(self):
        return type(self), (self.matrix, *self._metadata()[:-1], _copy_provenance(self.provenance))
