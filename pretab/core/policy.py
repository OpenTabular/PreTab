"""Central, explicit edge-case policy for representations.

:class:`RepresentationPolicy` names, in one place, how every transformer reacts
to the recurring edge cases that would otherwise diverge silently per family:

* ``constant``: zero-variance input columns (``"error"`` / ``"warn"`` / ``"allow"``).
* ``out_of_range``: values at ``transform`` outside the fitted range
  (``"error"`` / ``"warn"`` / ``"clip"`` / ``"extrapolate"``).

The defaults reproduce the library's historical behaviour (constant columns pass
through, ranges extrapolate), so enabling the policy object changes nothing
until a stricter choice is requested. Transformers may narrow specific axes
through class-level override attributes without exposing a new constructor
parameter (see :class:`~pretab.core.base.BasePreTabTransformer`).

Missing values and non-finite (``inf`` / ``-inf``) inputs are handled elsewhere:
see ``Preprocessor.missing_policy`` for missing-value handling, and note that
``inf`` / ``-inf`` always raise during input validation regardless of any
policy (there is no configurable axis for it).

.. note::
   ``out_of_range`` is not yet reachable from any public API: no transformer
   constructor accepts a ``policy`` argument, and ``Preprocessor.policy`` is
   only used for its own top-level ``constant`` check, never threaded into
   ``PreprocessorConfig`` or the transformers it builds. Wiring this through
   (transformer constructors, the registry's ``allowed_args``, and
   ``PreprocessorConfig``) is deferred to a follow-up; see the "Spline
   expansions" section of ``dev/todo/release-1.0.0/bugfixes-1.0.0.md``.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, fields, replace

import numpy as np

from ..exceptions import DataWarning, PretabDataError, invalid_param_error

__all__ = [
    "RepresentationPolicy",
    "apply_constant_policy",
    "find_constant_columns",
    "resolve_out_of_range",
]

_CONSTANT_CHOICES = ("error", "warn", "allow")
_OUT_OF_RANGE_CHOICES = ("error", "warn", "clip", "extrapolate")

_CHOICES = {
    "constant": _CONSTANT_CHOICES,
    "out_of_range": _OUT_OF_RANGE_CHOICES,
}


@dataclass(frozen=True)
class RepresentationPolicy:
    """Declarative edge-case policy shared across transformers.

    Parameters
    ----------
    constant : {"error", "warn", "allow"}, default="allow"
        Reaction to a zero-variance (constant) input column.
    out_of_range : {"error", "warn", "clip", "extrapolate"}, default="extrapolate"
        Reaction to ``transform``-time values outside the fitted range.
    """

    constant: str = "allow"
    out_of_range: str = "extrapolate"

    def __post_init__(self):
        for name, choices in _CHOICES.items():
            value = getattr(self, name)
            if value not in choices:
                raise invalid_param_error("RepresentationPolicy", name, value, f"one of {choices}", valid=choices)

    @classmethod
    def resolve(cls, policy) -> RepresentationPolicy:
        """Coerce ``None`` / a mapping / an instance into a ``RepresentationPolicy``."""
        if policy is None:
            return cls()
        if isinstance(policy, cls):
            return policy
        if isinstance(policy, dict):
            return cls(**policy)
        raise invalid_param_error(
            "RepresentationPolicy",
            "policy",
            policy,
            "None, a dict, or a RepresentationPolicy instance",
        )

    def merge(self, **overrides) -> RepresentationPolicy:
        """Return a copy with the non-``None`` ``overrides`` applied."""
        valid = {f.name for f in fields(self)}
        applied = {}
        for key, value in overrides.items():
            if value is None:
                continue
            if key not in valid:
                raise invalid_param_error(
                    "RepresentationPolicy.merge", "override", key, f"one of {sorted(valid)}", valid=valid
                )
            applied[key] = value
        return replace(self, **applied)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary of the policy fields."""
        return asdict(self)


def find_constant_columns(X) -> list[int]:
    """Return the indices of zero-variance columns in ``X`` (ignoring ``NaN``)."""
    X = np.asarray(X, dtype=np.float64)
    constant = []
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        if finite.size and float(np.ptp(finite)) == 0.0:
            constant.append(j)
    return constant


def apply_constant_policy(X, policy: RepresentationPolicy, *, estimator) -> None:
    """Enforce ``policy.constant`` against the constant columns of ``X``.

    ``"allow"`` is a no-op; ``"warn"`` emits a :class:`~pretab.exceptions.DataWarning`;
    ``"error"`` raises :class:`~pretab.exceptions.PretabDataError`.
    """
    if policy.constant == "allow":
        return
    constant = find_constant_columns(X)
    if not constant:
        return
    name = type(estimator).__name__
    message = f"{name} received constant (zero-variance) column(s) at index {constant}."
    if policy.constant == "error":
        raise PretabDataError(message)
    warnings.warn(message, DataWarning, stacklevel=2)


def resolve_out_of_range(X, lower, upper, policy: RepresentationPolicy, *, estimator):
    """Apply ``policy.out_of_range`` to ``X`` given the fitted ``[lower, upper]`` bounds.

    ``lower`` / ``upper`` are per-column arrays. ``"extrapolate"`` returns ``X``
    unchanged, ``"clip"`` clamps into range, ``"warn"`` / ``"error"`` react when
    any value lies outside the fitted bounds.
    """
    X = np.asarray(X, dtype=np.float64)
    if policy.out_of_range == "extrapolate":
        return X
    lower = np.asarray(lower, dtype=np.float64).ravel()
    upper = np.asarray(upper, dtype=np.float64).ravel()
    with np.errstate(invalid="ignore"):
        below = X < lower
        above = X > upper
    if not (below.any() or above.any()):
        return X
    if policy.out_of_range == "clip":
        return np.clip(X, lower, upper)
    name = type(estimator).__name__
    message = f"{name} received values outside the fitted range at transform time."
    if policy.out_of_range == "error":
        raise PretabDataError(message)
    warnings.warn(message, DataWarning, stacklevel=2)
    return X
