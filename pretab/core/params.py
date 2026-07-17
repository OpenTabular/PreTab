"""Canonical parameter vocabulary and legacy-alias resolution.

Different transformer families historically spelled the same concept in
different ways -- the number of basis units per feature alone was named
``n_basis_functions`` / ``n_knots`` / ``n_bins`` / ``n_centers`` / ``bins`` /
``n_basis``. :data:`CANONICAL_PARAMS` records the single cross-family vocabulary
new code and docs should use, and :class:`AliasResolverMixin` lets a transformer
keep accepting its old names as *aliases* that resolve to the canonical name at
``fit`` time (with a :class:`FutureWarning`).

The mechanism follows scikit-learn's deprecation constraint: ``get_params`` and
``clone`` introspect ``__init__`` and re-instantiate with the exact same
argument names, so aliases cannot hide behind ``**kwargs``. Instead the
canonical parameter and every legacy alias are ordinary constructor arguments
that default to :data:`UNSET` and are stored verbatim; the effective value is
resolved inside ``fit``.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from .exceptions import InvalidParamError


class _Unset:
    """Sentinel marking a constructor argument the user did not provide.

    A dedicated singleton (rather than ``None``) is used because ``None`` is a
    meaningful value for several parameters (for example an unset adaptive
    bound). Being a singleton keeps it identity-comparable and clone-safe.
    """

    __slots__ = ()
    _instance: _Unset | None = None

    def __new__(cls) -> _Unset:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET = _Unset()


def is_set(value) -> bool:
    """Return ``True`` when ``value`` is not the :data:`UNSET` sentinel."""
    return value is not UNSET


# §8.3 canonical vocabulary: the family-neutral name for each shared concept.
CANONICAL_PARAMS: dict[str, str] = {
    "n_basis": "Number of basis units (knots / bins / centers) produced per feature.",
    "min_basis": "Lower bound on the number of basis units in adaptive mode.",
    "max_basis": "Upper bound on the number of basis units in adaptive mode.",
    "adaptive": "Whether the number of basis units may vary per feature.",
    "strategy": "Placement strategy for basis units ('uniform' or 'quantile').",
    "degree": "Polynomial degree of the basis (where meaningful).",
    "selector": "Target-aware selector used to place basis units.",
    "use_target": "Whether to use the target to place basis units.",
    "task": "Supervised task used for target-aware placement.",
}


class AliasResolverMixin:
    """Accept legacy parameter names as aliases of the canonical vocabulary.

    Subclasses declare a class-level ``_param_aliases`` mapping each legacy
    constructor argument to its canonical name, e.g.::

        _param_aliases = {"n_knots": "n_basis", "knot_strategy": "strategy"}

    Both the canonical parameter and every legacy alias are ordinary
    constructor parameters that default to :data:`UNSET` and are stored
    verbatim (so ``get_params`` / ``clone`` / ``set_params`` stay
    sklearn-correct). Inside ``fit`` call :meth:`_resolve_param` to obtain the
    effective value: an explicit canonical value wins, an explicit legacy alias
    is honoured with a ``FutureWarning``, and setting both a canonical and one
    of its aliases -- or two conflicting aliases -- raises
    :class:`~pretab.core.exceptions.InvalidParamError`.
    """

    _param_aliases: ClassVar[dict[str, str]] = {}

    def _aliases_for(self, canonical: str) -> list[str]:
        """Return the legacy alias names that map to ``canonical``."""
        return [alias for alias, target in self._param_aliases.items() if target == canonical]

    def _resolve_param(self, canonical: str, default=UNSET) -> Any:
        """Return the effective value for a canonical parameter.

        Resolution order: an explicitly set canonical value, otherwise an
        explicitly set legacy alias (emitting a ``FutureWarning``), otherwise
        ``default``.

        Raises
        ------
        InvalidParamError
            If both the canonical parameter and a legacy alias are set, or if
            two conflicting legacy aliases for the same canonical are set.
        """
        canon_val = getattr(self, canonical, UNSET)
        set_aliases = [
            (alias, getattr(self, alias, UNSET))
            for alias in self._aliases_for(canonical)
            if is_set(getattr(self, alias, UNSET))
        ]

        if is_set(canon_val):
            if set_aliases:
                names = ", ".join(repr(alias) for alias, _ in set_aliases)
                raise InvalidParamError(
                    f"Set {canonical!r} or its legacy alias(es) {names}, not both."
                )
            return canon_val

        if not set_aliases:
            return default

        if len(set_aliases) > 1:
            names = ", ".join(repr(alias) for alias, _ in set_aliases)
            raise InvalidParamError(
                f"Conflicting legacy aliases for {canonical!r}: {names}. Use {canonical!r} instead."
            )

        alias, value = set_aliases[0]
        warnings.warn(
            f"{alias!r} is deprecated and will be removed in a future release; "
            f"use {canonical!r} instead.",
            FutureWarning,
            stacklevel=3,
        )
        return value
