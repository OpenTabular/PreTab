"""Shared parameter names and backward-compatible alias resolution.

Every transformer family uses the same parameter names for the same concepts.
``output_dim`` controls how many output columns are produced per input feature;
``placement_strategy`` controls where basis functions are placed; and so on.
These shared names are listed in :data:`CANONICAL_PARAMS`.

Older code may pass the historic names that existed before the vocabulary was
unified (e.g. ``n_knots``, ``n_bins``, ``n_centers``). :class:`AliasResolverMixin`
lets a transformer accept those old names transparently: the old name still works
but emits a :class:`FutureWarning` so you know to update your code. Using both
the old and new name for the same parameter at the same time raises an error.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from ..exceptions import InvalidParamError


class _Unset:
    """Sentinel for a constructor argument the caller did not supply.

    Using a dedicated singleton rather than ``None`` matters because ``None``
    is a valid value for several parameters (for example, an unbounded adaptive
    dimension). The singleton is identity-comparable, so ``value is UNSET``
    is always unambiguous, and it survives sklearn's ``clone`` safely.
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


#: Placement strategies valid when ``target_aware`` is True (target-aware selectors).
TARGET_AWARE_STRATEGIES: tuple[str, ...] = ("cart", "lightgbm")

#: Placement strategies valid when ``target_aware`` is False (unsupervised spacing).
UNSUPERVISED_STRATEGIES: tuple[str, ...] = ("uniform", "quantile")


def validate_placement(target_aware: bool, placement_strategy: str) -> None:
    """Check that ``target_aware`` and ``placement_strategy`` are compatible.

    Target-aware transformers must use ``"cart"`` or ``"lightgbm"`` as their
    placement strategy; unsupervised transformers must use ``"uniform"`` or
    ``"quantile"``.

    .. note::
        This is enforced at ``fit`` time, not at construction, so you will only
        see the error once you call ``fit()`` or ``fit_transform()``.
    """
    if target_aware and placement_strategy not in TARGET_AWARE_STRATEGIES:
        raise InvalidParamError("When target_aware=True, placement_strategy must be 'cart' or 'lightgbm'.")
    if not target_aware and placement_strategy not in UNSUPERVISED_STRATEGIES:
        raise InvalidParamError("When target_aware=False, placement_strategy must be 'uniform' or 'quantile'.")


#: Shared parameter names with a short description of what each one controls.
CANONICAL_PARAMS: dict[str, str] = {
    "output_dim": "Number of non-bias output columns produced per input feature.",
    "min_output_dim": "Lower bound on the per-feature output dimension in adaptive mode.",
    "max_output_dim": "Upper bound on the per-feature output dimension in adaptive mode.",
    "adaptive": "Whether the per-feature output dimension may vary per feature.",
    "placement_strategy": "How basis units are placed ('cart'/'lightgbm' when target-aware, else 'uniform'/'quantile').",
    "degree": "Polynomial degree of the basis (where meaningful).",
    "target_aware": "Whether the target is used to place basis units.",
    "task": "Supervised task used for target-aware placement.",
}


class AliasResolverMixin:
    """Allow old parameter names to be used alongside the current ones.

    Some transformers used to accept parameter names like ``n_knots`` or
    ``n_bins`` that have since been renamed to ``output_dim``. This mixin lets
    a transformer keep accepting the old names so existing code does not break,
    while nudging users toward the new names via a :class:`FutureWarning`.

    To enable aliases for a transformer, set a class-level ``_param_aliases``
    dict mapping each old name to its current equivalent::

        _param_aliases = {"n_knots": "output_dim"}

    Both names must be real constructor arguments defaulting to :data:`UNSET`.
    This keeps scikit-learn's ``get_params``, ``set_params``, and ``clone``
    working correctly, since they inspect ``__init__`` directly.

    Inside ``fit``, call :meth:`_resolve_param` to get the effective value.
    It returns the current-name value if supplied, falls back to the old name
    with a warning, and raises if both are set at the same time.

    .. warning::
        Setting both the current name and a legacy alias for the same parameter
        raises :class:`~pretab.exceptions.InvalidParamError`. Pick one.
    """

    _param_aliases: ClassVar[dict[str, str]] = {}

    def _aliases_for(self, canonical: str) -> list[str]:
        """Return the legacy alias names that map to ``canonical``."""
        return [alias for alias, target in self._param_aliases.items() if target == canonical]

    def _resolve_param(self, canonical: str, default=UNSET) -> Any:
        """Return the effective value for a parameter, resolving any legacy alias.

        Checks the canonical name first. If not set, looks for a legacy alias
        and returns its value with a deprecation warning. Returns ``default``
        if neither is set.

        Raises
        ------
        InvalidParamError
            If both the canonical name and a legacy alias are set, or if two
            conflicting aliases for the same parameter are both set.
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
                raise InvalidParamError(f"Set {canonical!r} or its legacy alias(es) {names}, not both.")
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
            f"{alias!r} is deprecated and will be removed in a future release; use {canonical!r} instead.",
            FutureWarning,
            stacklevel=3,
        )
        return value
