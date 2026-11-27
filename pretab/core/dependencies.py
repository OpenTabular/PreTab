"""Helpers for importing optional third-party dependencies.

Each helper performs the lazy import and, on failure, raises a consistent,
actionable :class:`~pretab.exceptions.OptionalDependencyError` that names the
missing package and the extra that installs it. Centralizing this avoids the
duplicated ``try/except ImportError`` blocks that previously lived inside the
supervised selectors and the language-embedding transformer.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

from ..exceptions import OptionalDependencyError

__all__ = [
    "require_lightgbm",
    "require_module",
    "require_sentence_transformers",
]


def _check_optional_dependency(extra: str, purpose: str) -> None:
    """Check top-level discovery without importing an optional model package."""
    module_name = "sentence_transformers" if extra == "embeddings" else extra
    if importlib.util.find_spec(module_name.partition(".")[0]) is None:
        raise OptionalDependencyError(
            f"{purpose} requires the optional '{module_name}' dependency. Install it with: pip install pretab[{extra}]"
        )


def require_module(module_name: str, extra: str, purpose: str) -> ModuleType:
    """Import ``module_name`` or raise :class:`OptionalDependencyError`.

    Parameters
    ----------
    module_name : str
        The importable module name (e.g. ``"lightgbm"``).
    extra : str
        The PreTab optional extra that installs it (e.g. ``"lightgbm"``), used to
        build the ``pip install pretab[<extra>]`` hint.
    purpose : str
        Human-readable description of what needs the dependency, used to prefix
        the error message (e.g. ``"LightGBM placement"``).
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise OptionalDependencyError(
            f"{purpose} requires the optional '{module_name}' dependency. Install it with: pip install pretab[{extra}]"
        ) from exc


def require_lightgbm(purpose: str = "This feature") -> ModuleType:
    """Import and return ``lightgbm`` or raise a clear optional-dependency error."""
    return require_module("lightgbm", "lightgbm", purpose)


def require_sentence_transformers(purpose: str = "This feature") -> ModuleType:
    """Import and return ``sentence_transformers`` or raise a clear error."""
    return require_module("sentence_transformers", "embeddings", purpose)
