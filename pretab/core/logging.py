"""Logging foundation for PreTab.

PreTab follows the library convention of never configuring logging on import: the
package logger only carries a :class:`~logging.NullHandler`, so an embedding
application (such as DeepTab) keeps full control of handlers and levels. Use
:func:`get_logger` inside the package to obtain a child logger.
"""

import logging

from ..exceptions import PretabWarning  # re-exported for convenience

__all__ = [
    "PretabWarning",
    "configure_logging",
    "get_logger",
    "set_verbosity",
]

_LOGGER = logging.getLogger("pretab")
_LOGGER.addHandler(logging.NullHandler())

# Integer verbosity -> logging level. Levels are cumulative; 2 and 3 both map to
# DEBUG, with the extra level-3 detail gated on the ``verbose`` value itself.
_LEVELS = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG, 3: logging.DEBUG}


def get_logger(name: str = "pretab") -> logging.Logger:
    """Return a PreTab logger.

    Call ``get_logger(__name__)`` from a module to obtain a child of the
    ``"pretab"`` logger, which inherits the package-level configuration.
    """
    return logging.getLogger(name)


def _has_real_handler(logger: logging.Logger) -> bool:
    """Whether ``logger`` already carries a handler other than ``NullHandler``."""
    return any(not isinstance(handler, logging.NullHandler) for handler in logger.handlers)


def set_verbosity(level: int = 1) -> None:
    """Set the ``"pretab"`` logger level from an integer verbosity.

    Parameters
    ----------
    level : int, default=1
        Verbosity level ``0``-``3`` (``0`` = WARNING, ``1`` = INFO, ``2``/``3`` =
        DEBUG). ``bool`` is accepted and coerced (``True`` -> ``1``, ``False`` ->
        ``0``).
    """
    _LOGGER.setLevel(_LEVELS.get(int(level), logging.INFO))


def configure_logging(level: int = 1, handler: "logging.Handler | None" = None) -> None:
    """Opt-in console logging for standalone use.

    Sets the ``"pretab"`` logger level and attaches a stream handler so PreTab's
    messages become visible. This never touches the root logger, and it is a
    no-op when a real (non-``NullHandler``) handler is already attached -- so an
    embedding host (such as DeepTab) that owns handler/level policy always wins.

    Parameters
    ----------
    level : int, default=1
        Verbosity level ``0``-``3`` (see :func:`set_verbosity`).
    handler : logging.Handler, optional
        Handler to attach. When ``None`` a :class:`logging.StreamHandler` writing
        to stderr with a ``"pretab: <message>"`` format is used.
    """
    if _has_real_handler(_LOGGER):
        return
    set_verbosity(level)
    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    _LOGGER.addHandler(handler)
