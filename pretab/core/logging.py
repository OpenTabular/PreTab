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


# Marks the handler :func:`configure_logging` attaches, so later calls can tell
# PreTab's own handler apart from one owned by the host application.
_OWN_HANDLER_FLAG = "_pretab_own_handler"


def _is_host_handler(handler: logging.Handler) -> bool:
    """Whether ``handler`` belongs to the host (neither a ``NullHandler`` nor PreTab's own)."""
    return not isinstance(handler, logging.NullHandler) and not getattr(handler, _OWN_HANDLER_FLAG, False)


def _has_real_handler(logger: logging.Logger) -> bool:
    """Whether ``logger`` itself carries a host handler (one PreTab did not attach)."""
    return any(_is_host_handler(handler) for handler in logger.handlers)


def _ancestor_has_real_handler(logger: logging.Logger) -> bool:
    """Whether a host handler on an ancestor receives ``logger``'s records through propagation."""
    current = logger
    while current.propagate and current.parent is not None:
        current = current.parent
        if _has_real_handler(current):
            return True
    return False


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
    no-op when the host has attached a real (non-``NullHandler``) handler to the
    ``"pretab"`` logger -- so an embedding host (such as DeepTab) that owns
    handler/level policy always wins.

    Repeated calls update the level (the handler PreTab attached earlier is
    reused, never duplicated). When a host handler on an ancestor logger -- such
    as a root handler from :func:`logging.basicConfig` -- already receives
    PreTab's records, no handler is attached (and one attached earlier is
    removed), so every message is printed once.

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
    own_handlers = [existing for existing in _LOGGER.handlers if getattr(existing, _OWN_HANDLER_FLAG, False)]
    if _ancestor_has_real_handler(_LOGGER):
        for existing in own_handlers:
            _LOGGER.removeHandler(existing)
        return
    if own_handlers:
        return
    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    setattr(handler, _OWN_HANDLER_FLAG, True)
    _LOGGER.addHandler(handler)
