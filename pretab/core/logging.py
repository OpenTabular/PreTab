"""Logging foundation for PreTab.

PreTab follows the library convention of never configuring logging on import: the
package logger only carries a :class:`~logging.NullHandler`, so an embedding
application (such as DeepTab) keeps full control of handlers and levels. Use
:func:`get_logger` inside the package to obtain a child logger.
"""

import logging

from .exceptions import PretabWarning  # re-exported for convenience

__all__ = ["PretabWarning", "get_logger"]

_LOGGER = logging.getLogger("pretab")
_LOGGER.addHandler(logging.NullHandler())


def get_logger(name: str = "pretab") -> logging.Logger:
    """Return a PreTab logger.

    Call ``get_logger(__name__)`` from a module to obtain a child of the
    ``"pretab"`` logger, which inherits the package-level configuration.
    """
    return logging.getLogger(name)
