"""Shared building blocks for PreTab transformers.

The ``core`` package holds reusable *math + scikit-learn-contract* primitives
(validation, tags, feature names, logging, typed errors) shared by the concrete,
user-facing transformers. It never defines user-facing transformers itself.
"""

from .base import BasePreTabTransformer
from .exceptions import (
    ConfigWarning,
    DataWarning,
    EmptyDataError,
    IncompatibleParamsError,
    InsufficientSamplesError,
    InvalidParamError,
    OptionalDependencyError,
    PretabConfigError,
    PretabDataError,
    PretabError,
    PretabNotFittedError,
    PretabWarning,
    insufficient_samples_error,
    invalid_param_error,
)
from .logging import get_logger
from .validation import validate_2d_allow_nan

__all__ = [
    "BasePreTabTransformer",
    "ConfigWarning",
    "DataWarning",
    "EmptyDataError",
    "IncompatibleParamsError",
    "InsufficientSamplesError",
    "InvalidParamError",
    "OptionalDependencyError",
    "PretabConfigError",
    "PretabDataError",
    "PretabError",
    "PretabNotFittedError",
    "PretabWarning",
    "get_logger",
    "insufficient_samples_error",
    "invalid_param_error",
    "validate_2d_allow_nan",
]
