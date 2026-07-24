"""Shared building blocks for PreTab transformers.

The ``core`` package holds reusable *math + scikit-learn-contract* primitives
(validation, tags, feature names, logging, typed errors) shared by the concrete,
user-facing transformers. It never defines user-facing transformers itself.
"""

from .adaptive import AdaptiveResolutionMixin
from .base import BasePreTabTransformer
from .centers import center_identification_using_decision_tree
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
from .knots import (
    basis_to_knots,
    generate_internal_knots,
    quantile_knots,
    select_knots,
    spanning_knots,
    uniform_knots,
)
from .locations import resolve_locations, trim_to_count
from .logging import get_logger
from .params import CANONICAL_PARAMS, UNSET, AliasResolverMixin, is_set
from .selectors import (
    BaseLocationSelector,
    CARTLocationSelector,
    LightGBMLocationSelector,
)
from .validation import validate_2d_allow_nan

__all__ = [
    "CANONICAL_PARAMS",
    "UNSET",
    "AdaptiveResolutionMixin",
    "AliasResolverMixin",
    "BaseLocationSelector",
    "BasePreTabTransformer",
    "CARTLocationSelector",
    "ConfigWarning",
    "DataWarning",
    "EmptyDataError",
    "IncompatibleParamsError",
    "InsufficientSamplesError",
    "InvalidParamError",
    "LightGBMLocationSelector",
    "OptionalDependencyError",
    "PretabConfigError",
    "PretabDataError",
    "PretabError",
    "PretabNotFittedError",
    "PretabWarning",
    "basis_to_knots",
    "center_identification_using_decision_tree",
    "generate_internal_knots",
    "get_logger",
    "insufficient_samples_error",
    "invalid_param_error",
    "is_set",
    "quantile_knots",
    "resolve_locations",
    "select_knots",
    "spanning_knots",
    "trim_to_count",
    "uniform_knots",
    "validate_2d_allow_nan",
]
