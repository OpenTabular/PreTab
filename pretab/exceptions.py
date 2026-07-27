"""Typed exceptions and warning categories for PreTab.

Internal modules should raise these typed errors -- ideally via the message
factories below -- instead of bare ``ValueError``/``TypeError`` with ad-hoc
strings, so callers (including DeepTab) can catch precise, consistently
formatted errors.

For backward compatibility the data/config errors also inherit ``ValueError`` and
:class:`PretabNotFittedError` inherits scikit-learn's ``NotFittedError``, so
existing ``except ValueError`` and ``check_is_fitted`` call sites keep working
unchanged.
"""

from sklearn.exceptions import NotFittedError

__all__ = [
    "ConfigWarning",
    "DataWarning",
    "EmptyDataError",
    "FrozenRepresentationError",
    "IncompatibleParamsError",
    "InsufficientSamplesError",
    "InvalidParamError",
    "LeakageWarning",
    "OptionalDependencyError",
    "OutputBudgetError",
    "PretabConfigError",
    "PretabDataError",
    "PretabError",
    "PretabNotFittedError",
    "PretabSerializationError",
    "PretabWarning",
    "RepresentationConformanceError",
    "insufficient_samples_error",
    "invalid_param_error",
]


# --- Warning hierarchy ---
class PretabWarning(UserWarning):
    """Base category for every warning emitted by PreTab."""


class DataWarning(PretabWarning):
    """Data-quality issue such as dropped columns or clamped values."""


class ConfigWarning(PretabWarning):
    """Configuration fallback or deprecation notice."""


class LeakageWarning(PretabWarning):
    """Potential target leakage: a supervised transformer fit outside a
    cross-fitting / Pipeline / cross-validation context."""


# --- Error hierarchy ---
class PretabError(Exception):
    """Base class for every error raised by PreTab."""


class PretabDataError(PretabError, ValueError):
    """Input data is invalid or incompatible with the transformer."""


class EmptyDataError(PretabDataError):
    """No usable samples or features remained after validation."""


class InsufficientSamplesError(PretabDataError):
    """Too few samples for the requested transformation."""


class PretabConfigError(PretabError, ValueError):
    """A parameter value or combination is invalid."""


class InvalidParamError(PretabConfigError):
    """A single parameter has an invalid value."""


class IncompatibleParamsError(PretabConfigError):
    """Two or more parameters conflict with each other."""


class PretabNotFittedError(PretabError, NotFittedError):
    """A transformer method was called before ``fit``."""


class OptionalDependencyError(PretabError, ImportError):
    """A required optional dependency is not installed."""


class OutputBudgetError(PretabError, ValueError):
    """The fitted preprocessor would exceed a configured output budget
    (``max_output_features`` / ``max_features_per_input`` / ``max_dense_memory``)."""


class PretabSerializationError(PretabError, ValueError):
    """A representation spec could not be serialized or reconstructed
    (unsupported value, unknown class, or incompatible ``schema_version``)."""


class FrozenRepresentationError(PretabError):
    """A mutating operation (e.g. ``set_params``) was attempted on a frozen
    preprocessor. Call :meth:`~pretab.preprocessor.Preprocessor.clone_unfitted`
    to obtain a fresh, mutable copy."""


class RepresentationConformanceError(PretabError, AssertionError):
    """A representation class failed a :func:`pretab.check_representation` check.

    Inherits ``AssertionError`` so a failed conformance check also surfaces
    naturally when :func:`~pretab.check_representation` is called from a test."""


# --- Message factories ---
def invalid_param_error(estimator, param, value, constraint, valid=None):
    """Build an :class:`InvalidParamError` with a consistent, actionable message.

    Parameters
    ----------
    estimator : str
        Name of the estimator (e.g. ``type(self).__name__``).
    param : str
        Offending parameter name.
    value : object
        The invalid value that was supplied.
    constraint : str
        Human-readable description of the constraint that was violated.
    valid : iterable, optional
        The set of valid values, listed in the message when provided.
    """
    msg = f"{estimator}.{param} = {value!r} is invalid.\nConstraint: {constraint}"
    if valid is not None:
        msg += f"\nValid values: {sorted(valid)}"
    return InvalidParamError(msg)


def insufficient_samples_error(n_rows, min_required, reason):
    """Build an :class:`InsufficientSamplesError` with a consistent message."""
    return InsufficientSamplesError(f"Got {n_rows} row(s) but at least {min_required} are required for {reason}.")
