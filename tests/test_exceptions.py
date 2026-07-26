"""Phase 14: typed exception hierarchy and message contracts.

These tests lock two guarantees:

1. Every migrated raise site emits a *typed* ``core.exceptions`` class.
2. The typed classes stay back-compatible: config/data errors remain
   ``ValueError`` subclasses and optional-dependency errors remain
   ``ImportError`` subclasses, so pre-existing ``pytest.raises(ValueError)``
   call sites keep working.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from pretab import Preprocessor, PretabWarning
from pretab.core.adaptive import AdaptiveResolutionMixin
from pretab.core.exceptions import (
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
    insufficient_samples_error,
    invalid_param_error,
)
from pretab.core.knots import generate_internal_knots
from pretab.transformers import (
    BSplineTransformer,
    LagFeatureTransformer,
    PLETransformer,
    RollingStatsTransformer,
    ThinPlateSplineTransformer,
)
from pretab.transformers.splines.knot_selectors import CARTKnotSelector


@pytest.fixture
def xy():
    rng = np.random.RandomState(0)
    X = np.linspace(0, 1, 60).reshape(-1, 1)
    y = rng.rand(60)
    return X, y


# --------------------------------------------------------------------------- #
# Hierarchy: typed classes keep their back-compatible bases.
# --------------------------------------------------------------------------- #


def test_data_and_config_errors_subclass_valueerror():
    assert issubclass(PretabDataError, ValueError)
    assert issubclass(PretabConfigError, ValueError)
    assert issubclass(InvalidParamError, PretabConfigError)
    assert issubclass(IncompatibleParamsError, PretabConfigError)
    assert issubclass(EmptyDataError, PretabDataError)
    assert issubclass(InsufficientSamplesError, PretabDataError)
    assert issubclass(PretabDataError, PretabError)


def test_notfitted_and_optional_dependency_bases():
    assert issubclass(PretabNotFittedError, NotFittedError)
    assert issubclass(PretabNotFittedError, PretabError)
    assert issubclass(OptionalDependencyError, ImportError)
    assert issubclass(OptionalDependencyError, PretabError)


def test_warning_hierarchy():
    assert issubclass(DataWarning, PretabWarning)
    assert issubclass(ConfigWarning, PretabWarning)
    assert issubclass(PretabWarning, UserWarning)


# --------------------------------------------------------------------------- #
# Message factories.
# --------------------------------------------------------------------------- #


def test_invalid_param_error_message_has_value_constraint_and_valid():
    err = invalid_param_error(
        "Foo",
        "strategy",
        "bogus",
        "must be 'uniform' or 'quantile'",
        valid={"uniform", "quantile"},
    )
    assert isinstance(err, InvalidParamError)
    assert isinstance(err, ValueError)
    msg = str(err)
    assert "Foo.strategy" in msg
    assert "'bogus'" in msg  # offending value repr
    assert "must be 'uniform' or 'quantile'" in msg  # constraint
    assert "uniform" in msg and "quantile" in msg  # valid choices listed


def test_insufficient_samples_error_message():
    err = insufficient_samples_error(3, 10, "spline fitting")
    assert isinstance(err, InsufficientSamplesError)
    assert isinstance(err, ValueError)
    msg = str(err)
    assert "3" in msg and "10" in msg and "spline fitting" in msg


# --------------------------------------------------------------------------- #
# Adaptive-resolution bounds route through the typed classes.
# --------------------------------------------------------------------------- #


class _Dummy(AdaptiveResolutionMixin):
    def __init__(self, adaptive):
        self.adaptive = adaptive


def test_resolve_output_bounds_error_types():
    fixed = _Dummy(adaptive=False)
    with pytest.raises(IncompatibleParamsError, match="output_dim must be >= min_output_dim"):
        fixed._resolve_output_bounds(6, 8, None, floor=1)
    with pytest.raises(IncompatibleParamsError, match="output_dim must be <= max_output_dim"):
        fixed._resolve_output_bounds(6, None, 4, floor=1)

    adaptive = _Dummy(adaptive=True)
    with pytest.raises(InvalidParamError, match="min_output_dim must be >= 4"):
        adaptive._resolve_output_bounds(6, 2, 8, floor=4, floor_label="4")
    with pytest.raises(InvalidParamError, match="should be <= 50"):
        adaptive._resolve_output_bounds(6, 5, 99, floor=1, ceil=50)
    with pytest.raises(IncompatibleParamsError, match="min_output_dim must be <= max_output_dim"):
        adaptive._resolve_output_bounds(6, 8, 5, floor=1)


# --------------------------------------------------------------------------- #
# Knot strategy validation.
# --------------------------------------------------------------------------- #


def test_generate_internal_knots_unknown_strategy():
    x = np.linspace(0, 1, 20)
    with pytest.raises(InvalidParamError) as exc:
        generate_internal_knots(x, 5, strategy="bogus")
    assert isinstance(exc.value, ValueError)
    assert "'bogus'" in str(exc.value)


# --------------------------------------------------------------------------- #
# Spline / PLE / thin-plate transformers.
# --------------------------------------------------------------------------- #


def test_bspline_output_dim_too_small(xy):
    X, y = xy
    with pytest.raises(InvalidParamError) as exc:
        BSplineTransformer(output_dim=2, degree=3).fit(X, y)
    assert isinstance(exc.value, ValueError)
    assert "output_dim must be >= degree" in str(exc.value)


def test_bspline_output_dim_too_large(xy):
    X, y = xy
    with pytest.raises(InvalidParamError, match="<= 50"):
        BSplineTransformer(output_dim=99, degree=3).fit(X, y)


def test_bspline_error_is_catchable_as_pretab_error(xy):
    X, y = xy
    with pytest.raises(PretabError):
        BSplineTransformer(output_dim=2, degree=3).fit(X, y)


def test_ple_unsupported_task(xy):
    X, y = xy
    with pytest.raises(InvalidParamError, match="Unsupported task"):
        PLETransformer(output_dim=5, task="bogus").fit(X, y)  # type: ignore[arg-type]


def test_ple_length_mismatch_is_data_error(xy):
    X, y = xy
    with pytest.raises(PretabDataError) as exc:
        PLETransformer(output_dim=5).fit(X, y[:-1])
    assert isinstance(exc.value, ValueError)


def test_ple_all_nan_is_empty_data_error(xy):
    X, y = xy
    X_nan = np.full_like(X, np.nan)
    with pytest.raises(EmptyDataError) as exc:
        PLETransformer(output_dim=5).fit(X_nan, y)
    assert isinstance(exc.value, PretabDataError)


def test_thinplate_multivariate_is_data_error():
    X = np.random.RandomState(1).rand(30, 2)
    with pytest.raises(PretabDataError, match="univariate"):
        ThinPlateSplineTransformer(output_dim=3).fit(X)


# --------------------------------------------------------------------------- #
# Knot selectors.
# --------------------------------------------------------------------------- #


def test_cart_selector_requires_y(xy):
    X, _ = xy
    with pytest.raises(IncompatibleParamsError, match="requires y"):
        CARTKnotSelector().get_knot_locations(X, y=None)


# --------------------------------------------------------------------------- #
# Temporal transformers.
# --------------------------------------------------------------------------- #


def test_lag_insufficient_samples():
    X = np.arange(5).reshape(-1, 1).astype(float)
    with pytest.raises(InsufficientSamplesError) as exc:
        LagFeatureTransformer(n_lags=10).fit(X)
    assert isinstance(exc.value, ValueError)


def test_rolling_unsupported_stat():
    X = np.arange(20).reshape(-1, 1).astype(float)
    transformer = RollingStatsTransformer(window_size=3, stats=("mean", "bogus"))
    transformer.fit(X)
    with pytest.raises(InvalidParamError, match="bogus"):
        transformer.transform(X)


# --------------------------------------------------------------------------- #
# Preprocessor entry point forwards typed config errors.
# --------------------------------------------------------------------------- #


def test_preprocessor_unknown_numerical_method():
    df = pd.DataFrame({"a": np.linspace(0, 1, 40)})
    y = df["a"].to_numpy()
    with pytest.raises(InvalidParamError) as exc:
        Preprocessor(numerical_method="bogus").fit(df, y)
    assert isinstance(exc.value, ValueError)


def test_preprocessor_unknown_categorical_method():
    df = pd.DataFrame({"c": ["a", "b", "a", "c"] * 10})
    y = np.arange(40)
    with pytest.raises(InvalidParamError) as exc:
        Preprocessor(categorical_method="bogus").fit(df, y)
    assert isinstance(exc.value, ValueError)
