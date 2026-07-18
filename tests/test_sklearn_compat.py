"""Phase 12: sklearn estimator-contract conformance via ``check_estimator``.

Two tiers:

* **Near-conformant** transformers run the *full* ``check_estimator`` suite. Only
  a small, documented set of checks is allowed to fail (``expected_failed_checks``);
  any *other* failing check is a real regression and fails the test. This is the
  enforced contract for the migrated spline / PLE / feature-map families.

* **Deferred** transformers have deep structural gaps (univariate-only,
  sample-count-changing, or categorical-only) and are tracked as strict
  ``xfail`` so each remaining gap stays a visible, closable result. If one ever
  becomes fully conformant the strict xfail flips to a failure, prompting its
  promotion into the near-conformant tier.

``LanguageEmbeddingTransformer`` is intentionally excluded: it depends on the
optional ``sentence-transformers`` package and would download a model at fit.
"""

import pytest
from sklearn.utils.estimator_checks import check_estimator

from pretab.transformers import (
    BSplineTransformer,
    ContinuousOrdinalTransformer,
    CubicSplineTransformer,
    CustomBinTransformer,
    CyclicalTimeTransformer,
    ISplineTransformer,
    LagFeatureTransformer,
    MSplineTransformer,
    NaturalCubicSplineTransformer,
    NoTransformer,
    OneHotFromOrdinalTransformer,
    PLETransformer,
    PSplineTransformer,
    RBFExpansionTransformer,
    ReLUExpansionTransformer,
    RollingStatsTransformer,
    SigmoidExpansionTransformer,
    TanhExpansionTransformer,
    TensorProductSplineTransformer,
    ThinPlateSplineTransformer,
    ToFloatTransformer,
)

# --- documented reasons for the tolerated check failures ------------------- #

SENTINEL = (
    "Phase 8 alias resolution uses the UNSET sentinel as the default for aliased "
    "parameters (strategy/selector and the output-dim bounds), which "
    "check_parameters_default_constructible rejects. Closing this needs an alias "
    "redesign that drops sentinel defaults."
)
INIT_VALIDATION = (
    "Feature maps validate `strategy`/`task` eagerly in __init__ (deferred Phase-2 "
    "constraint: existing tests assert construction-time errors). sklearn requires "
    "parameter validation to happen in fit, not __init__."
)
REQUIRES_Y_NONE = (
    "Supervised transformer does not yet raise a clear message when y=None is "
    "passed to fit."
)
DTYPE = (
    "Numeric encoder casts to float output and does not accept/preserve object "
    "dtype input."
)

# --- near-conformant tier: (estimator, expected_failed_checks) ------------- #

_SPLINE_EXPECTED = {"check_parameters_default_constructible": SENTINEL}

NEAR_CONFORMANT = [
    (BSplineTransformer(), _SPLINE_EXPECTED),
    (MSplineTransformer(), _SPLINE_EXPECTED),
    (ISplineTransformer(), _SPLINE_EXPECTED),
    (CubicSplineTransformer(), _SPLINE_EXPECTED),
    (NaturalCubicSplineTransformer(), _SPLINE_EXPECTED),
    (PSplineTransformer(), _SPLINE_EXPECTED),
    (TensorProductSplineTransformer(), _SPLINE_EXPECTED),
    (
        PLETransformer(),
        {
            "check_parameters_default_constructible": SENTINEL,
            "check_requires_y_none": REQUIRES_Y_NONE,
            "check_dtype_object": DTYPE,
            "check_transformer_preserve_dtypes": DTYPE,
        },
    ),
    (
        RBFExpansionTransformer(),
        {
            "check_parameters_default_constructible": SENTINEL,
            "check_do_not_raise_errors_in_init_or_set_params": INIT_VALIDATION,
            "check_requires_y_none": REQUIRES_Y_NONE,
        },
    ),
    (
        ReLUExpansionTransformer(),
        {
            "check_parameters_default_constructible": SENTINEL,
            "check_do_not_raise_errors_in_init_or_set_params": INIT_VALIDATION,
            "check_requires_y_none": REQUIRES_Y_NONE,
        },
    ),
    (
        SigmoidExpansionTransformer(),
        {
            "check_parameters_default_constructible": SENTINEL,
            "check_do_not_raise_errors_in_init_or_set_params": INIT_VALIDATION,
            "check_requires_y_none": REQUIRES_Y_NONE,
        },
    ),
    (
        TanhExpansionTransformer(),
        {
            "check_parameters_default_constructible": SENTINEL,
            "check_do_not_raise_errors_in_init_or_set_params": INIT_VALIDATION,
            "check_requires_y_none": REQUIRES_Y_NONE,
        },
    ),
]


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "estimator, expected_failed_checks",
    NEAR_CONFORMANT,
    ids=[type(e).__name__ for e, _ in NEAR_CONFORMANT],
)
def test_check_estimator_near_conformant(estimator, expected_failed_checks):
    """Full check_estimator must pass apart from the documented, tolerated checks."""
    check_estimator(
        estimator,
        expected_failed_checks=expected_failed_checks,
        on_fail="raise",
    )


# --- deferred tier: whole-estimator strict xfail --------------------------- #

DEFERRED = [
    pytest.param(
        ThinPlateSplineTransformer(),
        id="ThinPlateSplineTransformer",
        marks=pytest.mark.xfail(
            reason="Univariate-only (single input feature); incompatible with the "
            "multi-feature transformer checks.",
            strict=True,
        ),
    ),
    pytest.param(
        CustomBinTransformer(),
        id="CustomBinTransformer",
        marks=pytest.mark.xfail(
            reason="Single-column ordinal binner; expects (n_samples, 1) input, "
            "incompatible with generic multi-feature checks.",
            strict=True,
        ),
    ),
    pytest.param(
        CyclicalTimeTransformer(period=12),
        id="CyclicalTimeTransformer",
        marks=pytest.mark.xfail(
            reason="Requires a `period` constructor argument (not default-"
            "constructible) and constrains inputs to [0, period].",
            strict=True,
        ),
    ),
    pytest.param(
        LagFeatureTransformer(),
        id="LagFeatureTransformer",
        marks=pytest.mark.xfail(
            reason="Windowing transformer changes the sample count, so it fails "
            "checks that assume transform preserves n_samples.",
            strict=True,
        ),
    ),
    pytest.param(
        RollingStatsTransformer(),
        id="RollingStatsTransformer",
        marks=pytest.mark.xfail(
            reason="Windowing transformer changes the sample count, so it fails "
            "checks that assume transform preserves n_samples.",
            strict=True,
        ),
    ),
    pytest.param(
        ContinuousOrdinalTransformer(),
        id="ContinuousOrdinalTransformer",
        marks=pytest.mark.xfail(
            reason="Categorical encoder; validates object/string input and does not "
            "use numeric validate_data, so numeric-only checks fail.",
            strict=True,
        ),
    ),
    pytest.param(
        NoTransformer(),
        id="NoTransformer",
        marks=pytest.mark.xfail(
            reason="Identity pass-through with a hardcoded n_features_in_=1; does "
            "not perform 2D feature validation.",
            strict=True,
        ),
    ),
    pytest.param(
        ToFloatTransformer(),
        id="ToFloatTransformer",
        marks=pytest.mark.xfail(
            reason="Per-column float cast with a hardcoded n_features_in_=1; does "
            "not perform 2D feature validation.",
            strict=True,
        ),
    ),
    pytest.param(
        OneHotFromOrdinalTransformer(),
        id="OneHotFromOrdinalTransformer",
        marks=pytest.mark.xfail(
            reason="Categorical one-hot encoder; expects integer-coded input and "
            "does not use numeric validate_data.",
            strict=True,
        ),
    ),
]


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("estimator", DEFERRED)
def test_check_estimator_deferred(estimator):
    """Deferred transformers are not yet fully conformant (tracked as strict xfail)."""
    check_estimator(estimator, on_fail="raise")
