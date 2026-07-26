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
    CubicRegressionSplineTransformer,
    ISplineTransformer,
    MSplineTransformer,
    NaturalCubicSplineTransformer,
    NoTransformer,
    NumericBinningTransformer,
    OneHotFromOrdinalTransformer,
    PeriodicEncodingTransformer,
    PLETransformer,
    PSplineTransformer,
    RBFExpansionTransformer,
    ReLUExpansionTransformer,
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
REQUIRES_Y_NONE = "Supervised transformer does not yet raise a clear message when y=None is passed to fit."
DTYPE = "Numeric encoder casts to float output and does not accept/preserve object dtype input."

# --- near-conformant tier: (estimator, expected_failed_checks) ------------- #

_SPLINE_EXPECTED = {"check_parameters_default_constructible": SENTINEL}

NEAR_CONFORMANT = [
    (BSplineTransformer(), _SPLINE_EXPECTED),
    (MSplineTransformer(), _SPLINE_EXPECTED),
    (ISplineTransformer(), _SPLINE_EXPECTED),
    (CubicRegressionSplineTransformer(), _SPLINE_EXPECTED),
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
        },
    ),
    (
        ReLUExpansionTransformer(),
        {
            "check_parameters_default_constructible": SENTINEL,
        },
    ),
    (
        SigmoidExpansionTransformer(),
        {
            "check_parameters_default_constructible": SENTINEL,
        },
    ),
    (
        TanhExpansionTransformer(),
        {
            "check_parameters_default_constructible": SENTINEL,
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
            reason="Landmark low-rank basis needs at least n_components + d + 1 "
            "samples; the generic small-sample estimator checks fall below that "
            "threshold and fail at fit.",
            strict=True,
        ),
    ),
    pytest.param(
        NumericBinningTransformer(),
        id="NumericBinningTransformer",
        marks=pytest.mark.xfail(
            reason="Requires an explicit `output_dim` bin count (not default-"
            "constructible into a fittable state), so the generic estimator "
            "checks fail at fit.",
            strict=True,
        ),
    ),
    pytest.param(
        PeriodicEncodingTransformer(period=12),
        id="PeriodicEncodingTransformer",
        marks=pytest.mark.xfail(
            reason="Requires a `period` constructor argument (not default-"
            "constructible) and constrains inputs to [0, period].",
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
            reason="Identity pass-through that does not perform 2D feature "
            "validation (no validate_data), so numeric multi-feature checks fail.",
            strict=True,
        ),
    ),
    pytest.param(
        ToFloatTransformer(),
        id="ToFloatTransformer",
        marks=pytest.mark.xfail(
            reason="Per-column float cast that does not perform 2D feature "
            "validation (no validate_data), so numeric multi-feature checks fail.",
            strict=True,
        ),
    ),
    pytest.param(
        OneHotFromOrdinalTransformer(),
        id="OneHotFromOrdinalTransformer",
        marks=pytest.mark.xfail(
            reason="Categorical one-hot encoder; expects integer-coded input and does not use numeric validate_data.",
            strict=True,
        ),
    ),
]


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("estimator", DEFERRED)
def test_check_estimator_deferred(estimator):
    """Deferred transformers are not yet fully conformant (tracked as strict xfail)."""
    check_estimator(estimator, on_fail="raise")
