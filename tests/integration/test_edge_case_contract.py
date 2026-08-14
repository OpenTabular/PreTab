"""Edge-case contract for every transformer family (roadmap Phase 8, P8.2).

These tests pin down how each numerical representation family reacts to the
degenerate inputs that show up in real tabular data: a constant column, a fully
missing column, partially missing values, out-of-range values at transform time,
non-finite values, duplicate support points, and too few samples. The behaviour
asserted here *is* the public contract -- if a change makes a family behave
differently on one of these inputs, that is an intentional contract change and
this file must be updated alongside it.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from pretab import Preprocessor, RepresentationPolicy
from pretab.exceptions import DataWarning, InsufficientSamplesError, PretabDataError
from pretab.transformers import (
    BSplineTransformer,
    CubicRegressionSplineTransformer,
    ISplineTransformer,
    MSplineTransformer,
    NaturalCubicSplineTransformer,
    NumericBinningTransformer,
    PLETransformer,
    PSplineTransformer,
    RBFExpansionTransformer,
    ReLUExpansionTransformer,
    SigmoidExpansionTransformer,
    TanhExpansionTransformer,
    TensorProductSplineTransformer,
    ThinPlateSplineTransformer,
)

pytestmark = pytest.mark.filterwarnings("ignore::pretab.exceptions.LeakageWarning")


def _factory(name):
    """Build a transformer configured for an unsupervised (y-optional) fit."""
    return {
        "BSpline": lambda: BSplineTransformer(target_aware=False),
        "MSpline": lambda: MSplineTransformer(target_aware=False),
        "ISpline": lambda: ISplineTransformer(target_aware=False),
        "RBF": lambda: RBFExpansionTransformer(target_aware=False),
        "ReLU": lambda: ReLUExpansionTransformer(target_aware=False),
        "Sigmoid": lambda: SigmoidExpansionTransformer(target_aware=False),
        "Tanh": lambda: TanhExpansionTransformer(target_aware=False),
        "NaturalCubic": lambda: NaturalCubicSplineTransformer(target_aware=False),
        "CubicReg": lambda: CubicRegressionSplineTransformer(target_aware=False),
        "PSpline": lambda: PSplineTransformer(),
        "TensorProduct": lambda: TensorProductSplineTransformer(),
        "ThinPlate": lambda: ThinPlateSplineTransformer(),
        "Binning": lambda: NumericBinningTransformer(output_dim=5),
        "PLE": lambda: PLETransformer(output_dim=5),
    }[name]()


# Families that cannot build a basis on a zero-range (constant) column and must
# say so with a typed :class:`PretabDataError`.
CONSTANT_RAISES = [
    "BSpline",
    "MSpline",
    "ISpline",
    "NaturalCubic",
    "CubicReg",
    "PSpline",
    "TensorProduct",
]

# Families that degrade gracefully on a constant column (single bin / collapsed
# basis) and still return a finite design matrix.
CONSTANT_GRACEFUL = [
    "RBF",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "ThinPlate",
    "Binning",
    "PLE",
]

ALL_FAMILIES = CONSTANT_RAISES + CONSTANT_GRACEFUL

# Families that let missing values pass through the basis (NaN in -> NaN row out).
# The B/M/I splines instead clip a missing value to the fitted boundary, so they
# are intentionally excluded here.
NAN_PROPAGATING = [
    "RBF",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "NaturalCubic",
    "CubicReg",
    "PSpline",
    "TensorProduct",
]


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# Constant column
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", CONSTANT_RAISES)
def test_constant_column_raises_typed_error(name, rng):
    X = np.full((40, 1), 3.14)
    y = rng.normal(size=40)
    with pytest.raises(PretabDataError, match="constant"):
        _factory(name).fit(X, y)


@pytest.mark.parametrize("name", CONSTANT_GRACEFUL)
def test_constant_column_degrades_gracefully(name, rng):
    X = np.full((40, 1), 3.14)
    y = rng.normal(size=40)
    transformer = _factory(name)
    out = transformer.fit(X, y).transform(X)
    assert out.shape[0] == 40
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Fully-missing column
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_all_missing_column_is_rejected(name, rng):
    X = np.full((40, 1), np.nan)
    y = rng.normal(size=40)
    # PretabDataError (typed) for the families that own their validation, plain
    # ValueError for the ones that delegate to scikit-learn -- both are ValueError.
    with pytest.raises(ValueError):
        _factory(name).fit(X, y)


# --------------------------------------------------------------------------- #
# Partially-missing column: missing values propagate, they do not poison knots
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", NAN_PROPAGATING)
def test_partial_missing_propagates_only_on_missing_rows(name, rng):
    X = rng.normal(size=(40, 1))
    X[7, 0] = np.nan
    y = rng.normal(size=40)
    transformer = _factory(name).fit(X, y)
    out = transformer.transform(X)
    missing_rows = np.isnan(out).any(axis=1)
    # Exactly the one missing input row is missing in the output; the fitted
    # support points stay finite (the NaN never reached min/max/quantiles).
    assert missing_rows.sum() == 1
    assert missing_rows[7]


# --------------------------------------------------------------------------- #
# Non-finite (infinity) input
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_infinite_values_are_rejected(name, rng):
    X = rng.normal(size=(40, 1))
    X[0, 0] = np.inf
    y = rng.normal(size=40)
    with pytest.raises(ValueError):
        _factory(name).fit(X, y)


# --------------------------------------------------------------------------- #
# Out-of-range values at transform time stay finite (clip / clamp / evaluate)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_out_of_range_transform_stays_finite(name, rng):
    X = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
    y = rng.normal(size=40)
    transformer = _factory(name).fit(X, y)
    out_of_range = np.array([[-5.0], [5.0]])
    out = transformer.transform(out_of_range)
    assert out.shape[0] == 2
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Duplicate support points (few distinct values) do not crash
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_duplicate_support_points_do_not_crash(name, rng):
    # 40 rows but only five distinct values -> many duplicate knot / center / edge
    # candidates that must be de-duplicated instead of raising.
    X = np.repeat(np.linspace(0.0, 1.0, 5), 8).reshape(-1, 1)
    y = rng.normal(size=40)
    out = _factory(name).fit(X, y).transform(X)
    assert out.shape[0] == 40
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Too few samples
# --------------------------------------------------------------------------- #
def test_binning_rejects_too_few_samples(rng):
    X = rng.normal(size=(2, 1))
    with pytest.raises(InsufficientSamplesError):
        NumericBinningTransformer(output_dim=5).fit(X)


# --------------------------------------------------------------------------- #
# Feature-count mismatch between fit and transform
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["BSpline", "RBF", "Binning"])
def test_feature_count_mismatch_raises(name, rng):
    X = rng.normal(size=(40, 2))
    y = rng.normal(size=40)
    transformer = _factory(name).fit(X, y)
    with pytest.raises((ValueError, PretabDataError)):
        transformer.transform(rng.normal(size=(40, 3)))


# --------------------------------------------------------------------------- #
# Central RepresentationPolicy wiring on the Preprocessor
# --------------------------------------------------------------------------- #
def _frame_with_constant(rng):
    return pd.DataFrame(
        {
            "a": rng.normal(size=60),
            "const": np.full(60, 2.0),
            "c": rng.normal(size=60),
        }
    )


def test_preprocessor_default_policy_allows_constant(rng):
    df = _frame_with_constant(rng)
    y = rng.normal(size=60)
    # Default policy reproduces the historical behaviour: a constant column is fine.
    Preprocessor(numerical_method="standardization").fit(df, y)


def test_preprocessor_policy_errors_on_constant(rng):
    df = _frame_with_constant(rng)
    y = rng.normal(size=60)
    with pytest.raises(PretabDataError):
        Preprocessor(numerical_method="standardization", policy={"constant": "error"}).fit(df, y)


def test_preprocessor_policy_warns_on_constant(rng):
    df = _frame_with_constant(rng)
    y = rng.normal(size=60)
    with pytest.warns(DataWarning):
        Preprocessor(numerical_method="standardization", policy={"constant": "warn"}).fit(df, y)


def test_preprocessor_stores_resolved_policy(rng):
    df = _frame_with_constant(rng)
    y = rng.normal(size=60)
    pre = Preprocessor(policy={"constant": "warn"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pre.fit(df, y)
    assert isinstance(pre.policy_, RepresentationPolicy)
    assert pre.policy_.constant == "warn"
