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

from pretab import CrossFittedTransformer, Preprocessor, RepresentationPolicy
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
NAN_PROPAGATING = [
    "BSpline",
    "MSpline",
    "ISpline",
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


# --------------------------------------------------------------------------- #
# Column with no observed value at fit, through the Preprocessor
# --------------------------------------------------------------------------- #
def _frame_with_empty_column(rng, kind, n=60):
    """A frame whose ``empty`` column (numerical or categorical) is missing on every row."""
    empty = np.full(n, np.nan) if kind == "numerical" else pd.Series([np.nan] * n, dtype=object)
    return pd.DataFrame({"age": rng.normal(40, 10, n), "city": rng.choice(["a", "b", "c"], n), "empty": empty})


def _observed(X, kind, rng):
    """``X`` with its ``empty`` column observed on every row (as later data may be)."""
    values = rng.normal(size=len(X)) if kind == "numerical" else rng.choice(["p", "q"], len(X))
    return X.assign(empty=values)


def _fit_empty(pre, X, y):
    with pytest.warns(DataWarning, match=r"\['empty'\] with no observed"):
        return pre.fit(X, y)


def _assert_metadata_matches_output(pre, X, X_later):
    """Every detected feature has a block, and every metadata view agrees with the output."""
    out = pre.transform(X, return_array=True)
    width = out.shape[1]
    assert pre.n_features_in_ == 3
    assert set(pre.output_dims_) == {"age", "city", "empty"}
    assert pre.output_dims_["empty"] >= 1
    assert sum(pre.output_dims_.values()) == width == pre.total_output_dim_
    assert len(pre.get_feature_names_out()) == width
    lineage = pre.get_feature_lineage()
    assert len(lineage) == width
    assert sum(record.source_features == ("empty",) for record in lineage) == pre.output_dims_["empty"]
    blocks = pre.transform(X, return_array=False)
    assert {name.split("_", 1)[1] for name in blocks} == {"age", "city", "empty"}
    numerical, categorical, _ = pre.get_feature_info(verbose=False)
    assert set(numerical) | set(categorical) == {"age", "city", "empty"}
    info = {**numerical, **categorical}["empty"]
    if info["dimension"] is not None:
        assert info["dimension"] == pre.output_dims_["empty"]
    # Data where the column is observed later is transformed with the fitted width.
    assert pre.transform(X_later, return_array=True).shape == (len(X_later), width)


# Numerical methods that fit a constant column, with (target_aware, placement_strategy).
_EMPTY_NUMERICAL_CASES = [
    (method, target_aware, "cart" if target_aware else "uniform")
    for method in [
        "ple",
        "minmax",
        "standardization",
        "robust",
        "quantile",
        "yeo-johnson",
        "rbf",
        "relu",
        "sigmoid",
        "tanh",
        "binning",
        "fourier",
        "polynomial",
        "none",
    ]
    for target_aware in (True, False)
    if not (method == "ple" and not target_aware)
]


@pytest.mark.parametrize(("method", "target_aware", "placement"), _EMPTY_NUMERICAL_CASES)
def test_empty_numerical_column_keeps_its_block(method, target_aware, placement, rng):
    X = _frame_with_empty_column(rng, "numerical")
    y = rng.normal(size=len(X))
    pre = Preprocessor(numerical_method=method, target_aware=target_aware, placement_strategy=placement)
    _fit_empty(pre, X, y)
    _assert_metadata_matches_output(pre, X, _observed(X, "numerical", rng))


# The methods that cannot be fitted on a constant column (see CONSTANT_RAISES).
@pytest.mark.parametrize(
    "method", ["bspline", "mspline", "ispline", "pspline", "naturalspline", "cubicspline", "box-cox"]
)
def test_empty_numerical_column_names_the_column_when_the_method_needs_spread(method, rng):
    X = _frame_with_empty_column(rng, "numerical")
    y = rng.normal(size=len(X))
    pre = Preprocessor(numerical_method=method, target_aware=False, placement_strategy="uniform")
    with pytest.warns(DataWarning), pytest.raises(PretabDataError, match=f"Column 'empty' .* {method!r}") as info:
        pre.fit(X, y)
    assert info.value.__cause__ is not None  # chained to the method's own error


@pytest.mark.parametrize(
    ("method", "y_kind", "message"),
    [
        ("ple", "short", "same length"),
        ("bspline", "strings", "could not convert string to float"),
        ("rbf", "strings", "could not convert string to float"),
    ],
)
def test_empty_column_is_not_blamed_for_an_unrelated_fit_failure(method, y_kind, message, rng):
    """A failure the block also hits with observed values (here a bad ``y``) keeps
    its own error instead of being attributed to the empty column."""
    X = _frame_with_empty_column(rng, "numerical")
    y = rng.normal(size=len(X) - 3) if y_kind == "short" else rng.choice(["p", "q"], len(X))
    params = {"target_aware": True, "placement_strategy": "cart"} if method != "ple" else {}
    with pytest.warns(DataWarning), pytest.raises(ValueError, match=message) as info:
        Preprocessor(numerical_method=method, **params).fit(X, y)
    assert "has no observed" not in str(info.value)


@pytest.mark.parametrize(
    "params",
    [
        {"missing_policy": "impute"},
        {"missing_policy": "impute_with_indicator"},
        {"missing_policy": "separate_state"},
        {"missing_policy": "propagate"},
        {"add_missing_indicator": True},
        {"numerical_imputation": None},
        {"numerical_imputation": None, "add_missing_indicator": True},
        {"numerical_imputation": "constant"},
    ],
)
def test_empty_numerical_column_under_missing_value_settings(params, rng):
    X = _frame_with_empty_column(rng, "numerical")
    y = rng.normal(size=len(X))
    pre = Preprocessor(numerical_method="minmax", **params)
    with warnings.catch_warnings():
        # The all-NaN column reaches MinMaxScaler when imputation is disabled.
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        _fit_empty(pre, X, y)
    _assert_metadata_matches_output(pre, X, _observed(X, "numerical", rng))


@pytest.mark.parametrize("method", ["int", "one-hot", "none"])
@pytest.mark.parametrize(
    "params",
    [
        {},
        {"missing_policy": "impute_with_indicator"},
        {"missing_policy": "separate_state"},
        {"missing_policy": "propagate"},
        {"add_missing_indicator": True},
        {"categorical_imputation": None},
        {"categorical_imputation": "constant"},
    ],
)
def test_empty_categorical_column_keeps_its_block(method, params, rng):
    X = _frame_with_empty_column(rng, "categorical")
    y = rng.normal(size=len(X))
    pre = Preprocessor(numerical_method="minmax", categorical_method=method, **params)
    _fit_empty(pre, X, y)
    _assert_metadata_matches_output(pre, X, _observed(X, "categorical", rng))


def test_empty_numerical_column_counts_as_constant_for_the_policy(rng):
    X = _frame_with_empty_column(rng, "numerical")
    y = rng.normal(size=len(X))
    with pytest.warns(DataWarning), pytest.raises(PretabDataError, match="constant"):
        Preprocessor(numerical_method="minmax", policy={"constant": "error"}).fit(X, y)


def test_column_empty_in_a_cross_fitting_fold_keeps_the_width(rng):
    n = 60
    X = _frame_with_empty_column(rng, "numerical", n=n)
    X.loc[n - 6 :, "empty"] = rng.normal(size=6)  # observed only in the last fold
    y = rng.normal(size=n)
    cross_fitted = CrossFittedTransformer(Preprocessor(numerical_method="rbf"), n_folds=10, shuffle=False)
    out_of_fold = np.asarray(cross_fitted.fit_transform(X, y))
    assert out_of_fold.shape == cross_fitted.transform(X).shape
