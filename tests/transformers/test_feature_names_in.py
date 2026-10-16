"""Standalone transformers follow scikit-learn's feature-name contract.

Fitted on a DataFrame they record ``feature_names_in_``, name their outputs after
the columns, reject renamed or reordered columns at ``transform`` and validate
``get_feature_names_out(input_features)``. Before, column names were dropped
(outputs were named ``x0_...``) and a reordered frame was silently accepted.
"""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.utils.estimator_checks import (
    check_dataframe_column_names_consistency,
    check_transformer_get_feature_names_out_pandas,
)

from pretab import transformers
from pretab.transformers import (
    BSplineTransformer,
    ContinuousOrdinalTransformer,
    CubicRegressionSplineTransformer,
    FourierFeatureTransformer,
    ISplineTransformer,
    LanguageEmbeddingTransformer,
    MissingStateIndicator,
    MSplineTransformer,
    NaturalCubicSplineTransformer,
    NoTransformer,
    NumericBinningTransformer,
    NystroemFeaturesTransformer,
    OneHotFromOrdinalTransformer,
    PeriodicEncodingTransformer,
    PLETransformer,
    PSplineTransformer,
    RandomFourierFeaturesTransformer,
    RBFExpansionTransformer,
    ReLUExpansionTransformer,
    SigmoidExpansionTransformer,
    TanhExpansionTransformer,
    TensorProductSplineTransformer,
    ThinPlateSplineTransformer,
    ToFloatTransformer,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore::pretab.exceptions.LeakageWarning"),
    pytest.mark.filterwarnings("ignore:OneHotFromOrdinalTransformer is deprecated:DeprecationWarning"),
]


class _StubTextModel:
    """Stand-in for a SentenceTransformer, so no optional dependency is needed."""

    def encode(self, values, convert_to_numpy=True):
        return np.array([[len(str(value)), 1.0] for value in values], dtype=float)


# Every transformer exported from ``pretab.transformers``. Each case names the
# kind of input frame the transformer accepts.
CASES = {
    "rbf": (lambda: RBFExpansionTransformer(output_dim=3), "numeric"),
    "relu": (lambda: ReLUExpansionTransformer(output_dim=3), "numeric"),
    "sigmoid": (lambda: SigmoidExpansionTransformer(output_dim=3), "numeric"),
    "tanh": (lambda: TanhExpansionTransformer(output_dim=3), "numeric"),
    "ple": (lambda: PLETransformer(output_dim=3, random_state=0), "numeric"),
    "binning": (lambda: NumericBinningTransformer(output_dim=3, encode="onehot"), "numeric"),
    "periodic": (lambda: PeriodicEncodingTransformer(period=12), "periodic"),
    "bspline": (lambda: BSplineTransformer(output_dim=5), "numeric"),
    "mspline": (lambda: MSplineTransformer(output_dim=5), "numeric"),
    "ispline": (lambda: ISplineTransformer(output_dim=5), "numeric"),
    "naturalspline": (lambda: NaturalCubicSplineTransformer(), "numeric"),
    "cubicspline": (lambda: CubicRegressionSplineTransformer(), "numeric"),
    "pspline": (lambda: PSplineTransformer(), "numeric"),
    "tensorspline": (lambda: TensorProductSplineTransformer(), "numeric"),
    "thinplate": (lambda: ThinPlateSplineTransformer(), "numeric"),
    "fourier": (lambda: FourierFeatureTransformer(), "numeric"),
    "random_fourier": (lambda: RandomFourierFeaturesTransformer(random_state=0), "numeric"),
    "nystroem": (lambda: NystroemFeaturesTransformer(random_state=0), "numeric"),
    "ordinal": (lambda: ContinuousOrdinalTransformer(), "categorical"),
    "onehot_from_ordinal": (lambda: OneHotFromOrdinalTransformer(), "codes"),
    "language_embedding": (lambda: LanguageEmbeddingTransformer(model=_StubTextModel()), "categorical"),
    "missing_state": (lambda: MissingStateIndicator(), "numeric"),
    "no_transform": (lambda: NoTransformer(), "numeric"),
    "to_float": (lambda: ToFloatTransformer(), "numeric"),
}
IDS = list(CASES)
# scikit-learn's checks feed standardized data, which is outside the periodic
# encoding's required input range [0, period].
SKLEARN_CHECK_IDS = [case for case in IDS if case != "periodic"]


def _frame(kind):
    rng = np.random.default_rng(0)
    n = 100
    if kind == "numeric":
        X = pd.DataFrame({"age": rng.normal(40, 10, n), "income": rng.uniform(1, 5, n)})
    elif kind == "periodic":
        X = pd.DataFrame({"age": rng.uniform(0, 12, n), "income": rng.uniform(0, 12, n)})
    elif kind == "categorical":
        X = pd.DataFrame({"age": rng.choice(["young", "old"], n), "income": rng.choice(["low", "mid", "high"], n)})
    else:
        X = pd.DataFrame({"age": rng.integers(0, 3, n), "income": rng.integers(0, 4, n)})
    return X, rng.normal(size=n)


def _fitted(case):
    make, kind = CASES[case]
    X, y = _frame(kind)
    return make().fit(X, y), X, y


def test_cases_cover_every_exported_transformer():
    exported = {name for name in transformers.__all__ if inspect.isclass(getattr(transformers, name))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        covered = {type(make()).__name__ for make, _ in CASES.values()}
    assert covered == exported


@pytest.mark.parametrize("case", IDS)
def test_dataframe_fit_records_feature_names_and_uses_them(case):
    transformer, _, _ = _fitted(case)
    assert list(transformer.feature_names_in_) == ["age", "income"]
    names = [str(name) for name in transformer.get_feature_names_out()]
    assert all(("age" in name) or ("income" in name) for name in names)
    assert not any("x0" in name or "x1" in name for name in names)


@pytest.mark.parametrize("case", IDS)
def test_reordered_columns_are_rejected(case):
    transformer, X, _ = _fitted(case)
    with pytest.raises(ValueError, match="feature names"):
        transformer.transform(X[["income", "age"]])


@pytest.mark.parametrize("case", IDS)
def test_renamed_columns_are_rejected(case):
    transformer, X, _ = _fitted(case)
    with pytest.raises(ValueError, match="feature names"):
        transformer.transform(X.rename(columns={"income": "salary"}))


@pytest.mark.parametrize("case", IDS)
def test_array_after_a_dataframe_fit_warns_like_scikit_learn(case):
    transformer, X, _ = _fitted(case)
    with pytest.warns(UserWarning, match="does not have valid feature names"):
        transformer.transform(X.to_numpy())


@pytest.mark.parametrize("case", IDS)
def test_input_features_are_validated(case):
    transformer, _, _ = _fitted(case)
    with pytest.raises(ValueError):
        transformer.get_feature_names_out(["age"])
    with pytest.raises(ValueError, match="not equal to feature_names_in_"):
        transformer.get_feature_names_out(["a", "b"])
    np.testing.assert_array_equal(
        transformer.get_feature_names_out(["age", "income"]), transformer.get_feature_names_out()
    )


@pytest.mark.parametrize("case", IDS)
def test_array_fit_keeps_generic_names_and_drops_stale_feature_names(case):
    transformer, X, y = _fitted(case)
    transformer.fit(X.to_numpy(), y)
    assert not hasattr(transformer, "feature_names_in_")
    assert "x0" in str(transformer.get_feature_names_out()[0])
    with pytest.raises(ValueError):
        transformer.get_feature_names_out(["a"])


@pytest.mark.parametrize("case", SKLEARN_CHECK_IDS)
def test_scikit_learn_feature_name_checks_pass(case):
    make, _ = CASES[case]
    estimator = make()
    name = type(estimator).__name__
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_dataframe_column_names_consistency(name, estimator)
        check_transformer_get_feature_names_out_pandas(name, estimator)


@pytest.mark.parametrize("cls", [ToFloatTransformer, NoTransformer])
def test_sparse_input_records_its_width(cls):
    """The one-hot encoder hands ``ToFloatTransformer`` a sparse matrix, which
    ``np.asarray`` turns into a 0-d object array, so one input feature was
    recorded instead of three; the now validated output names then failed."""
    transformer = cls().fit(sparse.csr_matrix(np.eye(3)))
    assert transformer.n_features_in_ == 3
    assert list(transformer.get_feature_names_out(["a", "b", "c"])) == ["a", "b", "c"]
