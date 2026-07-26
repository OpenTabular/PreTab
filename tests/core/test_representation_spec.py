import warnings

import numpy as np
import pytest

from pretab.core.representation import RepresentationSpec, RepresentationSpecMixin
from pretab.transformers import (
    BSplineTransformer,
    ContinuousOrdinalTransformer,
    CubicRegressionSplineTransformer,
    FourierFeatureTransformer,
    ISplineTransformer,
    MSplineTransformer,
    NaturalCubicSplineTransformer,
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
)

RNG = np.random.default_rng(0)
X_UNI = np.linspace(0.1, 5.0, 80).reshape(-1, 1)
X_MULTI = RNG.uniform(0.0, 1.0, size=(80, 2))
X_PERIODIC = RNG.uniform(0.0, 24.0, size=(80, 1))
X_CAT = np.array([["a"], ["b"], ["a"], ["c"]] * 20, dtype=object)
X_ORDINAL = np.array([[0], [1], [2], [1]] * 20)
Y = RNG.uniform(0.0, 1.0, size=80)


def _fit(transformer, X, y=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return transformer.fit(X, y)


# (id, transformer, X, y, expected_family, expected_scope, expected_supervision)
CASES = [
    ("bspline", BSplineTransformer(output_dim=6), X_UNI, None, "bspline", "univariate", "optional"),
    ("mspline", MSplineTransformer(output_dim=6), X_UNI, None, "mspline", "univariate", "optional"),
    ("ispline", ISplineTransformer(output_dim=6), X_UNI, None, "ispline", "univariate", "optional"),
    (
        "naturalspline",
        NaturalCubicSplineTransformer(output_dim=5),
        X_UNI,
        None,
        "naturalspline",
        "univariate",
        "optional",
    ),
    (
        "cubicspline",
        CubicRegressionSplineTransformer(output_dim=5),
        X_UNI,
        None,
        "cubicspline",
        "univariate",
        "optional",
    ),
    ("pspline", PSplineTransformer(output_dim=8), X_UNI, None, "pspline", "univariate", "unsupervised"),
    (
        "tensorspline",
        TensorProductSplineTransformer(output_dim=4),
        X_MULTI,
        None,
        "tensorspline",
        "multivariate",
        "unsupervised",
    ),
    (
        "thinplate",
        ThinPlateSplineTransformer(n_components=6),
        X_MULTI,
        None,
        "thinplate",
        "multivariate",
        "unsupervised",
    ),
    ("rbf", RBFExpansionTransformer(output_dim=5), X_UNI, None, "rbf", "univariate", "optional"),
    ("relu", ReLUExpansionTransformer(output_dim=5), X_UNI, None, "relu", "univariate", "optional"),
    ("sigmoid", SigmoidExpansionTransformer(output_dim=5), X_UNI, None, "sigmoid", "univariate", "optional"),
    ("tanh", TanhExpansionTransformer(output_dim=5), X_UNI, None, "tanh", "univariate", "optional"),
    (
        "fourier",
        FourierFeatureTransformer(n_frequencies=4),
        X_UNI,
        None,
        "fourier",
        "univariate",
        "unsupervised",
    ),
    (
        "random_fourier",
        RandomFourierFeaturesTransformer(n_components=10, random_state=0),
        X_MULTI,
        None,
        "random_fourier",
        "multivariate",
        "unsupervised",
    ),
    (
        "nystroem",
        NystroemFeaturesTransformer(n_components=8, random_state=0),
        X_MULTI,
        None,
        "nystroem",
        "multivariate",
        "unsupervised",
    ),
    (
        "periodic",
        PeriodicEncodingTransformer(period=24, harmonics=2),
        X_PERIODIC,
        None,
        "periodic",
        "univariate",
        "unsupervised",
    ),
    (
        "binning",
        NumericBinningTransformer(output_dim=4, encode="onehot"),
        X_UNI,
        None,
        "binning",
        "univariate",
        "unsupervised",
    ),
    (
        "piecewise_linear",
        PLETransformer(output_dim=4),
        X_UNI,
        Y,
        "piecewise_linear",
        "univariate",
        "supervised",
    ),
    ("ordinal", ContinuousOrdinalTransformer(), X_CAT, None, "ordinal", "univariate", "unsupervised"),
    (
        "onehot",
        OneHotFromOrdinalTransformer(),
        X_ORDINAL,
        None,
        "onehot",
        "univariate",
        "unsupervised",
    ),
]
CASE_IDS = [case[0] for case in CASES]


@pytest.mark.parametrize(
    ("transformer", "X", "y", "family", "scope", "supervision"),
    [case[1:] for case in CASES],
    ids=CASE_IDS,
)
def test_get_representation_spec_metadata(transformer, X, y, family, scope, supervision):
    _fit(transformer, X, y)
    spec = transformer.get_representation_spec()
    assert isinstance(spec, RepresentationSpec)
    assert spec.family == family
    assert spec.scope == scope
    assert spec.supervision == supervision
    assert spec.is_interaction == (scope == "multivariate")


@pytest.mark.parametrize(
    ("transformer", "X", "y"),
    [(case[1], case[2], case[3]) for case in CASES],
    ids=CASE_IDS,
)
def test_output_features_match_get_feature_names_out(transformer, X, y):
    _fit(transformer, X, y)
    input_features = [f"col{i}" for i in range(transformer.n_features_in_)]
    spec = transformer.get_representation_spec(input_features=input_features)
    expected = tuple(str(name) for name in transformer.get_feature_names_out(input_features))
    assert spec.output_features == expected
    assert spec.output_dim == len(expected)
    assert spec.input_features == tuple(input_features)


@pytest.mark.parametrize(
    ("transformer", "X", "y"),
    [(case[1], case[2], case[3]) for case in CASES],
    ids=CASE_IDS,
)
def test_spec_round_trips_through_dict(transformer, X, y):
    _fit(transformer, X, y)
    spec = transformer.get_representation_spec()
    assert RepresentationSpec.from_dict(spec.to_dict()) == spec


def test_every_transformer_has_representation_spec():
    for _id, transformer, *_ in CASES:
        assert isinstance(transformer, RepresentationSpecMixin)
        assert hasattr(transformer, "get_representation_spec")


def test_periodic_spec_reports_period():
    x_month = RNG.uniform(0.0, 12.0, size=(80, 1))
    transformer = _fit(PeriodicEncodingTransformer(period=12, harmonics=1), x_month)
    spec = transformer.get_representation_spec()
    assert spec.periodic is True
    assert spec.period == 12.0


def test_spline_spec_exposes_knots_and_degree():
    transformer = _fit(BSplineTransformer(output_dim=6, degree=3), X_UNI)
    spec = transformer.get_representation_spec()
    assert spec.degree == 3
    assert spec.location_kind == "knots"
    assert spec.locations is not None
    assert all(isinstance(value, float) for group in spec.locations for value in group)


def test_center_expansion_spec_exposes_centers():
    transformer = _fit(RBFExpansionTransformer(output_dim=5), X_UNI)
    spec = transformer.get_representation_spec()
    assert spec.component_kind == "center"
    assert spec.location_kind == "centers"
    assert spec.local_support is True


def test_to_dict_is_json_friendly():
    transformer = _fit(BSplineTransformer(output_dim=5), X_UNI)
    data = transformer.get_representation_spec().to_dict()
    assert isinstance(data["input_features"], list)
    assert isinstance(data["output_features"], list)
    assert data["locations"] is None or isinstance(data["locations"], list)
