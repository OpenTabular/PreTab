"""Backward-compatibility tests for the canonical parameter vocabulary.

Each transformer family accepts a canonical ``n_basis`` (plus ``min_basis`` /
``max_basis`` / ``use_target`` where relevant) while still honouring its legacy
argument names. Legacy names must emit a ``FutureWarning``, produce identical
output to the canonical spelling, and raise :class:`InvalidParamError` when a
canonical and a legacy alias are supplied together.
"""

import numpy as np
import pytest

from pretab.core.exceptions import InvalidParamError
from pretab.transformers import (
    BSplineTransformer,
    CubicSplineTransformer,
    CustomBinTransformer,
    ISplineTransformer,
    MSplineTransformer,
    NaturalCubicSplineTransformer,
    PLETransformer,
    PSplineTransformer,
    RBFExpansionTransformer,
    ReLUExpansionTransformer,
    SigmoidExpansionTransformer,
    TanhExpansionTransformer,
    TensorProductSplineTransformer,
)


@pytest.fixture
def X():
    rng = np.random.RandomState(0)
    return rng.uniform(-3, 3, size=(200, 1))


# (transformer, canonical kwargs, legacy kwargs) - both spellings request the
# same number of basis functions and must yield an identical design matrix.
SPLINE_CASES = [
    (CubicSplineTransformer, {"n_basis": 6}, {"n_knots": 6}),
    (NaturalCubicSplineTransformer, {"n_basis": 6}, {"n_knots": 6}),
    (PSplineTransformer, {"n_basis": 6}, {"n_knots": 6}),
    (TensorProductSplineTransformer, {"n_basis": 6}, {"n_knots": 6}),
    (BSplineTransformer, {"n_basis": 8}, {"n_basis_functions": 8}),
    (MSplineTransformer, {"n_basis": 8}, {"n_basis_functions": 8}),
    (ISplineTransformer, {"n_basis": 8}, {"n_basis_functions": 8}),
]


@pytest.mark.parametrize("cls, canonical, legacy", SPLINE_CASES)
def test_spline_legacy_alias_emits_futurewarning(cls, canonical, legacy, X):
    with pytest.warns(FutureWarning):
        cls(**legacy).fit(X)


@pytest.mark.parametrize("cls, canonical, legacy", SPLINE_CASES)
def test_spline_canonical_matches_legacy(cls, canonical, legacy, X):
    expected = cls(**canonical).fit_transform(X)
    with pytest.warns(FutureWarning):
        legacy_out = cls(**legacy).fit_transform(X)
    assert expected.shape == legacy_out.shape
    np.testing.assert_allclose(expected, legacy_out, rtol=1e-6)


@pytest.mark.parametrize("cls, canonical, legacy", SPLINE_CASES)
def test_spline_canonical_legacy_conflict_raises(cls, canonical, legacy, X):
    with pytest.raises(InvalidParamError):
        cls(**{**canonical, **legacy}).fit(X)


FEATURE_MAP_CLASSES = [
    RBFExpansionTransformer,
    ReLUExpansionTransformer,
    SigmoidExpansionTransformer,
    TanhExpansionTransformer,
]


@pytest.mark.parametrize("cls", FEATURE_MAP_CLASSES)
def test_feature_map_legacy_alias_emits_futurewarning(cls, X):
    with pytest.warns(FutureWarning):
        cls(n_centers=4, use_target=False).fit(X)


@pytest.mark.parametrize("cls", FEATURE_MAP_CLASSES)
def test_feature_map_canonical_matches_legacy(cls, X):
    expected = cls(n_basis=4, use_target=False).fit_transform(X)
    with pytest.warns(FutureWarning):
        legacy_out = cls(n_centers=4, use_decision_tree=False).fit_transform(X)
    assert expected.shape == legacy_out.shape
    np.testing.assert_allclose(expected, legacy_out, rtol=1e-6)


@pytest.mark.parametrize("cls", FEATURE_MAP_CLASSES)
def test_feature_map_n_basis_conflict_raises(cls, X):
    with pytest.raises(InvalidParamError):
        cls(n_basis=4, n_centers=4, use_target=False).fit(X)


@pytest.mark.parametrize("cls", FEATURE_MAP_CLASSES)
def test_feature_map_use_target_conflict_raises(cls, X):
    with pytest.raises(InvalidParamError):
        cls(n_basis=4, use_target=False, use_decision_tree=False).fit(X)


@pytest.fixture
def Xy():
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 1, size=(200, 1))
    y = np.sin(X[:, 0]) + 0.1 * rng.randn(200)
    return X, y


def test_ple_legacy_alias_emits_futurewarning(Xy):
    X, y = Xy
    with pytest.warns(FutureWarning):
        PLETransformer(n_bins=8).fit(X, y)


def test_ple_canonical_matches_legacy(Xy):
    X, y = Xy
    expected = PLETransformer(n_basis=8).fit_transform(X, y)
    with pytest.warns(FutureWarning):
        legacy_out = PLETransformer(n_bins=8).fit_transform(X, y)
    assert expected.shape == legacy_out.shape
    np.testing.assert_allclose(expected, legacy_out, rtol=1e-6)


def test_ple_canonical_legacy_conflict_raises(Xy):
    X, y = Xy
    with pytest.raises(InvalidParamError):
        PLETransformer(n_basis=8, n_bins=8).fit(X, y)


def test_custombin_legacy_alias_emits_futurewarning():
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    with pytest.warns(FutureWarning):
        CustomBinTransformer(bins=4).fit_transform(X)


def test_custombin_canonical_matches_legacy():
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    expected = CustomBinTransformer(n_basis=4).fit_transform(X)
    with pytest.warns(FutureWarning):
        legacy_out = CustomBinTransformer(bins=4).fit_transform(X)
    np.testing.assert_array_equal(expected, legacy_out)


def test_custombin_canonical_legacy_conflict_raises():
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    with pytest.raises(InvalidParamError):
        CustomBinTransformer(n_basis=4, bins=4).fit_transform(X)
