"""Shared invariants for every spline family's penalty matrix.

Closes a gap flagged during the v1.0.0 hardening review: symmetry/PSD, rank/nullity,
and affine-rescaling behavior were previously checked ad hoc per family (if at all)
rather than through one shared, reusable check that every family can be run against.
"""

import numpy as np
import pytest

from pretab.exceptions import ConfigWarning
from pretab.transformers import (
    BSplineTransformer,
    CubicRegressionSplineTransformer,
    ISplineTransformer,
    MSplineTransformer,
    NaturalCubicSplineTransformer,
    PSplineTransformer,
    TensorProductSplineTransformer,
    ThinPlateSplineTransformer,
)

# Families whose penalty is the pure D^T D finite-difference operator: it depends only
# on the basis width and `diff_order`, never on the fitted data's values.
DIFFERENCE_PENALTY_FAMILIES = [BSplineTransformer, MSplineTransformer, ISplineTransformer]

ALL_SPLINE_FAMILIES = [
    (BSplineTransformer, {"output_dim": 8}, 1),
    (MSplineTransformer, {"output_dim": 8}, 1),
    (ISplineTransformer, {"output_dim": 8}, 1),
    (PSplineTransformer, {"output_dim": 8}, 1),
    (NaturalCubicSplineTransformer, {"output_dim": 6}, 1),
    (CubicRegressionSplineTransformer, {"output_dim": 8}, 1),
    (TensorProductSplineTransformer, {"output_dim": 4}, 2),
]


def assert_valid_penalty(P, *, expect_psd=True, atol=1e-8):
    """Assert a penalty matrix is square, symmetric, and (optionally) positive semi-definite."""
    assert P.ndim == 2
    assert P.shape[0] == P.shape[1]
    np.testing.assert_allclose(P, P.T, atol=atol)
    if expect_psd:
        assert np.linalg.eigvalsh(P).min() >= -atol


@pytest.fixture
def X_uniform():
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, size=(200, 1))


@pytest.fixture
def X_multi():
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, size=(200, 2))


@pytest.mark.parametrize("cls", DIFFERENCE_PENALTY_FAMILIES)
def test_difference_penalty_is_symmetric_psd(cls, X_uniform):
    assert_valid_penalty(cls(output_dim=8).fit(X_uniform).get_penalty_matrix())


def test_pspline_penalty_is_symmetric_psd(X_uniform):
    assert_valid_penalty(PSplineTransformer(output_dim=8).fit(X_uniform).get_penalty_matrix())


def test_natural_cubic_penalty_is_symmetric_psd(X_uniform):
    assert_valid_penalty(NaturalCubicSplineTransformer(output_dim=6).fit(X_uniform).get_penalty_matrix())


def test_cubic_regression_penalty_is_symmetric_psd(X_uniform):
    assert_valid_penalty(CubicRegressionSplineTransformer(output_dim=8).fit(X_uniform).get_penalty_matrix())


def test_tensor_product_penalty_is_symmetric_psd(X_multi):
    transformer = TensorProductSplineTransformer(output_dim=4).fit(X_multi)
    for P in transformer.get_penalty_matrices():
        assert_valid_penalty(P)


def test_thinplate_penalty_is_symmetric_but_not_guaranteed_psd():
    X = np.linspace(0, 1, 40).reshape(-1, 1)
    transformer = ThinPlateSplineTransformer(n_components=6, random_state=0).fit(X)
    with pytest.warns(ConfigWarning, match="experimental"):
        P = transformer.get_penalty_matrix()
    # PSD is explicitly not guaranteed here (see the class docstring); only symmetry is.
    assert_valid_penalty(P, expect_psd=False)


@pytest.mark.parametrize("cls", DIFFERENCE_PENALTY_FAMILIES)
@pytest.mark.parametrize("diff_order", [1, 2, 3])
def test_difference_penalty_rank_matches_null_space_formula(cls, diff_order, X_uniform):
    # D^T D built from a diff_order-th difference operator has rank n_basis - diff_order;
    # its null space is exactly the discrete polynomials of degree < diff_order.
    transformer = cls(output_dim=8, include_bias=False).fit(X_uniform)
    n_basis = transformer.n_basis_[0]
    P = transformer.get_penalty_matrix(diff_order=diff_order)
    assert np.linalg.matrix_rank(P) == n_basis - diff_order


@pytest.mark.parametrize("diff_order", [1, 2, 3])
def test_pspline_penalty_rank_matches_configured_diff_order(diff_order, X_uniform):
    transformer = PSplineTransformer(output_dim=8, diff_order=diff_order, include_bias=False).fit(X_uniform)
    n_basis = transformer.n_basis_[0]
    P = transformer.get_penalty_matrix()
    assert np.linalg.matrix_rank(P) == n_basis - diff_order


@pytest.mark.parametrize("diff_order", [1, 2])
def test_tensor_product_marginal_penalty_rank_matches_diff_order(diff_order, X_multi):
    transformer = TensorProductSplineTransformer(output_dim=4, diff_order=diff_order, include_bias=False).fit(X_multi)
    for i, n_basis in enumerate(transformer.marginal_sizes_):
        P = transformer.get_penalty_matrix(feature_index=i)
        assert np.linalg.matrix_rank(P) == n_basis - diff_order


@pytest.mark.parametrize("cls", DIFFERENCE_PENALTY_FAMILIES)
def test_difference_penalty_is_invariant_to_affine_rescaling(cls, X_uniform):
    P_original = cls(output_dim=8).fit(X_uniform).get_penalty_matrix()
    P_rescaled = cls(output_dim=8).fit(3.0 * X_uniform + 5.0).get_penalty_matrix()
    np.testing.assert_array_equal(P_original, P_rescaled)


def test_pspline_penalty_is_invariant_to_affine_rescaling(X_uniform):
    P_original = PSplineTransformer(output_dim=8).fit(X_uniform).get_penalty_matrix()
    P_rescaled = PSplineTransformer(output_dim=8).fit(3.0 * X_uniform + 5.0).get_penalty_matrix()
    np.testing.assert_array_equal(P_original, P_rescaled)


def test_tensor_product_penalty_is_invariant_to_affine_rescaling(X_multi):
    P_original = TensorProductSplineTransformer(output_dim=4).fit(X_multi).get_penalty_matrix(feature_index=0)
    P_rescaled = (
        TensorProductSplineTransformer(output_dim=4).fit(3.0 * X_multi + 5.0).get_penalty_matrix(feature_index=0)
    )
    np.testing.assert_array_equal(P_original, P_rescaled)


def test_natural_cubic_penalty_scales_as_cube_of_domain_scale(X_uniform):
    # Unlike the difference penalties above, an integrated-squared-second-derivative
    # penalty is not scale invariant: rescaling the fitted domain by `a` rescales every
    # penalty entry by exactly a**3 (verified numerically against a from-scratch fit).
    a = 3.0
    P_original = NaturalCubicSplineTransformer(output_dim=6).fit(X_uniform).get_penalty_matrix()
    P_rescaled = NaturalCubicSplineTransformer(output_dim=6).fit(a * X_uniform).get_penalty_matrix()
    np.testing.assert_allclose(P_rescaled, P_original * a**3, rtol=1e-6)


@pytest.mark.parametrize(("cls", "kwargs", "n_features"), ALL_SPLINE_FAMILIES)
def test_feature_names_out_length_matches_transform_width(cls, kwargs, n_features):
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(200, n_features))
    transformer = cls(**kwargs).fit(X)
    Xt = transformer.transform(X)
    assert len(transformer.get_feature_names_out()) == Xt.shape[1]


def test_thinplate_feature_names_out_length_matches_transform_width():
    X = np.linspace(0, 1, 40).reshape(-1, 1)
    transformer = ThinPlateSplineTransformer(n_components=6, random_state=0).fit(X)
    Xt = transformer.transform(X)
    assert len(transformer.get_feature_names_out()) == Xt.shape[1]
