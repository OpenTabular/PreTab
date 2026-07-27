import warnings

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pretab.transformers import PSplineTransformer


def test_pspline_single_feature_shape():
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    transformer = PSplineTransformer(output_dim=8)
    Xt = transformer.fit_transform(X)

    # width equals output_dim exactly (m = len(knots) - p - 1)
    assert Xt.shape == (30, 8)
    assert transformer.n_basis_ == [8]
    assert transformer.total_output_dim_ == 8
    assert np.isfinite(Xt).all()


def test_pspline_multi_feature_shape():
    X = np.random.rand(25, 2)
    transformer = PSplineTransformer(output_dim=5)
    Xt = transformer.fit_transform(X)

    total_basis = sum(transformer.n_basis_)
    assert Xt.shape == (25, total_basis)
    assert total_basis == 2 * 5
    assert np.isfinite(Xt).all()


def test_pspline_output_consistency():
    X = np.random.rand(20, 2)
    transformer = PSplineTransformer(output_dim=4)
    transformer.fit(X)
    Xt1 = transformer.transform(X)
    Xt2 = transformer.fit_transform(X)

    np.testing.assert_allclose(Xt1, Xt2, rtol=1e-5)


def test_pspline_penalty_matrix_shape_and_symmetry():
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    transformer = PSplineTransformer(output_dim=6)
    transformer.fit(X)
    P = transformer.get_penalty_matrix()

    assert P.shape[0] == P.shape[1]
    assert np.allclose(P, P.T, atol=1e-6)


def test_pspline_feature_names_out():
    X = np.random.rand(20, 2)
    transformer = PSplineTransformer(output_dim=5)
    Xt = transformer.fit_transform(X)

    names = transformer.get_feature_names_out(["a", "b"])
    assert len(names) == Xt.shape[1]
    assert names[0] == "a_ps0"
    assert all(name.startswith(("a_ps", "b_ps")) for name in names)


def test_pspline_feature_names_out_default_input():
    X = np.random.rand(15, 2)
    transformer = PSplineTransformer(output_dim=4).fit(X)

    names = transformer.get_feature_names_out()
    assert len(names) == sum(transformer.n_basis_)
    assert names[0].startswith("x0_ps")


def test_pspline_allow_nan_tag():
    tags = PSplineTransformer().__sklearn_tags__()
    assert tags.input_tags.allow_nan is True


def test_pspline_transform_requires_fit():
    transformer = PSplineTransformer()
    with pytest.raises(NotFittedError):
        transformer.transform(np.random.rand(5, 1))
    with pytest.raises(NotFittedError):
        transformer.get_penalty_matrix()


# --------------------------------------------------------------------------- #
# The B-spline basis must cover the whole fitted range, right endpoint included.
#
# The degree-0 recursion base used a half-open span, so the largest observed
# value belonged to no span and its entire row evaluated to zero.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_pspline_is_partition_of_unity_including_max(degree):
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    Xt = PSplineTransformer(output_dim=8, degree=degree).fit_transform(X)

    np.testing.assert_allclose(Xt.sum(axis=1), 1.0)


def test_pspline_max_row_is_not_all_zero():
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    Xt = PSplineTransformer(output_dim=8).fit_transform(X)

    assert np.abs(Xt[-1]).sum() > 0


def test_pspline_clips_out_of_range_input():
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    transformer = PSplineTransformer(output_dim=8).fit(X)

    out = transformer.transform(np.array([[-0.5], [0.5], [1.5]]))

    # Out-of-range values evaluate on the boundary instead of vanishing.
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    np.testing.assert_allclose(out[0], transformer.transform(np.array([[0.0]]))[0])
    np.testing.assert_allclose(out[2], transformer.transform(np.array([[1.0]]))[0])
