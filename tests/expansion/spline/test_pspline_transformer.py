import warnings

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pretab.exceptions import InvalidParamError
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


@pytest.mark.parametrize("diff_order", [-1, 0])
def test_pspline_rejects_nonpositive_diff_order(diff_order):
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    with pytest.raises(InvalidParamError, match="diff_order must be a positive integer"):
        PSplineTransformer(output_dim=8, diff_order=diff_order).fit(X)


def test_pspline_rejects_diff_order_too_large_for_output_dim():
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    with pytest.raises(InvalidParamError, match="diff_order=50 is too large"):
        PSplineTransformer(output_dim=8, diff_order=50).fit(X)


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
