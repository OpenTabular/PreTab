import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pretab.transformers import TensorProductSplineTransformer


def test_tensorproduct_spline_output_shape():
    X = np.random.rand(20, 2)
    transformer = TensorProductSplineTransformer(output_dim=4)
    Xt = transformer.fit_transform(X)

    n_basis_0 = transformer.bases_[0].shape[1]
    n_basis_1 = transformer.bases_[1].shape[1]
    assert Xt.shape == (20, n_basis_0 * n_basis_1)
    # output_dim is per-marginal; total width is the product across dimensions
    assert Xt.shape == (20, 4**2)
    assert transformer.total_output_dim_ == 4**2
    assert np.isfinite(Xt).all()


def test_tensorproduct_spline_output_consistency():
    X = np.random.rand(30, 2)
    transformer = TensorProductSplineTransformer(output_dim=5)
    transformer.fit(X)
    Xt1 = transformer.transform(X)
    Xt2 = transformer.fit_transform(X)

    np.testing.assert_allclose(Xt1, Xt2, rtol=1e-5)


def test_tensorproduct_spline_penalty_matrices():
    X = np.random.rand(25, 2)
    transformer = TensorProductSplineTransformer(output_dim=4)
    transformer.fit(X)
    penalties = transformer.get_penalty_matrices()

    assert len(penalties) == 2
    for P in penalties:
        assert P.shape[0] == P.shape[1]
        assert np.allclose(P, P.T, atol=1e-6)


def test_tensorproduct_feature_names_out():
    X = np.random.rand(20, 2)
    transformer = TensorProductSplineTransformer(output_dim=4)
    Xt = transformer.fit_transform(X)

    names = transformer.get_feature_names_out(["a", "b"])
    assert len(names) == Xt.shape[1]
    assert names[0] == "tp_a0_b0"
    assert all(name.startswith("tp_") for name in names)


def test_tensorproduct_feature_names_out_default_input():
    X = np.random.rand(15, 2)
    transformer = TensorProductSplineTransformer(output_dim=4).fit(X)

    names = transformer.get_feature_names_out()
    n_expected = transformer.bases_[0].shape[1] * transformer.bases_[1].shape[1]
    assert len(names) == n_expected
    assert names[0].startswith("tp_")


def test_tensorproduct_allow_nan_tag():
    tags = TensorProductSplineTransformer().__sklearn_tags__()
    assert tags.input_tags.allow_nan is True


def test_tensorproduct_transform_requires_fit():
    transformer = TensorProductSplineTransformer()
    with pytest.raises(NotFittedError):
        transformer.transform(np.random.rand(5, 2))


# --------------------------------------------------------------------------- #
# Same endpoint guarantee as the p-spline: both families share one basis
# implementation in ``pretab.core.knots``.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_tensorproduct_is_partition_of_unity_including_max(degree):
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    Xt = TensorProductSplineTransformer(output_dim=8, degree=degree).fit_transform(X)

    np.testing.assert_allclose(Xt.sum(axis=1), 1.0)


def test_tensorproduct_max_row_is_not_all_zero():
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    Xt = TensorProductSplineTransformer(output_dim=8).fit_transform(X)

    assert np.abs(Xt[-1]).sum() > 0


def test_tensorproduct_clips_out_of_range_input():
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    transformer = TensorProductSplineTransformer(output_dim=8).fit(X)

    out = transformer.transform(np.array([[-0.5], [0.5], [1.5]]))

    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def test_tensorproduct_multivariate_still_products_the_marginals():
    rng = np.random.default_rng(0)
    transformer = TensorProductSplineTransformer(output_dim=5).fit(rng.random((20, 2)))

    assert transformer.transform(rng.random((7, 2))).shape == (7, 25)
