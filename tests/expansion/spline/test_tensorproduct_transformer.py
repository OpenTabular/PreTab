import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pretab.transformers import TensorProductSplineTransformer


def test_tensorproduct_spline_output_shape():
    X = np.random.rand(20, 2)
    transformer = TensorProductSplineTransformer(output_dim=4)
    Xt = transformer.fit_transform(X)

    n_basis_0 = transformer.marginal_sizes_[0]
    n_basis_1 = transformer.marginal_sizes_[1]
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
    n_expected = transformer.marginal_sizes_[0] * transformer.marginal_sizes_[1]
    assert len(names) == n_expected
    assert names[0].startswith("tp_")


def test_tensorproduct_allow_nan_tag():
    tags = TensorProductSplineTransformer().__sklearn_tags__()
    assert tags.input_tags.allow_nan is True


def test_tensorproduct_transform_requires_fit():
    transformer = TensorProductSplineTransformer()
    with pytest.raises(NotFittedError):
        transformer.transform(np.random.rand(5, 2))


def test_tensorproduct_does_not_retain_training_design_matrix():
    """Regression guard for issue #19: fitted size must not scale with n_samples."""
    import pickle

    small = TensorProductSplineTransformer(output_dim=7).fit(np.random.rand(50, 2))
    large = TensorProductSplineTransformer(output_dim=7).fit(np.random.rand(20_000, 2))

    assert not hasattr(small, "bases_")
    assert not hasattr(small, "X_design_")
    assert len(pickle.dumps(large)) < 2 * len(pickle.dumps(small))
