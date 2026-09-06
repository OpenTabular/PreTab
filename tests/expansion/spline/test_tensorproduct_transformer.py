import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pretab.exceptions import InvalidParamError
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


def test_tensorproduct_spline_penalty_matrices_match_true_quadratic_form():
    # Regression test: get_penalty_matrices() used to always place the marginal
    # penalty leftmost in the Kronecker chain, which only matched the
    # einsum+reshape flatten order used by transform() for dimension 0.
    rng = np.random.default_rng(0)
    X = rng.random((100, 3))
    transformer = TensorProductSplineTransformer(output_dim=5).fit(X)
    sizes = transformer.marginal_sizes_
    penalties = transformer.get_penalty_matrices()

    beta = rng.normal(size=sizes)
    beta_flat = beta.ravel()

    for dim, P in enumerate(penalties):
        D = transformer.penalties_[dim]
        # True smoothness along `dim`: apply D on that axis, summed over every
        # combination of the other axes' indices.
        true_value = 0.0
        for index in np.ndindex(*sizes):
            for index2 in np.ndindex(*sizes):
                if any(index[k] != index2[k] for k in range(len(sizes)) if k != dim):
                    continue
                true_value += beta[index] * D[index[dim], index2[dim]] * beta[index2]
        lib_value = beta_flat @ P @ beta_flat
        assert lib_value == pytest.approx(true_value, rel=1e-8), f"mismatch for dim={dim}"


def test_tensorproduct_out_of_range_transform_clips_to_boundary():
    # Regression test: out-of-range transform inputs used to produce an abrupt
    # all-zero row instead of clipping like the B/M/I splines already do.
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    X = np.hstack([X, X])
    transformer = TensorProductSplineTransformer(output_dim=4).fit(X)
    at_max = transformer.transform(np.array([[10.0, 10.0]]))
    just_past_max = transformer.transform(np.array([[10.0 + 1e-4, 10.0 + 1e-4]]))
    far_past_max = transformer.transform(np.array([[20.0, 20.0]]))
    np.testing.assert_allclose(just_past_max, at_max, atol=1e-6)
    np.testing.assert_allclose(far_past_max, at_max, atol=1e-6)


@pytest.mark.parametrize("diff_order", [-1, 0])
def test_tensorproduct_rejects_nonpositive_diff_order(diff_order):
    X = np.random.default_rng(0).random((30, 2))
    with pytest.raises(InvalidParamError, match="diff_order must be a positive integer"):
        TensorProductSplineTransformer(output_dim=6, diff_order=diff_order).fit(X)


def test_tensorproduct_rejects_diff_order_too_large_for_output_dim():
    X = np.random.default_rng(0).random((30, 2))
    with pytest.raises(InvalidParamError, match="diff_order=50 is too large"):
        TensorProductSplineTransformer(output_dim=6, diff_order=50).fit(X)


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
