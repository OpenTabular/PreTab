import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pretab.transformers import NaturalCubicSplineTransformer


def test_natural_spline_single_feature_shape():
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    transformer = NaturalCubicSplineTransformer(output_dim=5)
    Xt = transformer.fit_transform(X)

    # width equals output_dim exactly (m = T - 1, T = output_dim + 1 spanning knots)
    assert Xt.shape == (20, 5)
    assert transformer.n_knots_ == [4]
    assert transformer.total_output_dim_ == 5
    assert np.isfinite(Xt).all()


def test_natural_spline_multi_feature_shape():
    X = np.random.rand(25, 2)
    transformer = NaturalCubicSplineTransformer(output_dim=6, include_bias=True)
    Xt = transformer.fit_transform(X)

    n_features = X.shape[1]
    n_basis_per_feature = transformer.n_basis_[0]
    assert Xt.shape == (25, n_features * n_basis_per_feature)
    assert np.isfinite(Xt).all()


def test_natural_spline_output_consistency():
    X = np.random.rand(30, 2)
    transformer = NaturalCubicSplineTransformer(output_dim=4)
    transformer.fit(X)
    Xt1 = transformer.transform(X)
    Xt2 = transformer.fit_transform(X)

    np.testing.assert_allclose(Xt1, Xt2, rtol=1e-5)


def test_natural_spline_penalty_matrix_symmetry():
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    transformer = NaturalCubicSplineTransformer(output_dim=5)
    transformer.fit(X)
    P = transformer.get_penalty_matrix()

    assert P.shape[0] == P.shape[1]
    assert np.allclose(P, P.T, atol=1e-6)


def test_natural_spline_penalty_matrix_is_grid_density_invariant():
    # Regression test: the old np.gradient(..., axis=0) call (missing the x_grid
    # spacing argument) differentiated with respect to grid *index*, so the
    # returned penalty shrunk by roughly dx**4 whenever the evaluation range
    # changed. The penalty is computed on a fixed internal 200-point grid
    # regardless of range, so as a proxy we assert it is not vanishingly small
    # relative to a hand-checkable order of magnitude on a modest range.
    X = np.linspace(0, 10, 200).reshape(-1, 1)
    transformer = NaturalCubicSplineTransformer(output_dim=5, placement_strategy="uniform").fit(X)
    P = transformer.get_penalty_matrix()

    # The buggy implementation produced a diagonal on the order of 1e-3 for this
    # range; the correctly-scaled penalty is on the order of 1e2-1e3.
    assert np.diag(P).max() > 10.0


def test_natural_spline_feature_names_out():
    X = np.random.rand(20, 2)
    transformer = NaturalCubicSplineTransformer(output_dim=5)
    Xt = transformer.fit_transform(X)

    names = transformer.get_feature_names_out(["a", "b"])
    assert len(names) == Xt.shape[1]
    assert names[0] == "a_ncs0"
    assert all(name.startswith(("a_ncs", "b_ncs")) for name in names)


def test_natural_spline_feature_names_out_default_input():
    X = np.random.rand(15, 2)
    transformer = NaturalCubicSplineTransformer(output_dim=4).fit(X)

    names = transformer.get_feature_names_out()
    assert len(names) == sum(transformer.n_basis_)
    assert names[0].startswith("x0_ncs")


def test_natural_spline_allow_nan_tag():
    tags = NaturalCubicSplineTransformer().__sklearn_tags__()
    assert tags.input_tags.allow_nan is True


def test_natural_spline_transform_requires_fit():
    transformer = NaturalCubicSplineTransformer()
    with pytest.raises(NotFittedError):
        transformer.transform(np.random.rand(5, 1))
    with pytest.raises(NotFittedError):
        transformer.get_penalty_matrix()


def test_natural_spline_does_not_retain_training_design_matrix():
    """Regression guard for issue #19: fitted size must not scale with n_samples."""
    import pickle

    small = NaturalCubicSplineTransformer(output_dim=7).fit(np.random.rand(50, 1))
    large = NaturalCubicSplineTransformer(output_dim=7).fit(np.random.rand(20_000, 1))

    assert not hasattr(small, "designs_")
    assert len(pickle.dumps(large)) < 2 * len(pickle.dumps(small))
