import numpy as np
import pytest

from pretab.transformers import (
    BSplineTransformer,
    ISplineTransformer,
    MSplineTransformer,
)
from pretab.transformers.splines import CARTKnotSelector


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, size=(200, 1))
    y = np.sin(X[:, 0]) + 0.1 * rng.randn(200)
    return X, y


def test_bspline_shape_with_bias(data):
    X, _ = data
    transformer = BSplineTransformer(n_basis_functions=8, include_bias=True)
    Xt = transformer.fit_transform(X)
    assert Xt.shape == (200, 9)  # 8 basis + 1 bias
    assert np.isfinite(Xt).all()


def test_bspline_shape_without_bias(data):
    X, _ = data
    transformer = BSplineTransformer(n_basis_functions=8, include_bias=False)
    Xt = transformer.fit_transform(X)
    assert Xt.shape == (200, 8)
    assert transformer.get_n_features_out() == 8


def test_bspline_multi_feature_shape():
    rng = np.random.RandomState(1)
    X = rng.uniform(0, 1, size=(120, 3))
    transformer = BSplineTransformer(n_basis_functions=6, include_bias=False)
    Xt = transformer.fit_transform(X)
    assert Xt.shape == (120, 18)  # 6 basis per feature, 3 features
    assert transformer.get_n_features_out() == 18


def test_bspline_reproducible(data):
    X, _ = data
    a = BSplineTransformer(n_basis_functions=7).fit_transform(X)
    b = BSplineTransformer(n_basis_functions=7).fit_transform(X)
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_bspline_feature_names_out(data):
    X, _ = data
    transformer = BSplineTransformer(n_basis_functions=8, include_bias=True).fit(X)
    names = transformer.get_feature_names_out(["age"])
    assert len(names) == transformer.get_n_features_out()
    assert names[0] == "age_bs0"


def test_bspline_n_knots_alias(data):
    X, _ = data
    a = BSplineTransformer(n_basis_functions=8).fit_transform(X)
    b = BSplineTransformer(n_knots=8).fit_transform(X)
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_bspline_rejects_small_basis():
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    with pytest.raises(ValueError, match="at least 1 internal knot"):
        BSplineTransformer(n_basis_functions=4).fit(X)


def test_bspline_rejects_large_basis():
    X = np.linspace(0, 1, 60).reshape(-1, 1)
    with pytest.raises(ValueError, match="<= 50"):
        BSplineTransformer(n_basis_functions=60).fit(X)


def test_mspline_non_negative(data):
    X, _ = data
    transformer = MSplineTransformer(n_basis_functions=8)
    Xt = transformer.fit_transform(X)
    assert Xt.shape == (200, 8)
    assert np.all(Xt >= -1e-9)


def test_mspline_handles_nan():
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    X[5] = np.nan
    transformer = MSplineTransformer(n_basis_functions=6)
    Xt = transformer.fit_transform(X)
    assert Xt.shape == (50, 6)
    assert np.isfinite(Xt).all()


def test_ispline_monotonic_increasing():
    X = np.linspace(0, 10, 200).reshape(-1, 1)
    transformer = ISplineTransformer(n_basis_functions=8, include_bias=False)
    Xt = transformer.fit_transform(X)
    # Each basis column is monotonically non-decreasing in x
    for j in range(Xt.shape[1]):
        assert np.all(np.diff(Xt[:, j]) >= -1e-9)


def test_ispline_bounded_unit_interval():
    X = np.linspace(0, 10, 200).reshape(-1, 1)
    Xt = ISplineTransformer(n_basis_functions=8).fit_transform(X)
    assert np.all(Xt >= -1e-9)
    assert np.all(Xt <= 1.0 + 1e-9)


def test_ispline_shape_multi_feature():
    rng = np.random.RandomState(2)
    X = rng.uniform(0, 5, size=(100, 2))
    transformer = ISplineTransformer(n_basis_functions=7, include_bias=False)
    Xt = transformer.fit_transform(X)
    assert Xt.shape == (100, 14)


def test_spline_with_cart_knot_selector(data):
    X, y = data
    selector = CARTKnotSelector(max_basis_functions=10, degree=3)
    transformer = BSplineTransformer(n_basis_functions=8, include_bias=False, knot_selector=selector)
    Xt = transformer.fit_transform(X, y)
    assert Xt.shape == (200, 8)
    assert np.isfinite(Xt).all()


def test_spline_penalty_matrix_symmetric(data):
    X, _ = data
    transformer = BSplineTransformer(n_basis_functions=8, include_bias=True).fit(X)
    P = transformer.get_penalty_matrix()
    assert P.shape[0] == P.shape[1]
    assert np.allclose(P, P.T, atol=1e-9)
