"""Regression tests for the shared adaptive / fixed output-dimension sizing.

Phase 10 gives every expansion family one adaptive implementation via
``pretab.core.adaptive.AdaptiveResolutionMixin``: ``adaptive=False`` reproduces
the fixed-width (or current data-driven) behavior, while ``adaptive=True`` sizes
each feature within ``[min_output_dim, max_output_dim]`` on the target-aware
(tree / selector) path.
"""

import numpy as np
import pandas as pd
import pytest

from pretab import Preprocessor
from pretab.core.adaptive import AdaptiveResolutionMixin
from pretab.transformers import (
    BSplineTransformer,
    CubicSplineTransformer,
    NaturalCubicSplineTransformer,
    PLETransformer,
    PSplineTransformer,
    RBFExpansionTransformer,
    ReLUExpansionTransformer,
    SigmoidExpansionTransformer,
    TanhExpansionTransformer,
    TensorProductSplineTransformer,
)
from pretab.transformers.splines import CARTKnotSelector


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    X = rng.uniform(0.0, 1.0, size=(300, 2))
    y = np.sin(6 * X[:, 0]) + X[:, 1] ** 2 + 0.05 * rng.standard_normal(300)
    return X, y


# --------------------------------------------------------------------------- #
# The shared mixin                                                            #
# --------------------------------------------------------------------------- #
class _Dummy(AdaptiveResolutionMixin):
    def __init__(self, adaptive):
        self.adaptive = adaptive


def test_mixin_fixed_collapses_to_output_dim():
    lo, hi = _Dummy(adaptive=False)._resolve_output_bounds(7, None, None, floor=1)
    assert (lo, hi) == (7, 7)


def test_mixin_fixed_validates_output_dim_against_requests():
    dummy = _Dummy(adaptive=False)
    with pytest.raises(ValueError, match="output_dim must be >= min_output_dim"):
        dummy._resolve_output_bounds(3, 5, 10, floor=1)
    with pytest.raises(ValueError, match="output_dim must be <= max_output_dim"):
        dummy._resolve_output_bounds(20, 5, 10, floor=1)


def test_mixin_adaptive_window_falls_back_to_output_dim():
    lo, hi = _Dummy(adaptive=True)._resolve_output_bounds(7, None, None, floor=1)
    assert (lo, hi) == (7, 7)
    lo, hi = _Dummy(adaptive=True)._resolve_output_bounds(7, 4, 12, floor=1)
    assert (lo, hi) == (4, 12)


def test_mixin_floor_and_ceil_and_ordering():
    dummy = _Dummy(adaptive=True)
    with pytest.raises(ValueError, match="min_output_dim must be >= 4"):
        dummy._resolve_output_bounds(7, 2, 12, floor=4, floor_label="4")
    with pytest.raises(ValueError, match="should be <= 50"):
        dummy._resolve_output_bounds(7, 5, 60, floor=1, ceil=50)
    with pytest.raises(ValueError, match="min_output_dim must be <= max_output_dim"):
        dummy._resolve_output_bounds(7, 9, 6, floor=1)


# --------------------------------------------------------------------------- #
# Feature maps                                                                #
# --------------------------------------------------------------------------- #
FEATURE_MAPS = [
    RBFExpansionTransformer,
    ReLUExpansionTransformer,
    SigmoidExpansionTransformer,
    TanhExpansionTransformer,
]


@pytest.mark.parametrize("Cls", FEATURE_MAPS)
def test_feature_map_adaptive_tree_clamps_within_window(Cls, data):
    X, y = data
    t = Cls(output_dim=4, use_target=True, adaptive=True, min_output_dim=2, max_output_dim=8)
    Xt = t.fit_transform(X, y)
    for centers in t.centers_:
        assert 2 <= len(centers) <= 8
    assert Xt.shape[1] == sum(len(c) for c in t.centers_)


@pytest.mark.parametrize("Cls", FEATURE_MAPS)
def test_feature_map_adaptive_false_reproduces_fixed(Cls, data):
    X, y = data
    fixed = Cls(output_dim=5, use_target=True).fit(X, y)
    non_adaptive = Cls(output_dim=5, use_target=True, adaptive=False).fit(X, y)
    assert [len(c) for c in fixed.centers_] == [len(c) for c in non_adaptive.centers_]


@pytest.mark.parametrize("Cls", FEATURE_MAPS)
def test_feature_map_adaptive_is_noop_on_quantile_path(Cls, data):
    X, _ = data
    t = Cls(
        output_dim=5,
        use_target=False,
        strategy="quantile",
        adaptive=True,
        min_output_dim=2,
        max_output_dim=9,
    ).fit(X)
    assert all(len(c) == 5 for c in t.centers_)


# --------------------------------------------------------------------------- #
# Legacy splines (selector = target-aware path)                               #
# --------------------------------------------------------------------------- #
LEGACY_SPLINES = [
    (CubicSplineTransformer, 8),
    (NaturalCubicSplineTransformer, 6),
    (PSplineTransformer, 10),
]


@pytest.mark.parametrize("Cls,output_dim", LEGACY_SPLINES)
def test_legacy_spline_adaptive_clamps_basis_within_window(Cls, output_dim, data):
    X, y = data
    lo, hi = max(output_dim - 3, 2), output_dim + 6
    t = Cls(
        output_dim=output_dim,
        selector=CARTKnotSelector(max_basis_functions=25),
        task="regression",
        adaptive=True,
        min_output_dim=lo,
        max_output_dim=hi,
    )
    t.fit_transform(X, y)
    for n_basis in t.n_basis_:
        assert lo <= n_basis <= hi


@pytest.mark.parametrize("Cls,output_dim", LEGACY_SPLINES)
def test_legacy_spline_adaptive_false_reproduces_selector_output(Cls, output_dim, data):
    X, y = data
    fixed = Cls(
        output_dim=output_dim,
        selector=CARTKnotSelector(max_basis_functions=25),
        task="regression",
    ).fit(X, y)
    non_adaptive = Cls(
        output_dim=output_dim,
        selector=CARTKnotSelector(max_basis_functions=25),
        task="regression",
        adaptive=False,
    ).fit(X, y)
    assert fixed.n_basis_ == non_adaptive.n_basis_


def test_tensor_product_adaptive_clamps_per_marginal(data):
    X, y = data
    t = TensorProductSplineTransformer(
        output_dim=5,
        selector=CARTKnotSelector(max_basis_functions=12),
        task="regression",
        adaptive=True,
        min_output_dim=4,
        max_output_dim=7,
    )
    Xt = t.fit_transform(X, y)
    marginals = [n + t.degree + 1 for n in t.n_knots_]
    for n_basis in marginals:
        assert 4 <= n_basis <= 7
    assert Xt.shape[1] == int(np.prod(marginals))


# --------------------------------------------------------------------------- #
# B/M/I and PLE remain adaptive after the shared-mixin retarget               #
# --------------------------------------------------------------------------- #
def test_bspline_adaptive_selector_within_window(data):
    X, y = data
    t = BSplineTransformer(
        output_dim=8,
        selector=CARTKnotSelector(max_basis_functions=25),
        adaptive=True,
        min_output_dim=6,
        max_output_dim=14,
    )
    t.fit_transform(X[:, [0]], y)
    for n_basis in t.n_basis_:
        assert 6 <= n_basis <= 14


def test_ple_adaptive_within_window(data):
    X, y = data
    t = PLETransformer(output_dim=8, adaptive=True, min_output_dim=3, max_output_dim=12)
    t.fit(X, y)
    for n_bins in t.n_bins_per_feature_:
        assert 3 <= n_bins <= 12


def test_ple_adaptive_false_fixed_width(data):
    X, y = data
    t = PLETransformer(output_dim=6, adaptive=False).fit(X, y)
    assert all(n == 6 for n in t.n_bins_per_feature_)


# --------------------------------------------------------------------------- #
# Preprocessor end-to-end                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def frame():
    rng = np.random.default_rng(11)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.uniform(size=400)})
    y = X["a"] ** 2 + X["b"] + 0.1 * rng.standard_normal(400)
    return X, y


def test_preprocessor_non_adaptive_output_dim_outside_default_window(frame):
    # Default min/max are 5/10; a fixed output_dim outside that must not raise.
    X, y = frame
    out = Preprocessor(numerical_method="ple", output_dim=32, cat_cutoff=0.0).fit_transform(
        X, y, return_array=True
    )
    assert isinstance(out, np.ndarray)
    assert out.shape[1] == 64


def test_preprocessor_adaptive_ple_within_window(frame):
    X, y = frame
    pre = Preprocessor(
        numerical_method="ple",
        adaptive=True,
        min_output_dim=4,
        max_output_dim=12,
        cat_cutoff=0.0,
    )
    out = pre.fit_transform(X, y, return_array=True)
    assert isinstance(out, np.ndarray)
    assert 2 * 4 <= out.shape[1] <= 2 * 12


def test_preprocessor_adaptive_rbf_within_window(frame):
    X, y = frame
    pre = Preprocessor(
        numerical_method="rbf",
        adaptive=True,
        use_target=True,
        min_output_dim=3,
        max_output_dim=9,
        cat_cutoff=0.0,
    )
    out = pre.fit_transform(X, y, return_array=True)
    assert isinstance(out, np.ndarray)
    assert 2 * 3 <= out.shape[1] <= 2 * 9
