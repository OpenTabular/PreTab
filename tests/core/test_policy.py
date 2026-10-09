"""Coverage for ``RepresentationPolicy.out_of_range`` actually reaching every spline family.

Each knot/range-based spline transformer (B/M/I-spline, P-spline, tensor-product,
natural-cubic, cubic-regression) accepts its own ``policy`` constructor parameter.
This is standalone-only wiring: the policy is genuinely respected by the transformer
itself, but it is not threaded through ``Preprocessor``/the registry (that remains a
separately-tracked gap; see ``dev/todo/release-1.0.0/bugfixes-1.0.0.md``).

These tests close the gap flagged in the v1.0.0 hardening review: no test previously
exercised ``RepresentationPolicy(out_of_range=...)`` through any spline transformer at
all, since ``resolve_out_of_range`` had zero call sites anywhere in the library.
``ThinPlateSplineTransformer`` is intentionally excluded: it has no single-feature
knot range in the same sense (a kernel evaluated at any point) and is out of scope for
this pass.
"""

import warnings

import numpy as np
import pytest

from pretab.core.policy import RepresentationPolicy
from pretab.exceptions import DataWarning, PretabDataError
from pretab.transformers import (
    BSplineTransformer,
    CubicRegressionSplineTransformer,
    ISplineTransformer,
    MSplineTransformer,
    NaturalCubicSplineTransformer,
    PSplineTransformer,
    TensorProductSplineTransformer,
)

SINGLE_FEATURE_FAMILIES = [
    (BSplineTransformer, {"output_dim": 8}),
    (MSplineTransformer, {"output_dim": 8}),
    (ISplineTransformer, {"output_dim": 8}),
    (PSplineTransformer, {"output_dim": 8}),
    (NaturalCubicSplineTransformer, {"output_dim": 6}),
    (CubicRegressionSplineTransformer, {"output_dim": 8}),
]


@pytest.mark.parametrize(("cls", "kwargs"), SINGLE_FEATURE_FAMILIES)
def test_out_of_range_error_policy_raises(cls, kwargs):
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    transformer = cls(policy=RepresentationPolicy(out_of_range="error"), **kwargs).fit(X)
    with pytest.raises(PretabDataError):
        transformer.transform(np.array([[20.0]]))


@pytest.mark.parametrize(("cls", "kwargs"), SINGLE_FEATURE_FAMILIES)
def test_out_of_range_warn_policy_warns(cls, kwargs):
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    transformer = cls(policy=RepresentationPolicy(out_of_range="warn"), **kwargs).fit(X)
    with pytest.warns(DataWarning):
        transformer.transform(np.array([[20.0]]))


@pytest.mark.parametrize(("cls", "kwargs"), SINGLE_FEATURE_FAMILIES)
def test_out_of_range_clip_policy_matches_boundary_value(cls, kwargs):
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    transformer = cls(policy=RepresentationPolicy(out_of_range="clip"), **kwargs).fit(X)
    at_max = transformer.transform(np.array([[10.0]]))
    past_max = transformer.transform(np.array([[20.0]]))
    np.testing.assert_allclose(past_max, at_max, atol=1e-6)


def test_tensorproduct_out_of_range_error_policy_raises():
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    X = np.hstack([X, X])
    transformer = TensorProductSplineTransformer(output_dim=4, policy=RepresentationPolicy(out_of_range="error")).fit(X)
    with pytest.raises(PretabDataError):
        transformer.transform(np.array([[20.0, 20.0]]))


def test_tensorproduct_out_of_range_clip_policy_matches_boundary_value():
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    X = np.hstack([X, X])
    transformer = TensorProductSplineTransformer(output_dim=4, policy=RepresentationPolicy(out_of_range="clip")).fit(X)
    at_max = transformer.transform(np.array([[10.0, 10.0]]))
    past_max = transformer.transform(np.array([[20.0, 20.0]]))
    np.testing.assert_allclose(past_max, at_max, atol=1e-6)


def test_natural_cubic_and_cubic_regression_default_to_extrapolate():
    # Unlike B/M/I/P-spline/tensor-product (which default to "clip"), these two
    # families extrapolate smoothly by design and only clip on request.
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    for cls, kwargs in [
        (NaturalCubicSplineTransformer, {"output_dim": 6}),
        (CubicRegressionSplineTransformer, {"output_dim": 8}),
    ]:
        default = cls(**kwargs).fit(X)
        clipped = cls(policy=RepresentationPolicy(out_of_range="clip"), **kwargs).fit(X)
        at_max = default.transform(np.array([[10.0]]))
        default_past_max = default.transform(np.array([[20.0]]))
        clipped_past_max = clipped.transform(np.array([[20.0]]))
        assert not np.allclose(default_past_max, at_max)
        np.testing.assert_allclose(clipped_past_max, at_max, atol=1e-6)


def test_bmi_and_pspline_and_tensor_default_to_clip():
    # These families have always clipped unconditionally; confirm the default
    # (policy=None) still does, now that it's routed through resolve_out_of_range.
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    for cls, kwargs in [
        (BSplineTransformer, {"output_dim": 8}),
        (MSplineTransformer, {"output_dim": 8}),
        (ISplineTransformer, {"output_dim": 8}),
        (PSplineTransformer, {"output_dim": 8}),
    ]:
        transformer = cls(**kwargs).fit(X)
        at_max = transformer.transform(np.array([[10.0]]))
        past_max = transformer.transform(np.array([[20.0]]))
        np.testing.assert_allclose(past_max, at_max, atol=1e-6)

    X2 = np.hstack([X, X])
    transformer = TensorProductSplineTransformer(output_dim=4).fit(X2)
    at_max = transformer.transform(np.array([[10.0, 10.0]]))
    past_max = transformer.transform(np.array([[20.0, 20.0]]))
    np.testing.assert_allclose(past_max, at_max, atol=1e-6)


def test_policy_dict_is_accepted():
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    transformer = BSplineTransformer(output_dim=8, policy={"out_of_range": "error"}).fit(X)
    with pytest.raises(PretabDataError):
        transformer.transform(np.array([[20.0]]))


# --- partial overrides and the "extrapolate" / "warn" reactions ---------------------

_CLIP_FAMILIES = [
    (BSplineTransformer, {"output_dim": 6}),
    (MSplineTransformer, {"output_dim": 6}),
    (ISplineTransformer, {"output_dim": 6}),
    (PSplineTransformer, {}),
]


@pytest.mark.parametrize(("cls", "kwargs"), _CLIP_FAMILIES)
def test_partial_policy_mapping_keeps_the_family_out_of_range_default(cls, kwargs):
    """A mapping naming only "constant" used to reset out_of_range to the dataclass
    default "extrapolate", silently turning the clip families' out-of-range rows to zero."""
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    probe = np.array([[1.05], [-0.05]])
    default = cls(**kwargs).fit(X).transform(probe)
    partial = cls(policy={"constant": "warn"}, **kwargs).fit(X).transform(probe)
    np.testing.assert_array_equal(partial, default)


def test_tensorproduct_partial_policy_mapping_keeps_clipping():
    X = np.random.default_rng(0).uniform(0, 1, (200, 2))
    probe = np.array([[1.05, 0.5]])
    default = TensorProductSplineTransformer().fit(X).transform(probe)
    partial = TensorProductSplineTransformer(policy={"constant": "warn"}).fit(X).transform(probe)
    np.testing.assert_array_equal(partial, default)


def test_policy_mapping_with_an_unknown_axis_is_rejected():
    from pretab.exceptions import InvalidParamError

    with pytest.raises(InvalidParamError):
        BSplineTransformer(policy={"missing": "error"}).fit(np.linspace(0, 1, 50).reshape(-1, 1)).transform([[0.5]])


@pytest.mark.parametrize("cls", [BSplineTransformer, PSplineTransformer])
@pytest.mark.parametrize("reaction", ["extrapolate", "warn"])
def test_extrapolated_bspline_rows_are_not_zeroed(cls, reaction):
    """Out-of-range rows were all zero under "extrapolate" / "warn": the basis is zero
    outside the knot span. They now extend the boundary pieces, a partition of unity."""
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    transformer = cls(policy=RepresentationPolicy(out_of_range=reaction)).fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataWarning)
        out = transformer.transform(np.array([[1.05], [-0.05], [0.5]]))
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    clipped = cls().fit(X).transform(np.array([[0.5]]))
    np.testing.assert_array_equal(out[2], clipped[0])  # in-range rows are unchanged


def test_extrapolated_mspline_and_tensor_rows_are_not_zeroed():
    policy = RepresentationPolicy(out_of_range="extrapolate")
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    assert (MSplineTransformer(output_dim=6, policy=policy).fit(X).transform([[1.05]]).sum() > 0).all()
    X2 = np.random.default_rng(0).uniform(0, 1, (200, 2))
    tensor = TensorProductSplineTransformer(policy=policy).fit(X2)
    np.testing.assert_allclose(tensor.transform([[1.05, 0.5]]).sum(), 1.0)
