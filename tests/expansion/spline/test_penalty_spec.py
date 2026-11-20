"""Value and validation contracts for coefficient-coordinate penalties."""

import pickle
from copy import copy, deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from pretab import PenaltySpec
from pretab.exceptions import InvalidParamError, PretabNotFittedError
from pretab.expansion.spline import PenaltySpec as SplinePenaltySpec
from pretab.transformers import (
    BSplineTransformer,
    CubicRegressionSplineTransformer,
    ISplineTransformer,
    MSplineTransformer,
    NaturalCubicSplineTransformer,
    PSplineTransformer,
    TensorProductSplineTransformer,
)


def _spec(matrix, **overrides):
    settings = {
        "kind": "coefficient_difference",
        "order": 1,
        "axis": ["age"],
        "coordinate_system": "output_coefficients",
        "coefficient_names": ["age_0", "age_1"],
        "coefficient_indices": [2, 3],
    }
    return PenaltySpec(matrix, **(settings | overrides))


def test_penalty_is_detached_immutable_and_has_derived_rank():
    original = np.array([[1, -1], [-1, 1]], dtype=np.int16)
    provenance = {"options": [{"bias": False}]}
    spec = _spec(original, provenance=provenance)
    original[:] = 0
    provenance["options"][0]["bias"] = True

    np.testing.assert_array_equal(spec.matrix, [[1.0, -1.0], [-1.0, 1.0]])
    assert spec.matrix.dtype == np.float64
    assert (spec.rank, spec.nullity) == (1, 1)
    assert spec.coefficient_indices == (2, 3)
    assert spec.provenance["options"][0]["bias"] is False
    with pytest.raises(FrozenInstanceError):
        setattr(spec, "rank", 2)  # noqa: B010
    with pytest.raises(ValueError):
        spec.matrix.setflags(write=True)
    with pytest.raises(TypeError):
        spec.provenance["options"][0]["bias"] = True
    assert deepcopy(spec) == spec


def test_material_negative_direction_is_rejected():
    with pytest.raises(InvalidParamError, match="positive semidefinite"):
        _spec([[1.0, 0.0], [0.0, -0.5]])


@pytest.mark.parametrize("restore", [copy, deepcopy, lambda spec: pickle.loads(pickle.dumps(spec))])  # noqa: S301
def test_restored_values_retain_immutable_storage(restore):
    spec = _spec(np.eye(2), provenance={"axes": ["age"], "options": {"bias": False}})
    restored = restore(spec)
    assert restored == spec
    assert restored != _spec(np.eye(2))
    with pytest.raises(ValueError):
        restored.matrix.setflags(write=True)
    with pytest.raises(ValueError):
        restored.matrix.base.setflags(write=True)
    with pytest.raises(TypeError):
        restored.provenance["options"]["bias"] = True


def test_public_export_and_ordered_metadata():
    names = ["joint_1", "joint_0"]
    indices = [np.int64(7), np.int64(3)]
    axes = ["north", "east"]
    spec = _spec(
        np.eye(2),
        coefficient_names=names,
        coefficient_indices=indices,
        axis=axes,
        kind="quadratic_form",
        order=None,
    )
    names.reverse()
    indices.reverse()
    axes.reverse()
    assert type(spec) is SplinePenaltySpec is PenaltySpec
    assert spec.coefficient_names == ("joint_1", "joint_0")
    assert spec.coefficient_indices == (7, 3)
    assert spec.axis == ("north", "east")
    assert spec.order is None
    assert spec != _spec(np.eye(2))
    with pytest.raises(TypeError):
        hash(spec)
    assert spec.matrix is not spec.matrix
    assert spec.matrix.base is not spec.matrix.base
    assert spec.matrix.shape == (2, 2)


@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e308])
def test_relative_tolerance_preserves_values_and_scales_rank(scale):
    matrix = scale * np.array([[1.0, 1e-12], [0.0, -1e-12]])
    spec = _spec(matrix, atol=0.0, rtol=1e-10)
    np.testing.assert_array_equal(spec.matrix, matrix)
    assert (spec.rank, spec.nullity) == (1, 1)
    with pytest.raises(InvalidParamError, match="symmetric"):
        _spec(matrix, atol=0.0, rtol=0.0)
    with pytest.raises(InvalidParamError, match="positive semidefinite"):
        _spec(np.diag([scale, -scale * 1e-12]), atol=0.0, rtol=0.0)


def test_absolute_tolerance_includes_boundary_and_rank_is_strict():
    spec = _spec(np.diag([0.25, -0.25]), atol=0.25, rtol=0.0)
    assert (spec.rank, spec.nullity) == (0, 2)
    assert (spec.atol, spec.rtol) == (0.25, 0.0)
    np.testing.assert_array_equal(spec.matrix, np.diag([0.25, -0.25]))
    zero = _spec(np.zeros((2, 2)), atol=0.0, rtol=0.0)
    assert (zero.rank, zero.nullity) == (0, 2)


@pytest.mark.parametrize(
    "matrix",
    [
        [],
        [1, 2],
        [[1], [1]],
        [[1, 2], [3]],
        np.eye(2, dtype=complex),
        np.eye(2, dtype=bool),
        [["1", "0"], ["0", "1"]],
        [[1.0, np.nan], [0.0, 1.0]],
        [[np.inf, 0], [0, 1]],
    ],
)
def test_invalid_matrices_raise_typed_errors(matrix):
    with pytest.raises(InvalidParamError, match="matrix"):
        _spec(matrix)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("kind", " "),
        ("order", True),
        ("order", np.bool_(True)),
        ("order", 0),
        ("order", 1.5),
        ("order", None),
        ("axis", "east"),
        ("axis", []),
        ("axis", ["east", "east"]),
        ("axis", [3]),
        ("coordinate_system", ""),
        ("coefficient_names", ["a"]),
        ("coefficient_names", ["a", "a"]),
        ("coefficient_names", ["a", ""]),
        ("coefficient_indices", [0]),
        ("coefficient_indices", [1, 1]),
        ("coefficient_indices", [0, -1]),
        ("coefficient_indices", [False, 1]),
        ("coefficient_indices", [0, 1.0]),
        ("atol", -1),
        ("atol", np.inf),
        ("rtol", np.nan),
        ("rtol", True),
        ("provenance", []),
        ("provenance", {1: "x"}),
        ("provenance", {"x": {1}}),
        ("provenance", {"x": [np.inf]}),
    ],
)
def test_invalid_metadata_raises_typed_errors(name, value):
    with pytest.raises(InvalidParamError, match=name):
        _spec(np.eye(2), **{name: value})


def test_provenance_rejects_cycles_but_accepts_shared_nested_values():
    nested: list[object] = ["age"]
    assert _spec(np.eye(2), provenance={"a": nested, "b": nested}).provenance == {"a": ("age",), "b": ("age",)}
    nested.append(nested)
    with pytest.raises(InvalidParamError, match="circular"):
        _spec(np.eye(2), provenance={"axes": nested})


@pytest.mark.parametrize("transformer_cls", [BSplineTransformer, MSplineTransformer, ISplineTransformer])
@pytest.mark.parametrize("include_bias", [False, True])
@pytest.mark.parametrize("diff_order", [None, np.int64(1), 2])
def test_base_typed_accessor_matches_legacy_and_named_output_span(transformer_cls, include_bias, diff_order):
    source = np.linspace(0, 1, 21)
    frame = pd.DataFrame({"left": source, "right": source**2})
    spline = transformer_cls(output_dim=6, include_bias=include_bias).fit(frame)
    spec = spline.get_penalty_spec(np.int64(1), diff_order=diff_order)
    width = len(spline.get_feature_names_out()) // 2
    order = 2 if diff_order is None else int(diff_order)
    legacy = spline.get_penalty_matrix(feature_index=1, diff_order=order)

    np.testing.assert_array_equal(spec.matrix, legacy)
    assert not spec.matrix.flags.writeable
    assert (spec.kind, spec.order, spec.axis) == ("coefficient_difference", order, ("right",))
    assert spec.coordinate_system == "output_coefficients"
    assert spec.coefficient_indices == tuple(range(width, 2 * width))
    assert spec.coefficient_names == tuple(spline.get_feature_names_out()[width:])
    assert spec.matrix.shape == (width, width)
    assert (spec.rank, spec.nullity) == (width - order - int(include_bias), order + int(include_bias))
    if include_bias:
        np.testing.assert_array_equal(spec.matrix[0], np.zeros(width))


def test_base_typed_accessor_requires_fit_and_uses_synthetic_names_for_arrays():
    spline = BSplineTransformer(output_dim=6)
    with pytest.raises(PretabNotFittedError, match="must be fitted"):
        spline.get_penalty_spec()
    spline.fit(np.linspace(0, 1, 21).reshape(-1, 1))
    spec = spline.get_penalty_spec()
    assert spec.axis == ("x0",)
    np.testing.assert_array_equal(spec.matrix, spline.get_penalty_matrix())


@pytest.mark.parametrize("feature_index", [-1, 1, True, np.bool_(False), 0.0, "0"])
def test_base_typed_accessor_rejects_invalid_indices(feature_index):
    spline = BSplineTransformer(output_dim=6).fit(np.linspace(0, 1, 21).reshape(-1, 1))
    with pytest.raises(InvalidParamError, match="feature_index"):
        spline.get_penalty_spec(feature_index)


@pytest.mark.parametrize("diff_order", [0, -1, 6, 7, True, np.bool_(False), 1.0, "1"])
def test_base_typed_accessor_rejects_invalid_explicit_orders(diff_order):
    spline = BSplineTransformer(output_dim=6, include_bias=True).fit(np.linspace(0, 1, 21).reshape(-1, 1))
    with pytest.raises(InvalidParamError, match="diff_order"):
        spline.get_penalty_spec(diff_order=diff_order)


def test_typed_validation_does_not_change_legacy_validation():
    spline = BSplineTransformer(output_dim=6).fit(np.linspace(0, 1, 21).reshape(-1, 1))
    np.testing.assert_array_equal(spline.get_penalty_matrix(-1, 0), np.eye(6))
    with pytest.raises(InvalidParamError, match="feature_index"):
        spline.get_penalty_spec(-1, diff_order=0)
    with pytest.raises(InvalidParamError, match="diff_order"):
        spline.get_penalty_spec(diff_order=0)


def test_base_typed_accessor_does_not_infer_an_overridden_convention():
    class OtherPenaltySpline(BSplineTransformer):
        def get_penalty_matrix(self, feature_index: int = 0, diff_order: int = 2):
            return np.eye(self.n_basis_[feature_index])

    spline = OtherPenaltySpline(output_dim=6)
    with pytest.raises(PretabNotFittedError, match="OtherPenaltySpline"):
        spline.get_penalty_spec()
    spline.fit(np.linspace(0, 1, 21).reshape(-1, 1))
    with pytest.raises(InvalidParamError, match=r"not supported; use get_penalty_matrix\(\)"):
        spline.get_penalty_spec()
    np.testing.assert_array_equal(spline.get_penalty_matrix(), np.eye(6))


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("include_bias", [False, True])
@pytest.mark.parametrize("index", [0, 1])
def test_pspline_typed_penalty_matches_fitted_matrix_and_output(order, include_bias, index):
    source = np.linspace(0, 1, 21)
    frame = pd.DataFrame({"left": source, "right": source**2})
    spline = PSplineTransformer(output_dim=6, diff_order=order, include_bias=include_bias).fit(frame)
    spec = spline.get_penalty_spec(index)
    assert spec == spline.get_penalty_spec(np.int64(index), diff_order=np.int64(order))
    width = 6 + int(include_bias)
    span = slice(index * width, (index + 1) * width)
    difference = np.diff(np.eye(6), n=order, axis=0)
    if include_bias:
        difference = np.pad(difference, ((0, 0), (1, 0)))
    legacy = spline.get_penalty_matrix(index)
    np.testing.assert_array_equal(spec.matrix, legacy)
    np.testing.assert_array_equal(spec.matrix, difference.T @ difference)
    assert not np.shares_memory(spec.matrix, legacy)
    assert (spec.kind, spec.order, spec.axis) == ("coefficient_difference", order, (frame.columns[index],))
    assert spec.coordinate_system == "output_coefficients"
    assert spec.coefficient_indices == tuple(range(span.start, span.stop))
    assert spec.coefficient_names == tuple(spline.get_feature_names_out()[span])
    assert spline.transform(frame)[:, span].shape[1] == spec.matrix.shape[0]
    assert (spec.rank, spec.nullity) == (6 - order, order + int(include_bias))
    assert spec.provenance["include_bias"] is include_bias
    coefficients = np.arange(width, dtype=float) ** 2
    assert coefficients @ spec.matrix @ coefficients == pytest.approx(np.linalg.norm(difference @ coefficients) ** 2)


def test_pspline_typed_penalty_uses_fit_order_until_refit():
    source = np.linspace(0, 1, 21).reshape(-1, 1)
    spline = PSplineTransformer(diff_order=1)
    with pytest.raises(PretabNotFittedError):
        spline.get_penalty_spec()
    spline.fit(source)
    original = spline.get_penalty_spec()
    spline.set_params(diff_order=2)
    assert original.axis == ("x0",)
    assert spline.get_penalty_spec() == original
    with pytest.raises(InvalidParamError, match=r"match the fitted order \(1\)"):
        spline.get_penalty_spec(diff_order=2)
    spline.fit(source)
    assert spline.get_penalty_spec().order == 2


@pytest.mark.parametrize("diff_order", [0, 2, True, np.bool_(True), 1.0, "1"])
def test_pspline_typed_penalty_rejects_an_invalid_explicit_order(diff_order):
    spline = PSplineTransformer(diff_order=1).fit(np.linspace(0, 1, 21).reshape(-1, 1))
    with pytest.raises(InvalidParamError, match="must match the fitted order"):
        spline.get_penalty_spec(diff_order=diff_order)


@pytest.mark.parametrize("feature_index", [-1, 1, False, 0.0])
def test_pspline_typed_penalty_rejects_invalid_indices(feature_index):
    spline = PSplineTransformer().fit(np.linspace(0, 1, 21).reshape(-1, 1))
    with pytest.raises(InvalidParamError, match="feature_index"):
        spline.get_penalty_spec(feature_index)


def test_pspline_typed_penalty_refuses_unqualified_higher_orders():
    spline = PSplineTransformer(diff_order=3).fit(np.linspace(0, 1, 21).reshape(-1, 1))
    with pytest.raises(InvalidParamError, match="support fitted orders 1 and 2"):
        spline.get_penalty_spec()
    assert np.linalg.matrix_rank(spline.get_penalty_matrix()) == 3


def test_pspline_typed_penalty_replay_requires_fitted_metadata():
    spline = PSplineTransformer(diff_order=1, include_bias=True).fit(np.linspace(0, 1, 21).reshape(-1, 1))
    expected = spline.get_penalty_spec()
    restored = pickle.loads(pickle.dumps(spline))  # noqa: S301
    assert restored.get_penalty_spec() == expected
    del restored._penalty_fit_
    with pytest.raises(InvalidParamError, match="refit to record the fitted penalty order"):
        restored.get_penalty_spec()
    np.testing.assert_array_equal(restored.get_penalty_matrix(), expected.matrix)


def test_pspline_typed_penalty_refuses_state_after_failed_refit():
    source = np.linspace(0, 1, 21).reshape(-1, 1)
    spline = PSplineTransformer(diff_order=1).fit(source)
    spline.set_params(diff_order=20)
    with pytest.raises(InvalidParamError, match="too large"):
        spline.fit(source)
    with pytest.raises(InvalidParamError, match="refit to record the fitted penalty order"):
        spline.get_penalty_spec()


@pytest.mark.parametrize("include_bias", [False, True])
def test_natural_cubic_typed_penalty_matches_approximate_curvature_in_output_coordinates(include_bias):
    source = np.linspace(-1, 2, 101)
    frame = pd.DataFrame({"left": source, "right": source * 2 + 3})
    spline = NaturalCubicSplineTransformer(output_dim=5, include_bias=include_bias).fit(frame)
    spec = spline.get_penalty_spec(1)
    width = 5 + int(include_bias)
    assert spec.kind == "approximate_integrated_squared_second_derivative"
    assert spec.order == 2
    assert spec.axis == ("right",)
    assert spec.coordinate_system == "output_coefficients"
    assert spec.coefficient_indices == tuple(range(width, 2 * width))
    assert spec.coefficient_names == tuple(spline.get_feature_names_out()[width:])
    assert spline.transform(frame).shape[1] == 2 * width
    np.testing.assert_array_equal(spec.matrix, spline.get_penalty_matrix(1))
    offset = 1 + int(include_bias)
    np.testing.assert_array_equal(spec.matrix[:offset], 0)
    assert (spec.rank, spec.nullity) == (4, offset)

    knots = spline.knots_[1]
    grid = np.linspace(knots[0], knots[-1], 4001)
    tails = np.maximum(grid[:, None] - knots, 0)
    left_weights = (knots[-1] - knots[1:-1]) / (knots[-1] - knots[0])
    curvature = 6 * (tails[:, 1:-1] - left_weights * tails[:, :1])
    exact_gram = (curvature[:-1].T @ curvature[:-1] + curvature[1:].T @ curvature[1:]) * (grid[1] - grid[0]) / 2
    np.testing.assert_allclose(spec.matrix[offset:, offset:], exact_gram, rtol=0.02)
    np.testing.assert_allclose(spec.matrix, 8 * spline.get_penalty_spec(0).matrix, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("include_bias", [False, True])
def test_cubic_regression_typed_penalty_matches_integrated_curvature_in_output_coordinates(include_bias):
    source = np.linspace(-1, 2, 101)
    frame = pd.DataFrame({"left": source, "right": source * 2 + 3})
    spline = CubicRegressionSplineTransformer(output_dim=5, include_bias=include_bias).fit(frame)
    spec = spline.get_penalty_spec(np.int64(1))
    width = 5 + int(include_bias)
    offset = 1 + int(include_bias)
    assert spec.kind == "integrated_squared_second_derivative"
    assert spec.order == 2
    assert spec.axis == ("right",)
    assert spec.coordinate_system == "output_coefficients"
    assert spec.coefficient_indices == tuple(range(width, 2 * width))
    assert spec.coefficient_names == tuple(spline.get_feature_names_out()[width:])
    assert spline.transform(frame).shape[1] == 2 * width
    np.testing.assert_array_equal(spec.matrix, spline.get_penalty_matrix(1))
    np.testing.assert_array_equal(spec.matrix[:offset], 0)
    assert (spec.rank, spec.nullity) == (4, offset)

    knots = spline.knots_[1]
    bounds = np.r_[frame["right"].min(), knots, frame["right"].max()]
    half_widths = np.diff(bounds) / 2
    midpoints = (bounds[1:] + bounds[:-1]) / 2
    nodes, weights = np.polynomial.legendre.leggauss(2)
    grid = (midpoints[:, None] + half_widths[:, None] * nodes).ravel()
    weights = (half_widths[:, None] * weights).ravel()
    curvature = np.column_stack((np.full_like(grid, 2), 6 * grid, 6 * np.maximum(grid[:, None] - knots, 0)))
    np.testing.assert_allclose(spec.matrix[offset:, offset:], curvature.T @ (weights[:, None] * curvature), rtol=1e-12)
    assert spec.matrix[offset, offset] == pytest.approx(2 * spline.get_penalty_spec(0).matrix[offset, offset])


@pytest.mark.parametrize("transformer_cls", [NaturalCubicSplineTransformer, CubicRegressionSplineTransformer])
def test_curvature_typed_penalty_validation_and_legacy_access(transformer_cls):
    spline = transformer_cls(output_dim=5)
    with pytest.raises(PretabNotFittedError, match="must be fitted"):
        spline.get_penalty_spec()
    spline.fit(np.linspace(0, 1, 21).reshape(-1, 1))
    spec = spline.get_penalty_spec()
    assert spec.axis == ("x0",)
    with pytest.raises(ValueError):
        spec.matrix.setflags(write=True)
    legacy = spline.get_penalty_matrix(-1)
    np.testing.assert_array_equal(spec.matrix, legacy)
    with pytest.raises(InvalidParamError, match="feature_index"):
        spline.get_penalty_spec(-1)
    with pytest.raises(InvalidParamError, match="feature_index"):
        spline.get_penalty_spec(True)
    with pytest.raises(InvalidParamError, match="not a difference penalty"):
        spline.get_penalty_spec(diff_order=2)
    spline.set_params(include_bias=True)
    with pytest.raises(InvalidParamError, match="refit"):
        spline.get_penalty_spec()


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("include_bias", [False, True])
def test_bspline_typed_difference_penalty_invariant_to_input_rescaling(order, include_bias):
    source = np.linspace(-1, 2, 101).reshape(-1, 1)
    spline = BSplineTransformer(output_dim=6, include_bias=include_bias).fit(source)
    penalty = spline.get_penalty_spec(diff_order=order)
    spline.fit(2 * source + 3)
    assert spline.get_penalty_spec(diff_order=order) == penalty
    assert penalty.kind == "coefficient_difference"
    assert (penalty.rank, penalty.nullity) == (6 - order, order + int(include_bias))
    coefficients = np.arange(6 + int(include_bias), dtype=float) ** 2
    energy = np.sum(np.diff(coefficients[int(include_bias) :], n=order) ** 2)
    assert coefficients @ penalty.matrix @ coefficients == energy


@pytest.mark.parametrize("axes", [("north", "east"), ("depth", "north", "east")])
@pytest.mark.parametrize("include_bias", [False, True])
@pytest.mark.parametrize("order", [1, 2])
def test_tensor_typed_penalties_match_joint_directions(axes, include_bias, order):
    source = np.linspace(0, 1, 31)
    frame = pd.DataFrame({axis: source**power for power, axis in enumerate(axes, 1)})
    spline = TensorProductSplineTransformer(output_dim=3, degree=2, include_bias=include_bias, diff_order=order).fit(
        frame
    )
    specs = spline.get_penalty_specs()
    legacy = spline.get_penalty_matrices()
    bias = int(include_bias)
    width = 3 + bias
    sizes = (width,) * len(axes)
    coefficients = np.random.default_rng(42).normal(size=sizes)
    vector = coefficients.ravel()
    names = tuple(
        "tp_" + "_".join(f"{axis}{index}" for axis, index in zip(axes, indices, strict=True))
        for indices in np.ndindex(*sizes)
    )
    assert tuple(spline.get_feature_names_out()) == names
    assert spline.transform(frame).shape[1] == len(names)
    assert isinstance(specs, tuple)
    assert [spec.axis for spec in specs] == [(axis,) for axis in axes]
    for direction, (spec, matrix) in enumerate(zip(specs, legacy, strict=True)):
        np.testing.assert_array_equal(spec.matrix, matrix)
        assert not spec.matrix.flags.writeable
        assert (spec.kind, spec.order, spec.coordinate_system) == (
            "coefficient_difference",
            order,
            "output_coefficients",
        )
        assert spec.coefficient_indices == tuple(range(len(names)))
        assert spec.coefficient_names == names
        assert spec.provenance["marginal_sizes"] == sizes
        assert spec.provenance["include_bias"] is include_bias
        assert spline.get_penalty_matrix(direction).shape == (width, width)
        direction_coefficients = np.moveaxis(coefficients, direction, 0)[bias:]
        energy = np.sum(np.diff(direction_coefficients, n=order, axis=0) ** 2)
        assert vector @ spec.matrix @ vector == pytest.approx(energy)
        repeats = width ** (len(axes) - 1)
        assert (spec.rank, spec.nullity) == ((3 - order) * repeats, (order + bias) * repeats)


def test_tensor_typed_penalties_use_fit_settings_and_refuse_missing_metadata():
    spline = TensorProductSplineTransformer(diff_order=1)
    with pytest.raises(PretabNotFittedError, match="must be fitted"):
        spline.get_penalty_specs()
    spline.fit(np.arange(32).reshape(-1, 2))
    expected = spline.get_penalty_specs()
    assert [spec.axis for spec in expected] == [("x0",), ("x1",)]
    spline.set_params(diff_order=2)
    assert spline.get_penalty_specs() == expected
    spline.set_params(include_bias=True)
    with pytest.raises(InvalidParamError, match="refit after changing"):
        spline.get_penalty_specs()
    spline.set_params(include_bias=False)
    del spline._penalty_fit_
    with pytest.raises(InvalidParamError, match="refit to record"):
        spline.get_penalty_specs()
    np.testing.assert_array_equal(spline.get_penalty_matrices()[0], expected[0].matrix)
