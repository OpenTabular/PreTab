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
from pretab.transformers import BSplineTransformer, ISplineTransformer, MSplineTransformer


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
