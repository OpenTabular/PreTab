"""Value and validation contracts for coefficient-coordinate penalties."""

import pickle
from copy import copy, deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pretab import PenaltySpec
from pretab.exceptions import InvalidParamError
from pretab.expansion.spline import PenaltySpec as SplinePenaltySpec


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
