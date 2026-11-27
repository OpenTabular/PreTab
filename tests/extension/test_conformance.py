"""Tests for the ``check_representation`` conformance suite (P10.3)."""

import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted

from pretab import BaseRepresentation, PenaltySpec, check_representation, register_representation
from pretab.compose.registry import TRANSFORMER_REGISTRY
from pretab.exceptions import RepresentationConformanceError
from pretab.transformers import BSplineTransformer


class _Good(BaseRepresentation):
    representation_name = "good_conf"
    feature_kind = "numerical"

    def fit(self, X, y=None):
        self._validate(X, reset=True)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        return np.asarray(self._validate(X, reset=False), dtype=float) ** 2

    def _output_sizes(self):
        return [1] * self.n_features_in_


class _GoodSupervised(BaseRepresentation):
    representation_name = "good_sup_conf"
    supervision = "supervised"

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y is required")
        self._validate(X, reset=True)
        self.scale_ = float(np.mean(y)) or 1.0
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        return np.asarray(self._validate(X, reset=False), dtype=float) * self.scale_

    def _output_sizes(self):
        return [1] * self.n_features_in_


def test_good_representation_passes():
    passed = check_representation(_Good)
    assert "unfitted_transform_raises" in passed
    assert "fit_returns_self_no_mutation" in passed
    assert "deterministic" in passed
    assert "spec_consistent" in passed


def test_good_supervised_representation_passes():
    passed = check_representation(_GoodSupervised)
    assert "supervised_requires_y" in passed


_TYPED_PENALTY_METHODS = ("bspline", "cubicspline", "ispline", "mspline", "naturalspline", "pspline", "tensorspline")


def test_typed_penalty_registry_is_explicit():
    actual = {name for name, spec in TRANSFORMER_REGISTRY.items() if spec.typed_penalty}
    assert actual == set(_TYPED_PENALTY_METHODS)
    assert not TRANSFORMER_REGISTRY["tprs"].typed_penalty


@pytest.mark.parametrize("name", _TYPED_PENALTY_METHODS)
def test_typed_penalty_builtin_conformance(name):
    spec = TRANSFORMER_REGISTRY[name]
    assert "typed_penalty_consistent" in check_representation(spec.transformer_cls)
    if spec.arity == "univariate":
        source = np.linspace(-1, 1, 40)
        X = np.column_stack((source, 2 * source + 1))
        assert "typed_penalty_consistent" in check_representation(spec.transformer_cls, X=X)


def test_legacy_matrix_does_not_opt_in_to_typed_penalties():
    class _LegacyOnly(_Good):
        def get_penalty_matrix(self, feature_index=0):
            return np.eye(1)

    record = register_representation("legacy_only", _LegacyOnly)
    assert not record.typed_penalty
    assert "typed_penalty_consistent" not in check_representation(_LegacyOnly)
    register_representation("legacy_only", _LegacyOnly, override=True, typed_penalty=True)
    with pytest.raises(RepresentationConformanceError, match=r"get_penalty_spec.*required by typed_penalty"):
        check_representation(_LegacyOnly)


def test_multivariate_claim_requires_the_plural_accessor():
    class _MarginalOnly(_Good):
        scope = "multivariate"

        def get_penalty_spec(self, feature_index=0):
            raise AssertionError("must not try to qualify a marginal penalty")

        def get_penalty_matrix(self, feature_index=0):
            return np.eye(1)

    register_representation("marginal_only", _MarginalOnly, typed_penalty=True)
    with pytest.raises(RepresentationConformanceError, match=r"get_penalty_specs.*required by typed_penalty"):
        check_representation(_MarginalOnly)


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("unfitted", "before fit must raise PretabNotFittedError"),
        ("axis", "must identify fitted axis"),
        ("names", "coefficient names and indices must match fitted output"),
        ("matrix", "matrix must match get_penalty_matrix"),
    ],
)
def test_typed_penalty_conformance_rejects_mismatched_metadata(fault, message):
    class _IncorrectPenalty(BSplineTransformer):
        def get_penalty_spec(self, feature_index=0, *, diff_order=None):
            if fault == "unfitted" and not hasattr(self, "n_basis_"):
                raise ValueError("unexpected unfitted error")
            penalty = super().get_penalty_spec(feature_index, diff_order=diff_order)
            return PenaltySpec(
                matrix=2 * penalty.matrix if fault == "matrix" else penalty.matrix,
                kind=penalty.kind,
                order=penalty.order,
                axis=("wrong",) if fault == "axis" else penalty.axis,
                coordinate_system=penalty.coordinate_system,
                coefficient_names=penalty.coefficient_names[::-1] if fault == "names" else penalty.coefficient_names,
                coefficient_indices=penalty.coefficient_indices,
            )

    register_representation("incorrect_penalty", _IncorrectPenalty, typed_penalty=True)
    with pytest.raises(RepresentationConformanceError, match=message):
        check_representation(_IncorrectPenalty)


def test_fit_not_returning_self_fails():
    class _NoSelf(BaseRepresentation):
        representation_name = "noself_conf"

        def fit(self, X, y=None):
            self._validate(X, reset=True)  # returns None

        def transform(self, X):
            check_is_fitted(self, "n_features_in_")
            return np.asarray(self._validate(X, reset=False), dtype=float)

        def _output_sizes(self):
            return [1] * self.n_features_in_

    with pytest.raises(RepresentationConformanceError, match="must return self"):
        check_representation(_NoSelf)


def test_missing_unfitted_guard_fails():
    class _NoGuard(BaseRepresentation):
        representation_name = "noguard_conf"

        def fit(self, X, y=None):
            self._validate(X, reset=True)
            return self

        def transform(self, X):
            return np.asarray(X, dtype=float) ** 2

        def _output_sizes(self):
            return [1] * self.n_features_in_

    with pytest.raises(RepresentationConformanceError, match="NotFittedError"):
        check_representation(_NoGuard)


def test_feature_names_length_mismatch_fails():
    class _BadNames(BaseRepresentation):
        representation_name = "badnames_conf"

        def fit(self, X, y=None):
            self._validate(X, reset=True)
            return self

        def transform(self, X):
            check_is_fitted(self, "n_features_in_")
            return np.asarray(self._validate(X, reset=False), dtype=float)

        def get_feature_names_out(self, input_features=None):
            return np.array(["a", "b"])  # width is 1, so length 2 is wrong

        def _output_sizes(self):
            return [1] * self.n_features_in_

    with pytest.raises(RepresentationConformanceError, match="get_feature_names_out length"):
        check_representation(_BadNames)


def test_input_mutation_fails():
    class _Mutates(BaseRepresentation):
        representation_name = "mutates_conf"

        def fit(self, X, y=None):
            np.asarray(X)[:] = 0.0
            self._validate(X, reset=True)
            return self

        def transform(self, X):
            check_is_fitted(self, "n_features_in_")
            return np.asarray(self._validate(X, reset=False), dtype=float)

        def _output_sizes(self):
            return [1] * self.n_features_in_

    with pytest.raises(RepresentationConformanceError, match="must not mutate"):
        check_representation(_Mutates)


def test_supervised_that_ignores_y_fails():
    class _IgnoresY(BaseRepresentation):
        representation_name = "ignoresy_conf"
        supervision = "supervised"

        def fit(self, X, y=None):
            self._validate(X, reset=True)
            return self

        def transform(self, X):
            check_is_fitted(self, "n_features_in_")
            return np.asarray(self._validate(X, reset=False), dtype=float)

        def _output_sizes(self):
            return [1] * self.n_features_in_

    with pytest.raises(RepresentationConformanceError, match="fit succeeded without y"):
        check_representation(_IgnoresY)
