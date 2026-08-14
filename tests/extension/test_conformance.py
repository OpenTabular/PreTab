"""Tests for the ``check_representation`` conformance suite (P10.3)."""

import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted

from pretab import BaseRepresentation, check_representation
from pretab.exceptions import RepresentationConformanceError


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
