"""Tests for the public ``BaseRepresentation`` extension base (P10.1)."""

import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted

from pretab import BaseRepresentation, RepresentationSpec


class _Square(BaseRepresentation):
    representation_name = "square_proto"
    feature_kind = "numerical"
    scope = "univariate"
    supervision = "unsupervised"

    def fit(self, X, y=None):
        self._validate(X, reset=True)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        X = self._validate(X, reset=False)
        return np.asarray(X, dtype=float) ** 2

    def _output_sizes(self):
        return [1] * self.n_features_in_


def test_declared_metadata_syncs_internal_hooks():
    assert _Square._representation_family == "square_proto"
    assert _Square._representation_scope == "univariate"
    assert _Square._representation_supervision == "unsupervised"
    assert _Square._requires_y is False


def test_supervised_flag_sets_requires_y():
    class _Sup(BaseRepresentation):
        representation_name = "sup_proto"
        supervision = "supervised"

        def fit(self, X, y=None):
            self._validate(X, reset=True)
            return self

        def transform(self, X):
            check_is_fitted(self, "n_features_in_")
            return np.asarray(self._validate(X, reset=False), dtype=float)

        def _output_sizes(self):
            return [1] * self.n_features_in_

    assert _Sup._requires_y is True
    assert _Sup._representation_supervision == "supervised"


def test_representation_spec_reflects_declaration():
    X = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
    est = _Square().fit(X)
    spec = est.get_representation_spec()
    assert isinstance(spec, RepresentationSpec)
    assert spec.scope == "univariate"
    assert spec.output_dim == 1


def test_invalid_scope_rejected():
    with pytest.raises(ValueError, match="scope"):

        class _Bad(BaseRepresentation):
            scope = "triple"


def test_invalid_supervision_rejected():
    with pytest.raises(ValueError, match="supervision"):

        class _Bad(BaseRepresentation):
            supervision = "sometimes"


def test_invalid_feature_kind_rejected():
    with pytest.raises(ValueError, match="feature_kind"):

        class _Bad(BaseRepresentation):
            feature_kind = "ordinal"
