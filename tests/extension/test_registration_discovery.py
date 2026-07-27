"""Tests for representation registration, entry-point loading, and discovery.

Covers P10.2 (``register_representation`` + entry points) and P10.4
(``list_representations`` capability discovery).
"""

import importlib.metadata as importlib_metadata

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from pretab import (
    BaseRepresentation,
    Preprocessor,
    list_representations,
    load_entry_point_representations,
    register_representation,
)
from pretab.compose import registry
from pretab.exceptions import ConfigWarning


class _Square(BaseRepresentation):
    representation_name = "square_reg"
    feature_kind = "numerical"

    def fit(self, X, y=None):
        self._validate(X, reset=True)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        return np.asarray(self._validate(X, reset=False), dtype=float) ** 2

    def _output_sizes(self):
        return [1] * self.n_features_in_


class _CatPassthrough(BaseRepresentation):
    representation_name = "cat_reg"
    feature_kind = "categorical"

    def fit(self, X, y=None):
        self._validate(X, reset=True)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        return np.asarray(self._validate(X, reset=False), dtype=float)

    def _output_sizes(self):
        return [1] * self.n_features_in_


def test_register_makes_method_selectable_and_discoverable():
    register_representation("square_reg", _Square)
    assert "square_reg" in registry.NUMERICAL_METHODS
    assert "square_reg" in list_representations(feature_kind="numerical")


def test_register_end_to_end_through_preprocessor():
    register_representation("square_reg", _Square)
    X = pd.DataFrame({"a": np.linspace(0, 3, 12), "b": np.linspace(-2, 2, 12)})
    pre = Preprocessor(
        numerical_method="square_reg",
        categorical_method="none",
        target_aware=False,
        placement_strategy="uniform",
    )
    out = np.asarray(pre.fit_transform(X, return_array=True))
    assert out.shape == (12, 2)
    # The Preprocessor scales columns into [0, 1] before applying the method, so
    # the registered "square" method yields non-negative values bounded by 1.
    assert (out >= -1e-9).all()
    assert out.max() <= 1 + 1e-9


def test_register_categorical_updates_categorical_view():
    register_representation("cat_reg", _CatPassthrough)
    assert "cat_reg" in registry.CATEGORICAL_METHODS
    assert "cat_reg" in list_representations(feature_kind="categorical")


def test_duplicate_registration_requires_override():
    register_representation("square_reg", _Square)
    with pytest.raises(ValueError, match="already registered"):
        register_representation("square_reg", _Square)
    # override replaces without raising.
    register_representation("square_reg", _Square, override=True)


def test_register_validation_errors():
    with pytest.raises(ValueError, match="non-empty"):
        register_representation("", _Square)
    with pytest.raises(TypeError, match="cls must be a class"):
        register_representation("bad", _Square())
    with pytest.raises(ValueError, match="feature_kind"):
        register_representation("bad", _Square, feature_kind="ordinal")


def test_list_representations_capability_filters():
    assert list_representations(periodic=True) == ["fourier"]
    assert list_representations(sparse_output=True) == ["one-hot"]
    assert "ple" in list_representations(supervised=True)
    multivariate = list_representations(scope="multivariate")
    assert "tensorspline" in multivariate
    assert "fourier" not in multivariate
    categorical = list_representations(feature_kind="categorical")
    assert {"int", "one-hot"}.issubset(set(categorical))


def test_list_representations_include_optional_toggle():
    with_optional = list_representations(feature_kind="categorical")
    without_optional = list_representations(feature_kind="categorical", include_optional=False)
    assert "pretrained" in with_optional
    assert "pretrained" not in without_optional


def test_load_entry_point_representations_registers(monkeypatch):
    class _EpRep(BaseRepresentation):
        representation_name = "ep_square"
        feature_kind = "numerical"

        def fit(self, X, y=None):
            self._validate(X, reset=True)
            return self

        def transform(self, X):
            check_is_fitted(self, "n_features_in_")
            return np.asarray(self._validate(X, reset=False), dtype=float)

        def _output_sizes(self):
            return [1] * self.n_features_in_

    class _FakeEntryPoint:
        name = "ep_square_entry"

        def load(self):
            return _EpRep

    monkeypatch.setattr(
        importlib_metadata, "entry_points", lambda group=None: [_FakeEntryPoint()]
    )
    loaded = load_entry_point_representations()
    assert loaded == ["ep_square"]
    assert "ep_square" in registry.TRANSFORMER_REGISTRY


def test_load_entry_point_representations_skips_broken(monkeypatch):
    class _BrokenEntryPoint:
        name = "broken_entry"

        def load(self):
            raise ImportError("boom")

    monkeypatch.setattr(
        importlib_metadata, "entry_points", lambda group=None: [_BrokenEntryPoint()]
    )
    with pytest.warns(ConfigWarning, match="broken_entry"):
        loaded = load_entry_point_representations()
    assert loaded == []
