"""A1: ``LanguageEmbeddingTransformer`` keeps ``__init__`` pure and resolves the
embedding model lazily in ``fit``.

These tests avoid the optional ``sentence-transformers`` dependency by passing a
lightweight preloaded ``model`` (any object exposing ``encode``) or by simulating
the missing dependency with ``monkeypatch``.
"""

import sys
from typing import cast

import numpy as np
import pytest
from sklearn.base import clone

from pretab.core.exceptions import OptionalDependencyError, PretabConfigError
from pretab.transformers import LanguageEmbeddingTransformer


class _DummyModel:
    """Stand-in for a SentenceTransformer that records ``encode`` calls."""

    def __init__(self):
        self.calls = 0

    def encode(self, X, convert_to_numpy=True):
        self.calls += 1
        return np.zeros((len(X), 4))


def test_init_has_no_side_effects():
    # Construction must not import/load anything, even without the optional dep.
    transformer = LanguageEmbeddingTransformer()
    assert transformer.model_name == "paraphrase-MiniLM-L3-v2"
    assert transformer.model is None
    # No fitted model resolved yet.
    assert not hasattr(transformer, "model_")


def test_clone_preserves_params_without_loading():
    transformer = LanguageEmbeddingTransformer(model_name="custom-model")
    cloned = cast(LanguageEmbeddingTransformer, clone(transformer))
    assert cloned.model_name == "custom-model"
    assert cloned.model is None
    assert not hasattr(cloned, "model_")


def test_fit_uses_preloaded_model():
    dummy = _DummyModel()
    transformer = LanguageEmbeddingTransformer(model=dummy)
    transformer.fit(np.array([["a"], ["b"], ["c"]]))
    assert transformer.model_ is dummy
    assert transformer.n_features_in_ == 1


def test_fit_records_feature_count():
    dummy = _DummyModel()
    transformer = LanguageEmbeddingTransformer(model=dummy)
    transformer.fit(np.array([["a", "b"], ["c", "d"]]))
    assert transformer.n_features_in_ == 2


def test_transform_before_fit_raises():
    transformer = LanguageEmbeddingTransformer(model=_DummyModel())
    with pytest.raises(PretabConfigError):
        transformer.transform(np.array([["a"]]))


def test_fit_transform_invokes_model_encode():
    dummy = _DummyModel()
    transformer = LanguageEmbeddingTransformer(model=dummy)
    embeddings = transformer.fit_transform(np.array([["a"], ["b"], ["c"]]))
    assert embeddings.shape == (3, 4)
    assert dummy.calls == 1


def test_fit_without_dependency_raises(monkeypatch):
    # Simulate sentence-transformers being absent; construction still succeeds,
    # and only ``fit`` surfaces the optional-dependency error.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    transformer = LanguageEmbeddingTransformer()
    with pytest.raises(OptionalDependencyError):
        transformer.fit(np.array([["a"], ["b"]]))
