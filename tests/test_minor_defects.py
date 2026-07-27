"""Regression tests for the small defects collected in issue #21.

Each block is independent; they are grouped only because the fixes are small.
"""

import numpy as np
import pytest

from pretab.core.exceptions import InvalidParamError
from pretab.core.locations import resolve_locations
from pretab.transformers import BSplineTransformer


# --------------------------------------------------------------------------- #
# 1. resolve_locations must keep ``importance`` aligned with ``locations``.
#
# ``locs`` was sorted and de-duplicated while ``importance`` kept its original
# order, so the trim ranked the wrong entries.
# --------------------------------------------------------------------------- #
def test_resolve_locations_keeps_importance_aligned():
    locations = np.array([5.0, 1.0, 3.0])
    importance = np.array([0.1, 9.9, 0.2])  # 1.0 is by far the most important

    kept = resolve_locations(locations, min_count=1, max_count=1, importance=importance)

    np.testing.assert_allclose(kept, [1.0])


def test_resolve_locations_importance_survives_dedupe():
    locations = np.array([5.0, 5.0, 1.0, 3.0])
    importance = np.array([0.1, 0.1, 9.9, 0.2])

    kept = resolve_locations(locations, min_count=1, max_count=2, importance=importance)

    assert 1.0 in kept


def test_resolve_locations_rejects_mismatched_importance():
    with pytest.raises(ValueError, match="importance has"):
        resolve_locations(
            np.array([1.0, 2.0, 3.0]), min_count=1, max_count=1, importance=np.array([1.0])
        )


def test_resolve_locations_without_importance_is_unchanged():
    locations = np.array([5.0, 1.0, 3.0, 3.0])

    kept = resolve_locations(locations, min_count=1, max_count=3)

    np.testing.assert_allclose(kept, [1.0, 3.0, 5.0])


# --------------------------------------------------------------------------- #
# 2. get_feature_names_out must reject a wrong-length ``input_features``.
#
# ``zip(..., strict=False)`` truncated silently, returning a short array.
# --------------------------------------------------------------------------- #
def test_feature_names_out_rejects_too_few_input_features():
    transformer = BSplineTransformer(output_dim=6).fit(np.random.default_rng(0).random((20, 2)))

    with pytest.raises(InvalidParamError, match="input_features has 1 entries"):
        transformer.get_feature_names_out(["only_one"])


def test_feature_names_out_rejects_too_many_input_features():
    transformer = BSplineTransformer(output_dim=6).fit(np.random.default_rng(0).random((20, 1)))

    with pytest.raises(InvalidParamError, match="was fitted on 1 features"):
        transformer.get_feature_names_out(["a", "b"])


def test_feature_names_out_accepts_the_right_length():
    transformer = BSplineTransformer(output_dim=6).fit(np.random.default_rng(0).random((20, 2)))

    names = transformer.get_feature_names_out(["a", "b"])

    assert len(names) == transformer.transform(np.random.default_rng(1).random((5, 2))).shape[1]
    assert names[0].startswith("a_")
    assert names[-1].startswith("b_")


# --------------------------------------------------------------------------- #
# 3. LanguageEmbeddingTransformer.fit must accept a plain list.
#
# ``X.shape[1]`` raised AttributeError on a list, so the documented example
# could not run (it is marked ``# doctest: +SKIP``, hiding the breakage).
# --------------------------------------------------------------------------- #
def test_language_embedding_fit_accepts_a_list():
    from pretab.transformers import LanguageEmbeddingTransformer

    class _DummyModel:
        def encode(self, texts, convert_to_numpy=True):
            return np.ones((len(texts), 4))

    transformer = LanguageEmbeddingTransformer(model=_DummyModel())
    transformer.fit([["red"], ["blue"], ["green"]])

    assert transformer.n_features_in_ == 1
    assert transformer.transform([["red"], ["blue"], ["green"]]).shape == (3, 4)


def test_language_embedding_fit_still_accepts_arrays():
    from pretab.transformers import LanguageEmbeddingTransformer

    class _DummyModel:
        def encode(self, texts, convert_to_numpy=True):
            return np.ones((len(texts), 2))

    transformer = LanguageEmbeddingTransformer(model=_DummyModel())
    transformer.fit(np.array([["a", "x"], ["b", "y"]], dtype=object))

    assert transformer.n_features_in_ == 2
