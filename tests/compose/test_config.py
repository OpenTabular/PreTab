"""Unit tests for :class:`pretab.compose.config.PreprocessorConfig`."""

import pytest

from pretab.exceptions import InvalidParamError


def test_none_method_normalizes_to_none(make_config):
    cfg = make_config(numerical_method=None, categorical_method=None)
    assert cfg.numerical_method == "none"
    assert cfg.categorical_method == "none"


def test_aliases_resolve_to_canonical(make_config):
    assert make_config(numerical_method="cubic").numerical_method == "cubicspline"
    assert make_config(categorical_method="ohe").categorical_method == "one-hot"


def test_invalid_placement_combo_raises(make_config):
    with pytest.raises(InvalidParamError):
        make_config(target_aware=True, placement_strategy="uniform")
    with pytest.raises(InvalidParamError):
        make_config(target_aware=False, placement_strategy="cart")


def test_feature_preprocessing_is_copied(make_config):
    fp = {"age": "standardization"}
    cfg = make_config(feature_preprocessing=fp)
    assert cfg.feature_preprocessing == fp
    assert cfg.feature_preprocessing is not fp


def test_none_feature_preprocessing_becomes_empty_dict(make_config):
    assert make_config(feature_preprocessing=None).feature_preprocessing == {}


def test_method_for_override_wins_over_global(make_config):
    cfg = make_config(numerical_method="standardization", feature_preprocessing={"age": "minmax"})
    assert cfg.method_for("age", is_numerical=True) == "minmax"
    assert cfg.method_for("height", is_numerical=True) == "standardization"


def test_method_for_resolves_in_the_requested_namespace(make_config):
    cfg = make_config(feature_preprocessing={"c": "ohe"})
    assert cfg.method_for("c", is_numerical=False) == "one-hot"


def test_seed_kwargs_reflects_random_state(make_config):
    assert make_config(random_state=None).seed_kwargs == {}
    assert make_config(random_state=42).seed_kwargs == {"random_state": 42}
