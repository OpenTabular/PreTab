"""Tests for the ``Preprocessor`` preset aliases and ``get_resolved_config`` (P10.5)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from pretab import Preprocessor
from pretab.exceptions import InvalidParamError


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "a": np.linspace(0, 3, 40),
            "b": np.linspace(-2, 2, 40),
            "c": ["x", "y"] * 20,
        }
    )
    y = np.linspace(0, 1, 40)
    return X, y


def test_no_preset_resolved_config_drops_preset_key():
    cfg = Preprocessor().get_resolved_config()
    assert "preset" not in cfg
    assert cfg["numerical_method"] == "ple"
    assert cfg["categorical_method"] == "int"


def test_standard_preset_matches_baseline():
    # Preprocessor()'s default task is "regression", so the preset resolves
    # numerical_method to "bspline" (see test_preset_numerical_method_follows_task).
    cfg = Preprocessor(preset="standard").get_resolved_config()
    assert cfg["numerical_method"] == "bspline"
    assert cfg["categorical_method"] == "int"
    assert cfg["output_dim"] == 7
    assert cfg["adaptive"] is False
    assert "preset" not in cfg


def test_preset_numerical_method_follows_task():
    """Every preset resolves numerical_method from task: bspline for regression,
    ple for classification. Explicitly setting numerical_method still wins."""
    for preset in ("standard", "expanded", "adaptive"):
        regression_cfg = Preprocessor(preset=preset, task="regression").get_resolved_config()
        classification_cfg = Preprocessor(preset=preset, task="classification").get_resolved_config()
        assert regression_cfg["numerical_method"] == "bspline"
        assert classification_cfg["numerical_method"] == "ple"

    explicit_cfg = Preprocessor(preset="standard", task="regression", numerical_method="rbf").get_resolved_config()
    assert explicit_cfg["numerical_method"] == "rbf"


def test_expanded_preset_widens_config():
    cfg = Preprocessor(preset="expanded").get_resolved_config()
    assert cfg["categorical_method"] == "one-hot"
    assert cfg["output_dim"] == 10
    assert cfg["adaptive"] is False


def test_adaptive_preset_enables_adaptive_width():
    cfg = Preprocessor(preset="adaptive").get_resolved_config()
    assert cfg["adaptive"] is True
    assert cfg["min_output_dim"] == 7
    assert cfg["max_output_dim"] == 15


def test_explicit_param_overrides_preset():
    cfg = Preprocessor(preset="expanded", output_dim=5).get_resolved_config()
    assert cfg["output_dim"] == 5  # user value wins over the preset's 10


def test_explicit_param_equal_to_constructor_default_overrides_preset():
    # Regression guard: an explicit value that happens to equal the ordinary
    # __init__ default must still win over the preset, not be mistaken for
    # "left unset".
    cfg = Preprocessor(preset="adaptive", adaptive=False).get_resolved_config()
    assert cfg["adaptive"] is False  # user explicitly said no, preset's True must not apply

    cfg2 = Preprocessor(preset="expanded", output_dim=7).get_resolved_config()
    assert cfg2["output_dim"] == 7  # user explicitly said 7, preset's 10 must not apply


def test_preset_is_preserved_by_get_params_and_clone():
    pre = Preprocessor(preset="standard")
    assert pre.get_params()["preset"] == "standard"
    cloned = clone(pre)
    assert isinstance(cloned, Preprocessor)
    assert cloned.get_params()["preset"] == "standard"


def test_invalid_preset_raises():
    with pytest.raises(InvalidParamError, match="preset"):
        Preprocessor(preset="nope").get_resolved_config()


def test_presets_fit_with_distinct_widths(data):
    X, y = data
    widths = {}
    for name in ("standard", "expanded", "adaptive"):
        pre = Preprocessor(preset=name)
        out = np.asarray(pre.fit_transform(X, y, return_array=True))
        assert out.shape[0] == X.shape[0]
        widths[name] = out.shape[1]
    # "expanded" one-hot encodes and widens the numerical basis, so it is wider
    # than the "standard" baseline.
    assert widths["expanded"] > widths["standard"]


def test_invalid_preset_raises_at_fit(data):
    X, y = data
    with pytest.raises(InvalidParamError, match="preset"):
        Preprocessor(preset="nope").fit(X, y)
