"""Tests for the ``Preprocessor`` preset aliases and ``get_resolved_config`` (P10.5)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from pretab import Preprocessor, RepresentationConfig
from pretab.compose.registry import TRANSFORMER_REGISTRY
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


@pytest.mark.parametrize("form", ["string", "mapping", "value"])
def test_resolved_config_reports_effective_feature_settings(form):
    requested = {"string": "RBF", "mapping": {"method": "RBF"}, "value": RepresentationConfig(method="RBF")}[form]
    overrides = {"age": requested}
    processor = Preprocessor(
        preset="expanded",
        feature_preprocessing=overrides,
        target_aware=False,
        placement_strategy="quantile",
        random_state=17,
    )

    report = processor.get_resolved_config()
    feature = report["feature_preprocessing"]["age"]
    assert feature["method"] == "rbf"
    assert feature["output_dim"] == 10
    assert feature["target_aware"] is False
    assert feature["placement_strategy"] == "quantile"
    assert feature["adaptive"] is False
    assert feature["params"] == {"task": "regression", "random_state": 17}
    assert processor.get_params()["feature_preprocessing"] is overrides
    assert overrides["age"] is requested


@pytest.mark.parametrize("form", ["string", "mapping", "value"])
@pytest.mark.parametrize(("method", "canonical"), [("OHE", "one-hot"), ("identity", "none")])
def test_resolved_config_reports_categorical_declarations_without_numerical_controls(form, method, canonical):
    requested = {"string": method, "mapping": {"method": method}, "value": RepresentationConfig(method=method)}[form]
    processor = Preprocessor(preset="adaptive", feature_preprocessing={"city": requested}, degree=2)

    report = processor.get_resolved_config()

    assert set(report) == set(processor.get_params()) - {"preset"}
    assert report["feature_preprocessing"] == {"city": {"method": canonical, "params": {}, "name": None}}
    assert report["adaptive"] is True
    assert report["degree"] == 2


@pytest.mark.parametrize("adaptive", [False, True])
def test_resolved_config_preserves_local_precedence_and_meaningful_none(adaptive):
    settings = {"method": "RBF", "output_dim": 3, "params": {"gamma": 0.5, "random_state": None}, "name": "local"}
    if adaptive:
        settings.update(adaptive=True, min_output_dim=None, max_output_dim=6)
    report = Preprocessor(
        preset="adaptive",
        feature_preprocessing={"age": settings},
        target_aware=False,
        placement_strategy="quantile",
        random_state=17,
    ).get_resolved_config()
    feature = report["feature_preprocessing"]["age"]

    assert feature["method"] == "rbf"
    assert feature["output_dim"] == 3
    assert feature["adaptive"] is adaptive
    assert feature["name"] == "local"
    assert feature["params"] == {"gamma": 0.5, "task": "regression", "random_state": None}
    if adaptive:
        assert feature["min_output_dim"] is None
        assert feature["max_output_dim"] == 6
    else:
        assert "min_output_dim" not in feature
        assert "max_output_dim" not in feature
    assert report["adaptive"] is True
    assert report["random_state"] == 17


@pytest.mark.parametrize("overrides", [None, {}])
def test_resolved_config_without_declarations_keeps_shape_and_detaches_globals(overrides):
    policy = {"constant": "allow"}
    processor = Preprocessor(feature_preprocessing=overrides, policy=policy)
    report = processor.get_resolved_config()

    assert set(report) == set(processor.get_params()) - {"preset"}
    assert report["feature_preprocessing"] == overrides
    assert report["policy"] == policy
    assert report["policy"] is not policy
    report["policy"]["constant"] = "raise"
    assert policy == {"constant": "allow"}
    assert processor.get_resolved_config()["policy"] == policy


def test_resolved_config_detaches_nested_params_and_observes_later_raw_edits(monkeypatch):
    spec = TRANSFORMER_REGISTRY["rbf"]
    monkeypatch.setitem(TRANSFORMER_REGISTRY, "rbf", replace(spec, allowed_args=(*spec.allowed_args, "options")))
    nested = {"weights": [1, {"weight": 0.5}]}
    overrides = {"age": {"method": "rbf", "params": {"options": nested}}}
    processor = Preprocessor(feature_preprocessing=overrides, target_aware=False, placement_strategy="quantile")
    nested["weights"].append(2)
    first = processor.get_resolved_config()
    returned = first["feature_preprocessing"]["age"]["params"]["options"]
    assert returned == {"weights": [1, {"weight": 0.5}, 2]}
    nested["weights"].append(3)
    returned["weights"][1]["weight"] = 9.0
    returned["weights"].append(4)
    first["feature_preprocessing"]["age"]["method"] = "none"

    assert nested == {"weights": [1, {"weight": 0.5}, 2, 3]}
    second = processor.get_resolved_config()
    assert second["feature_preprocessing"]["age"]["method"] == "rbf"
    assert second["feature_preprocessing"]["age"]["params"]["options"] == nested
    assert processor.get_params()["feature_preprocessing"] is overrides


@pytest.mark.parametrize("frozen", [False, True])
def test_resolved_config_stays_live_without_refitting_or_reading_fitted_outputs(data, monkeypatch, frozen):
    frame, target = data
    overrides = {"a": {"method": "rbf", "output_dim": 3}}
    processor = Preprocessor(
        numerical_method="none",
        feature_preprocessing=overrides,
        target_aware=False,
        placement_strategy="quantile",
        scaling="none",
    ).fit(frame, target)
    if frozen:
        processor.freeze()
    original = processor.transform(frame, return_array=True)
    original_names = processor.get_feature_names_out()
    first = processor.get_resolved_config()
    overrides["a"]["output_dim"] = 5
    fingerprint = processor.fingerprint_

    def reject_execution(*args, **kwargs):
        raise AssertionError("Configuration reporting must not execute a fitted graph")

    with monkeypatch.context() as patcher:
        patcher.setattr(type(processor), "fit", reject_execution)
        patcher.setattr(type(processor), "transform", reject_execution)
        report = processor.get_resolved_config()

    assert first["feature_preprocessing"]["a"]["output_dim"] == 3
    assert report["feature_preprocessing"]["a"]["output_dim"] == 5
    assert set(report["feature_preprocessing"]) == {"a"}
    assert processor.column_transformer_.named_transformers_["num_a"].named_steps["rbf"].output_dim == 3
    assert processor.fingerprint_ == fingerprint
    np.testing.assert_array_equal(processor.transform(frame, return_array=True), original)
    np.testing.assert_array_equal(processor.get_feature_names_out(), original_names)
    overrides["a"]["unexpected"] = True
    with pytest.raises(InvalidParamError, match=r"feature_preprocessing\['a'\]\.params\['unexpected'\]"):
        processor.get_resolved_config()
    np.testing.assert_array_equal(processor.transform(frame, return_array=True), original)


@pytest.mark.parametrize(
    ("settings", "location"),
    [
        ({"method": "unknown"}, "method"),
        ({"method": "rff"}, "method"),
        ({"method": "rbf", "outpuut_dim": 3}, "outpuut_dim"),
        ({"method": "one-hot", "degree": 2}, "degree"),
        ({"method": "rbf", "target_aware": True, "placement_strategy": "uniform"}, "placement_strategy"),
    ],
)
def test_resolved_config_representation_errors_keep_feature_paths(settings, location):
    processor = Preprocessor(feature_preprocessing={"declared": settings})
    with pytest.raises(InvalidParamError) as failure:
        processor.get_resolved_config()
    assert "feature_preprocessing['declared']" in str(failure.value)
    assert location in str(failure.value)


@pytest.mark.parametrize("feature", ["a", "not_in_input"])
def test_resolved_config_does_not_infer_input_schema_or_feature_kind(data, feature):
    frame, target = data
    processor = Preprocessor(feature_preprocessing={feature: {"method": "OHE"}})
    report = processor.get_resolved_config()
    assert report["feature_preprocessing"] == {feature: {"method": "one-hot", "params": {}, "name": None}}
    with pytest.raises(InvalidParamError, match="feature_preprocessing"):
        processor.fit(frame, target)


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


@pytest.mark.parametrize("preset", [None, "standard", "expanded", "adaptive"])
def test_invalid_task_raises_instead_of_resolving(preset):
    """``task="Regression"`` used to resolve the classification preset method "ple"."""
    pre = Preprocessor(preset=preset, task="Regression")  # validated lazily, like fit
    with pytest.raises(InvalidParamError, match=r"Preprocessor\.task = 'Regression' is invalid"):
        pre.get_resolved_config()


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
