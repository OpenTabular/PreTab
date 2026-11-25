"""Public API surface contract for the top-level ``pretab`` package.

Guards the names third parties import and confirms the legacy ``pretab.pipeline``
package (folded into ``pretab.compose`` during the 1.0.0 restructure) is gone.

Update current API metadata explicitly, then review the JSON diff before committing::

    PRETAB_UPDATE_API_SNAPSHOT=1 pytest tests/integration/test_public_api.py -k test_public_api_snapshot
"""

import difflib
import importlib
import inspect
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import pretab
from pretab import compose as pretab_compose
from pretab import exceptions as pretab_exceptions
from pretab.compose import registry

API_SNAPSHOT_PATH = Path(__file__).parents[1] / "regression" / "_snapshots" / "public_api.json"


def _snapshot_value(value):
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, frozenset):
        return sorted(value)
    raise TypeError(f"Unsupported API snapshot value: {type(value).__name__}")


def _public_api_contract():
    transformers = importlib.import_module("pretab.transformers")
    callables = {
        f"pretab.{name}": member
        for name in pretab.__all__
        if callable(member := getattr(pretab, name))
        and not (isinstance(member, type) and issubclass(member, BaseException))
    }
    callables.update({f"pretab.transformers.{name}": getattr(transformers, name) for name in transformers.__all__})
    exception_classes = {
        name: member
        for name in pretab_exceptions.__all__
        if isinstance(member := getattr(pretab_exceptions, name), type) and issubclass(member, BaseException)
    }
    capabilities = [name for name, member in vars(registry.TransformerSpec).items() if isinstance(member, property)]
    contract = {
        "exports": {
            "pretab": sorted(pretab.__all__),
            "pretab.compose": sorted(pretab_compose.__all__),
            "pretab.transformers": sorted(transformers.__all__),
            "pretab.exceptions": sorted(pretab_exceptions.__all__),
            "pretab.compose.registry": sorted(registry.__all__),
        },
        "signatures": {name: str(inspect.signature(member)) for name, member in callables.items()},
        "registry": {
            name: {**asdict(spec), "derived_capabilities": {key: getattr(spec, key) for key in capabilities}}
            for name, spec in registry.TRANSFORMER_REGISTRY.items()
        },
        "aliases": {"numerical": registry.NUMERICAL_ALIASES, "categorical": registry.CATEGORICAL_ALIASES},
        "selectable_methods": {
            "numerical": sorted(registry.NUMERICAL_METHODS),
            "categorical": sorted(registry.CATEGORICAL_METHODS),
        },
        "exceptions": {
            name: {
                "class": member,
                "bases": member.__bases__,
                "kind": "warning" if issubclass(member, Warning) else "error",
            }
            for name, member in exception_classes.items()
        },
    }
    return json.loads(json.dumps(contract, default=_snapshot_value, allow_nan=False))


def _assert_api_snapshot(actual, path, *, update=False):
    rendered = json.dumps(actual, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if update:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        return
    assert path.is_file(), (
        f"Missing API snapshot: {path}. Create it explicitly with PRETAB_UPDATE_API_SNAPSHOT=1 "
        "pytest tests/integration/test_public_api.py -k test_public_api_snapshot, then review the diff."
    )
    expected = json.loads(path.read_text(encoding="utf-8"))
    if expected != actual:
        difference = "".join(
            difflib.unified_diff(
                (json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n").splitlines(keepends=True),
                rendered.splitlines(keepends=True),
                fromfile="reviewed public API",
                tofile="current public API",
            )
        )
        pytest.fail(
            f"Public API changed. Review compatibility before updating the snapshot.\n{difference}", pytrace=False
        )


def test_public_api_snapshot():
    _assert_api_snapshot(
        _public_api_contract(), API_SNAPSHOT_PATH, update=os.environ.get("PRETAB_UPDATE_API_SNAPSHOT") == "1"
    )


def test_api_snapshot_update_is_explicit(tmp_path):
    path = tmp_path / "snapshots" / "api.json"
    with pytest.raises(AssertionError, match="PRETAB_UPDATE_API_SNAPSHOT=1"):
        _assert_api_snapshot({"default": 7}, path)
    assert not path.exists()

    _assert_api_snapshot({"default": 7}, path, update=True)
    _assert_api_snapshot({"default": 7}, path)
    with pytest.raises(pytest.fail.Exception, match="Public API changed"):
        _assert_api_snapshot({"default": 8}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"default": 7}

    _assert_api_snapshot({"default": 8}, path, update=True)
    assert json.loads(path.read_text(encoding="utf-8")) == {"default": 8}


@pytest.mark.parametrize("surface", ["signature", "default", "registry", "alias", "exception", "export"])
def test_api_snapshot_rejects_surface_drift_without_rewriting(tmp_path, monkeypatch, surface):
    path = tmp_path / "api.json"
    _assert_api_snapshot(_public_api_contract(), path, update=True)
    reviewed = path.read_bytes()

    if surface in {"signature", "default"}:
        signature = inspect.signature(pretab.Preprocessor)
        parameters = list(signature.parameters.values())
        if surface == "signature":
            parameters.append(inspect.Parameter("contract_probe", inspect.Parameter.KEYWORD_ONLY, default=None))
        else:
            parameters = [
                parameter.replace(default=99) if parameter.name == "output_dim" else parameter
                for parameter in parameters
            ]
        monkeypatch.setattr(
            pretab.Preprocessor, "__signature__", signature.replace(parameters=parameters), raising=False
        )
    elif surface == "registry":
        original = registry.TRANSFORMER_REGISTRY["rbf"]
        monkeypatch.setitem(
            registry.TRANSFORMER_REGISTRY,
            "rbf",
            replace(original, supports_adaptive_resolution=not original.supports_adaptive_resolution),
        )
    elif surface == "alias":
        monkeypatch.setitem(registry.NUMERICAL_ALIASES, "std", "minmax")
    elif surface == "exception":
        changed = type("PretabDataError", (pretab_exceptions.PretabError,), {"__module__": pretab_exceptions.__name__})
        monkeypatch.setattr(pretab_exceptions, "PretabDataError", changed)
    else:
        monkeypatch.setattr(pretab, "__all__", [*pretab.__all__, "contract_probe"])
        monkeypatch.setattr(pretab, "contract_probe", pretab.Preprocessor, raising=False)

    with pytest.raises(pytest.fail.Exception) as failure:
        _assert_api_snapshot(_public_api_contract(), path)
    message = str(failure.value)
    assert "--- reviewed public API" in message
    assert "+++ current public API" in message
    assert "@@" in message
    assert path.read_bytes() == reviewed


# The stable public transformer facade. Every name here must stay importable from
# ``pretab.transformers`` no matter how the internal modules are relocated during the
# 1.0.0 layout refactor. Pinning the exact set makes an accidental drop or rename fail
# loudly instead of silently shrinking ``__all__``.
FACADE_TRANSFORMERS = frozenset(
    {
        "BSplineTransformer",
        "ContinuousOrdinalTransformer",
        "CubicRegressionSplineTransformer",
        "FourierFeatureTransformer",
        "ISplineTransformer",
        "LanguageEmbeddingTransformer",
        "MSplineTransformer",
        "MissingStateIndicator",
        "NaturalCubicSplineTransformer",
        "NoTransformer",
        "NumericBinningTransformer",
        "NystroemFeaturesTransformer",
        "OneHotFromOrdinalTransformer",
        "PLETransformer",
        "PSplineTransformer",
        "PeriodicEncodingTransformer",
        "RBFExpansionTransformer",
        "RandomFourierFeaturesTransformer",
        "ReLUExpansionTransformer",
        "SigmoidExpansionTransformer",
        "TanhExpansionTransformer",
        "TensorProductSplineTransformer",
        "ThinPlateSplineTransformer",
        "ToFloatTransformer",
    }
)


def test_public_names_are_exported():
    for name in ("Preprocessor", "PretabWarning", "configure_logging", "set_verbosity", "__version__"):
        assert hasattr(pretab, name)


def test_dunder_all_is_resolvable():
    assert pretab.__all__
    for name in pretab.__all__:
        assert hasattr(pretab, name)


def test_preprocessor_is_constructible():
    assert pretab.Preprocessor() is not None


def test_representation_config_public_exports_are_identical():
    from pretab.compose.config import RepresentationConfig

    assert pretab.RepresentationConfig is RepresentationConfig
    assert pretab_compose.RepresentationConfig is RepresentationConfig
    assert "RepresentationConfig" in pretab.__all__
    assert pretab_compose.__all__ == ["FeatureGroup", "RepresentationConfig"]


def test_feature_group_public_exports_are_identical():
    from pretab.compose.config import FeatureGroup

    assert pretab.FeatureGroup is FeatureGroup
    assert pretab_compose.FeatureGroup is FeatureGroup
    assert "FeatureGroup" in pretab.__all__


def test_transformers_public_surface_is_resolvable():
    transformers = importlib.import_module("pretab.transformers")
    assert transformers.__all__
    for name in transformers.__all__:
        assert hasattr(transformers, name)


def test_transformers_facade_is_frozen():
    transformers = importlib.import_module("pretab.transformers")
    assert set(transformers.__all__) == FACADE_TRANSFORMERS


@pytest.mark.parametrize("name", sorted(FACADE_TRANSFORMERS))
def test_facade_transformer_is_an_importable_class(name):
    transformers = importlib.import_module("pretab.transformers")
    assert isinstance(getattr(transformers, name), type)


def test_legacy_pipeline_package_is_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pretab.pipeline")


def test_compose_subsystem_is_importable():
    for module in ("config", "registry", "factory", "output", "inspection", "feature_detection"):
        importlib.import_module(f"pretab.compose.{module}")


def test_py_typed_marker_is_present():
    # PEP 561 marker: without this, type checkers treat an installed pretab as
    # untyped by default, losing the benefit of the project's own annotations.
    import pathlib

    pretab_dir = pathlib.Path(pretab.__file__).parent
    assert (pretab_dir / "py.typed").is_file()
