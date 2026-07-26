"""Public API surface contract for the top-level ``pretab`` package.

Guards the names third parties import and confirms the legacy ``pretab.pipeline``
package (folded into ``pretab.compose`` during the 1.0.0 restructure) is gone.
"""

import importlib

import pytest

import pretab


def test_public_names_are_exported():
    for name in ("Preprocessor", "PretabWarning", "configure_logging", "set_verbosity", "__version__"):
        assert hasattr(pretab, name)


def test_dunder_all_is_resolvable():
    assert pretab.__all__
    for name in pretab.__all__:
        assert hasattr(pretab, name)


def test_preprocessor_is_constructible():
    assert pretab.Preprocessor() is not None


def test_transformers_public_surface_is_resolvable():
    transformers = importlib.import_module("pretab.transformers")
    assert transformers.__all__
    for name in transformers.__all__:
        assert hasattr(transformers, name)


def test_legacy_pipeline_package_is_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pretab.pipeline")


def test_compose_subsystem_is_importable():
    for module in ("config", "registry", "factory", "output", "inspection", "feature_detection"):
        importlib.import_module(f"pretab.compose.{module}")
