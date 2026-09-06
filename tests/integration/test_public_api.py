"""Public API surface contract for the top-level ``pretab`` package.

Guards the names third parties import and confirms the legacy ``pretab.pipeline``
package (folded into ``pretab.compose`` during the 1.0.0 restructure) is gone.
"""

import importlib

import pytest

import pretab

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
