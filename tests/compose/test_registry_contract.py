"""Contract tests driven by ``TRANSFORMER_REGISTRY``.

Every registered method is validated for a consistent capability record and for
behaviour that matches its declared flags. Adding a method to the registry
therefore automatically subjects it to these invariants.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from pretab import Preprocessor
from pretab.compose.registry import (
    NUMERICAL_METHODS,
    TRANSFORMER_REGISTRY,
    TransformerSpec,
    categorical_method_names,
    numerical_method_names,
)
from pretab.exceptions import InvalidParamError, OptionalDependencyError, PretabError

_VALID_KINDS = {"numerical", "categorical"}
_VALID_ARITY = {"univariate", "multivariate"}
_VALID_TARGET_USAGE = {"forbidden", "optional", "required"}
_UNSUPERVISED = frozenset({"uniform", "quantile"})
_TARGET_AWARE = frozenset({"cart", "lightgbm"})
_ALL_STRATEGIES = _UNSUPERVISED | _TARGET_AWARE

# Optional extra -> importable module used to detect whether the dependency is
# actually installed in the current environment.
_EXTRA_MODULE = {"embeddings": "sentence_transformers", "lightgbm": "lightgbm"}

_SPEC_ITEMS = list(TRANSFORMER_REGISTRY.items())
_SPEC_IDS = [name for name, _ in _SPEC_ITEMS]


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize("name, spec", _SPEC_ITEMS, ids=_SPEC_IDS)
def test_registry_key_matches_name(name, spec):
    assert isinstance(spec, TransformerSpec)
    assert spec.name == name


@pytest.mark.parametrize("name, spec", _SPEC_ITEMS, ids=_SPEC_IDS)
def test_transformer_cls_is_importable_class(name, spec):
    # The class object is resolved at registry import time; being a ``type`` here
    # proves the import path is valid.
    assert isinstance(spec.transformer_cls, type)


@pytest.mark.parametrize("name, spec", _SPEC_ITEMS, ids=_SPEC_IDS)
def test_feature_kind_valid(name, spec):
    assert spec.feature_kind, f"{name} has no feature kind"
    assert spec.feature_kind <= _VALID_KINDS


@pytest.mark.parametrize("name, spec", _SPEC_ITEMS, ids=_SPEC_IDS)
def test_arity_valid(name, spec):
    assert spec.arity in _VALID_ARITY


@pytest.mark.parametrize("name, spec", _SPEC_ITEMS, ids=_SPEC_IDS)
def test_target_usage_valid(name, spec):
    assert spec.target_usage in _VALID_TARGET_USAGE


@pytest.mark.parametrize("name, spec", _SPEC_ITEMS, ids=_SPEC_IDS)
def test_placement_strategies_valid(name, spec):
    assert spec.placement_strategies <= _ALL_STRATEGIES


@pytest.mark.parametrize("name, spec", _SPEC_ITEMS, ids=_SPEC_IDS)
def test_target_usage_and_placement_are_consistent(name, spec):
    if spec.target_usage == "required":
        # Always target-aware: only the supervised strategies apply.
        assert spec.placement_strategies == _TARGET_AWARE
    elif spec.target_usage == "optional":
        # Both modes available: every strategy applies.
        assert spec.placement_strategies == _ALL_STRATEGIES
    else:  # forbidden
        # Never uses y: any placement it has must be unsupervised.
        assert spec.placement_strategies <= _UNSUPERVISED


@pytest.mark.parametrize("name, spec", _SPEC_ITEMS, ids=_SPEC_IDS)
def test_adaptive_flag_matches_allowed_args(name, spec):
    assert spec.supports_adaptive_resolution == ("adaptive" in spec.allowed_args)


@pytest.mark.parametrize("name, spec", _SPEC_ITEMS, ids=_SPEC_IDS)
def test_optional_dependency_value(name, spec):
    assert spec.optional_dependency is None or spec.optional_dependency in _EXTRA_MODULE


@pytest.mark.parametrize("name, spec", _SPEC_ITEMS, ids=_SPEC_IDS)
def test_instantiable_when_dependency_present(name, spec):
    # Methods with no optional dependency (or whose dependency is installed)
    # must construct with defaults.
    if spec.optional_dependency and not _module_available(_EXTRA_MODULE[spec.optional_dependency]):
        pytest.skip(f"optional dependency {spec.optional_dependency!r} not installed")
    assert spec.transformer_cls() is not None


@pytest.mark.parametrize("name, spec", _SPEC_ITEMS, ids=_SPEC_IDS)
def test_required_target_methods_reject_missing_y(name, spec):
    if not spec.requires_target or spec.is_multivariate:
        pytest.skip("not a univariate required-target method")
    if spec.optional_dependency and not _module_available(_EXTRA_MODULE[spec.optional_dependency]):
        pytest.skip(f"optional dependency {spec.optional_dependency!r} not installed")
    transformer = spec.transformer_cls()
    X = np.linspace(0.0, 1.0, 60).reshape(-1, 1)
    with pytest.raises(PretabError):
        transformer.fit(X, None)


@pytest.mark.parametrize("name, spec", _SPEC_ITEMS, ids=_SPEC_IDS)
def test_optional_dependency_methods_fail_cleanly(name, spec):
    if spec.optional_dependency is None:
        pytest.skip("no optional dependency")
    module_name = _EXTRA_MODULE[spec.optional_dependency]
    if _module_available(module_name):
        pytest.skip(f"{module_name} is installed; cannot exercise the missing-dependency path")
    transformer = spec.transformer_cls()
    X = np.array([["a"], ["b"], ["c"]], dtype=object)
    with pytest.raises(OptionalDependencyError):
        transformer.fit(X)


def test_registry_covers_numerical_and_categorical_names():
    assert numerical_method_names() | categorical_method_names() == set(TRANSFORMER_REGISTRY)
    # ``custombin`` and ``none`` are the only dual-kind methods.
    dual = numerical_method_names() & categorical_method_names()
    assert dual == {"custombin", "none"}


# --------------------------------------------------------------------------- #
# End-to-end behavioural contract: the ``preprocessor_compatible`` flag and the
# target-usage declaration must match what the Preprocessor actually does.
# --------------------------------------------------------------------------- #
_PREPROC_NUMERICAL = [
    (name, spec)
    for name, spec in _SPEC_ITEMS
    if spec.is_numerical and spec.preprocessor_compatible and not spec.is_multivariate
]
_PREPROC_CATEGORICAL = [(name, spec) for name, spec in _SPEC_ITEMS if spec.is_categorical and spec.preprocessor_compatible]
_REQUIRED_NUMERICAL = [
    (name, spec) for name, spec in _SPEC_ITEMS if spec.is_numerical and spec.requires_target and not spec.is_multivariate
]


def _skip_if_dependency_missing(spec):
    if spec.optional_dependency and not _module_available(_EXTRA_MODULE[spec.optional_dependency]):
        pytest.skip(f"optional dependency {spec.optional_dependency!r} not installed")


@pytest.mark.parametrize(
    "name, spec", _PREPROC_NUMERICAL, ids=[name for name, _ in _PREPROC_NUMERICAL]
)
def test_preprocessor_compatible_numerical_methods_fit_transform(name, spec):
    _skip_if_dependency_missing(spec)
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"f0": rng.rand(60), "f1": rng.rand(60) * 5 + 1})
    y = rng.rand(60)
    if spec.requires_target:
        pre = Preprocessor(numerical_method=name, target_aware=True, placement_strategy="cart")
    else:
        pre = Preprocessor(numerical_method=name, target_aware=False, placement_strategy="uniform")
    out = pre.fit_transform(X, y, return_array=True)
    assert out.shape[0] == 60


@pytest.mark.parametrize(
    "name, spec", _PREPROC_CATEGORICAL, ids=[name for name, _ in _PREPROC_CATEGORICAL]
)
def test_preprocessor_compatible_categorical_methods_fit_transform(name, spec):
    _skip_if_dependency_missing(spec)
    rng = np.random.RandomState(0)
    # Integer-coded categories keep ``onehot_from_ordinal`` valid; a high cutoff
    # forces the low-cardinality integer column onto the categorical path.
    X = pd.DataFrame({"n": rng.rand(60), "c": rng.randint(0, 3, size=60)})
    y = rng.rand(60)
    pre = Preprocessor(
        numerical_method="standardization",
        categorical_method=name,
        cat_cutoff=0.5,
        target_aware=False,
        placement_strategy="uniform",
    )
    out = pre.fit_transform(X, y, return_array=True)
    assert out.shape[0] == 60


@pytest.mark.parametrize(
    "name, spec", _REQUIRED_NUMERICAL, ids=[name for name, _ in _REQUIRED_NUMERICAL]
)
def test_required_target_methods_raise_without_y_via_preprocessor(name, spec):
    _skip_if_dependency_missing(spec)
    X = pd.DataFrame({"f0": np.linspace(0.0, 1.0, 60)})
    pre = Preprocessor(numerical_method=name, target_aware=True, placement_strategy="cart")
    # A required-target method must fail loudly with a typed PretabError when fit
    # without a target (Phase 4 tightened this from the raw TypeError that
    # sklearn's fit_transform used to surface).
    with pytest.raises(PretabError):
        pre.fit(X)


# --------------------------------------------------------------------------- #
# Multivariate methods are standalone-only (D6): not selectable per column
# through the Preprocessor whitelist.
# --------------------------------------------------------------------------- #
_MULTIVARIATE_NUMERICAL = [(name, spec) for name, spec in _SPEC_ITEMS if spec.is_numerical and spec.is_multivariate]


@pytest.mark.parametrize(
    "name, spec", _MULTIVARIATE_NUMERICAL, ids=[name for name, _ in _MULTIVARIATE_NUMERICAL]
)
def test_multivariate_methods_not_preprocessor_selectable(name, spec):
    # The multivariate tensor-product / thin-plate splines are standalone-only and
    # deliberately excluded from the per-column Preprocessor whitelist; selecting
    # one must fail loudly rather than silently misbehave.
    assert spec.preprocessor_compatible is False
    assert name not in NUMERICAL_METHODS
    X = pd.DataFrame({"f0": np.linspace(0.0, 1.0, 60), "f1": np.linspace(1.0, 2.0, 60)})
    y = np.linspace(0.0, 1.0, 60)
    pre = Preprocessor(numerical_method=name)
    with pytest.raises(InvalidParamError):
        pre.fit(X, y)
