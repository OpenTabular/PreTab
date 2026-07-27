"""Shared fixtures for the extension-protocol tests.

Registration mutates process-global registry state, so every test in this
package runs against a snapshot that is restored afterwards to keep the suite
order-independent.
"""

import pytest

from pretab.compose import registry


@pytest.fixture(autouse=True)
def _restore_registry():
    saved_registry = dict(registry.TRANSFORMER_REGISTRY)
    saved_numerical = dict(registry.NUMERICAL_METHODS)
    saved_categorical = set(registry.CATEGORICAL_METHODS)
    try:
        yield
    finally:
        registry.TRANSFORMER_REGISTRY.clear()
        registry.TRANSFORMER_REGISTRY.update(saved_registry)
        registry.NUMERICAL_METHODS.clear()
        registry.NUMERICAL_METHODS.update(saved_numerical)
        registry.CATEGORICAL_METHODS.clear()
        registry.CATEGORICAL_METHODS.update(saved_categorical)
