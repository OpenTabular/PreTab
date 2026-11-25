"""Illustrative tests for the example extension.

Run from this directory with ``pip install -e . && pytest``. These are not part
of the main PreTab test suite (which only collects the top-level ``tests/``).
"""

import numpy as np
import pandas as pd
import pytest
from pretab_chebyshev import ChebyshevRepresentation
from sklearn.exceptions import NotFittedError

from pretab import Preprocessor, check_representation, list_representations, register_representation
from pretab.compose import registry


@pytest.fixture(autouse=True)
def _restore_registry():
    saved_registry = registry.TRANSFORMER_REGISTRY.copy()
    saved_numerical = registry.NUMERICAL_METHODS.copy()
    saved_categorical = registry.CATEGORICAL_METHODS.copy()
    try:
        yield
    finally:
        registry.TRANSFORMER_REGISTRY.clear()
        registry.TRANSFORMER_REGISTRY.update(saved_registry)
        registry.NUMERICAL_METHODS.clear()
        registry.NUMERICAL_METHODS.update(saved_numerical)
        registry.CATEGORICAL_METHODS.clear()
        registry.CATEGORICAL_METHODS.update(saved_categorical)


def test_passes_conformance_suite():
    passed = check_representation(ChebyshevRepresentation)
    assert "spec_consistent" in passed
    assert "deterministic" in passed


def test_rejects_nan_during_fit_and_transform():
    finite = np.array([[0.0], [1.0]])
    missing = np.array([[0.0], [np.nan], [1.0]])
    transformer = ChebyshevRepresentation()

    with pytest.raises(ValueError, match="NaN"):
        transformer.fit(missing)
    with pytest.raises(NotFittedError):
        transformer.transform(finite)

    transformer.fit(finite)
    with pytest.raises(ValueError, match="NaN"):
        transformer.transform(missing)
    assert np.isfinite(transformer.transform(finite)).all()


def test_register_and_use_through_preprocessor():
    register_representation("chebyshev", ChebyshevRepresentation, allowed_args=("degree",), override=True)
    assert "chebyshev" in list_representations(feature_kind="numerical")

    X = pd.DataFrame({"a": np.linspace(0, 1, 20), "b": np.linspace(-1, 1, 20)})
    X.loc[0, "a"] = np.nan
    pre = Preprocessor(
        numerical_method="chebyshev",
        categorical_method="none",
        degree=4,  # flows through because "degree" is in allowed_args
        target_aware=False,
        placement_strategy="uniform",
    )
    out = np.asarray(pre.fit_transform(X, return_array=True))
    assert out.shape == (20, 8)  # degree 4 x 2 features
    assert np.isfinite(out).all()
