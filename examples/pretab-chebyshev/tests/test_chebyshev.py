"""Illustrative tests for the example extension.

Run from this directory with ``pip install -e . && pytest``. These are not part
of the main PreTab test suite (which only collects the top-level ``tests/``).
"""

import numpy as np
import pandas as pd
from pretab_chebyshev import ChebyshevRepresentation

from pretab import Preprocessor, check_representation, list_representations, register_representation


def test_passes_conformance_suite():
    passed = check_representation(ChebyshevRepresentation)
    assert "spec_consistent" in passed
    assert "deterministic" in passed


def test_register_and_use_through_preprocessor():
    register_representation(
        "chebyshev", ChebyshevRepresentation, allowed_args=("degree",), override=True
    )
    assert "chebyshev" in list_representations(feature_kind="numerical")

    X = pd.DataFrame({"a": np.linspace(0, 1, 20), "b": np.linspace(-1, 1, 20)})
    pre = Preprocessor(
        numerical_method="chebyshev",
        categorical_method="none",
        degree=4,  # flows through because "degree" is in allowed_args
        target_aware=False,
        placement_strategy="uniform",
    )
    out = np.asarray(pre.fit_transform(X, return_array=True))
    assert out.shape == (20, 8)  # degree 4 x 2 features
