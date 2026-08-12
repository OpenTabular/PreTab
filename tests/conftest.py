"""Root test configuration.

``pyproject.toml`` turns :class:`~pretab.exceptions.LeakageWarning` into an error
so an *unintended* leakage warning fails the suite. The modules listed below
intentionally fit target-aware transformers directly (outside a Pipeline) to
exercise their behaviour, so the expected leakage warning is silenced there. The
dedicated leakage tests still assert the warning explicitly via ``pytest.warns``.
"""

import pytest

# Test modules that fit supervised transformers directly; the leakage warning is
# expected here and must not fail the suite. Any *other* module that emits it is
# a real regression and will error.
_LEAKAGE_EXPECTED_MODULES = frozenset(
    {
        "test_feature_map_selector.py",
        "test_adaptive_resolution.py",
        "test_ple_selector.py",
        "test_cross_fitted.py",
        "test_supervised_contract.py",
        "test_ple_transformer.py",
        "test_rbfexpansion_transformer.py",
        "test_reluexpansion_transformer.py",
        "test_sigmoidexpansion_transformer.py",
        "test_spline_api_parity.py",
        "test_spline_expansions.py",
        "test_output_dimension.py",
        "test_exceptions.py",
        "test_adaptive_output_dim.py",
        "test_reproducibility.py",
    }
)

_LEAKAGE_IGNORE = pytest.mark.filterwarnings("ignore::pretab.exceptions.LeakageWarning")


def pytest_collection_modifyitems(items):
    """Silence the expected leakage warning in modules that fit supervised transformers directly."""
    for item in items:
        if item.path.name in _LEAKAGE_EXPECTED_MODULES:
            item.add_marker(_LEAKAGE_IGNORE)
