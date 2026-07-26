"""Shared fixtures for the compose unit tests.

``make_config`` builds a :class:`~pretab.compose.config.PreprocessorConfig` from
a simple, target-free default set (standardization + int, unsupervised uniform
placement) so individual tests only override what they exercise.
"""

import pandas as pd
import pytest

from pretab.compose.config import PreprocessorConfig

_CONFIG_DEFAULTS = {
    "numerical_method": "standardization",
    "categorical_method": "int",
    "feature_preprocessing": None,
    "output_dim": 7,
    "degree": 3,
    "target_aware": False,
    "placement_strategy": "uniform",
    "task": "regression",
    "adaptive": False,
    "min_output_dim": 5,
    "max_output_dim": 10,
    "random_state": None,
    "scaling": None,
    "cat_cutoff": 0.03,
    "treat_all_integers_as_numerical": False,
    "numerical_imputation": "median",
    "categorical_imputation": "most_frequent",
    "add_missing_indicator": False,
    "verbose": 0,
}


@pytest.fixture
def make_config():
    """Return a factory that builds a config from the defaults plus overrides."""

    def _make(**overrides):
        return PreprocessorConfig.from_params(**{**_CONFIG_DEFAULTS, **overrides})

    return _make


@pytest.fixture
def sample_frame():
    """A tiny mixed numerical/categorical frame for factory/inspection tests."""
    return pd.DataFrame(
        {
            "age": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "city": ["a", "b", "a", "c", "b", "a"],
        }
    )
