"""P0.4 behaviour baseline: golden-output regression guard.

Captures the exact output of the ``Preprocessor`` for a handful of representative
configurations *before* the 1.0.0 restructure, so the file moves / renames of
Phases 1-3 can be proven behaviour-preserving. Each config is compared against a
committed golden array with a tight tolerance (robust across the OS/Python CI
matrix) plus exact shape and feature-name checks.

Regenerate the goldens intentionally (e.g. after a deliberate behaviour change in
Phases 4-5, recorded in CHANGELOG.md) with::

    PRETAB_REGEN_GOLDEN=1 pytest tests/regression/test_golden_baseline.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pretab.preprocessor import Preprocessor

GOLDEN_DIR = Path(__file__).parent / "_golden"
REGEN = os.environ.get("PRETAB_REGEN_GOLDEN") == "1"

RTOL = 1e-6
ATOL = 1e-8


def _make_dataset():
    """Deterministic mixed-type dataset shared by every golden config."""
    rng = np.random.RandomState(20240726)
    n = 200
    X = pd.DataFrame(
        {
            "num_linear": np.linspace(-3.0, 3.0, n),
            "num_normal": rng.randn(n),
            "num_skewed": rng.exponential(scale=2.0, size=n),
            "cat_str": rng.choice(["alpha", "beta", "gamma"], size=n),
            "cat_int": rng.randint(0, 5, size=n),
        }
    )
    y = pd.Series(
        2.0 * X["num_linear"] + X["num_normal"] - 0.5 * X["num_skewed"] + rng.randn(n) * 0.1,
        name="target",
    )
    return X, y


CONFIGS = {
    "spline_unsupervised": {
        "numerical_method": "naturalspline",
        "categorical_method": "one-hot",
        "target_aware": False,
        "placement_strategy": "quantile",
        "output_dim": 7,
        "random_state": 0,
    },
    "ple_supervised": {
        "numerical_method": "ple",
        "categorical_method": "one-hot",
        "target_aware": True,
        "placement_strategy": "cart",
        "task": "regression",
        "output_dim": 5,
        "random_state": 0,
    },
    "featuremap_unsupervised": {
        "numerical_method": "rbf",
        "categorical_method": "int",
        "target_aware": False,
        "placement_strategy": "uniform",
        "output_dim": 6,
        "random_state": 0,
    },
}


def _transform(config):
    X, y = _make_dataset()
    pre = Preprocessor(**config)
    array = np.asarray(pre.fit_transform(X, y, return_array=True))
    names = [str(name) for name in pre.get_feature_names_out()]
    return array, names


@pytest.mark.smoke
@pytest.mark.parametrize("config_id", sorted(CONFIGS))
def test_golden_output_is_behaviour_preserving(config_id):
    array, names = _transform(CONFIGS[config_id])
    npz_path = GOLDEN_DIR / f"{config_id}.npz"
    json_path = GOLDEN_DIR / f"{config_id}.json"

    if REGEN:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, output=array)
        json_path.write_text(json.dumps({"shape": list(array.shape), "feature_names": names}, indent=2))
        pytest.skip(f"regenerated golden for {config_id}")

    assert npz_path.exists(), (
        f"missing golden {npz_path.name}; regenerate with PRETAB_REGEN_GOLDEN=1 pytest {Path(__file__).name}"
    )
    golden = np.load(npz_path)["output"]
    meta = json.loads(json_path.read_text())

    assert list(array.shape) == meta["shape"]
    assert names == meta["feature_names"]
    np.testing.assert_allclose(array, golden, rtol=RTOL, atol=ATOL)
