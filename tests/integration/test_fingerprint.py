"""Tests for the fingerprint and reproducibility report (P9.2).

Covers determinism within and across processes, sensitivity to configuration /
data / seed changes, round-trip stability, and the ``reproducibility_report``
contents.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from pretab import Preprocessor


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.random(50), "b": rng.random(50) * 5.0, "c": rng.choice(["x", "y", "z"], 50)})


@pytest.fixture
def target():
    rng = np.random.default_rng(1)
    return rng.random(50)


def _fit(frame, target, **kwargs):
    params = {"numerical_method": "rbf", "target_aware": False, "placement_strategy": "quantile"}
    params.update(kwargs)
    return Preprocessor(**params).fit(frame, target)


def test_fingerprint_is_hex_sha256(frame, target):
    fp = _fit(frame, target).fingerprint_
    assert isinstance(fp, str)
    assert len(fp) == 64
    assert all(ch in "0123456789abcdef" for ch in fp)


def test_fingerprint_deterministic_same_fit(frame, target):
    assert _fit(frame, target).fingerprint_ == _fit(frame, target).fingerprint_


def test_fingerprint_survives_round_trip(frame, target):
    p = _fit(frame, target)
    restored = Preprocessor.from_spec(p.to_spec())
    assert restored.fingerprint_ == p.fingerprint_


def test_fingerprint_changes_with_config(frame, target):
    a = _fit(frame, target, output_dim=6)
    b = _fit(frame, target, output_dim=9)
    assert a.fingerprint_ != b.fingerprint_


def test_fingerprint_changes_with_data(frame, target):
    other = frame.copy()
    other.loc[other.index[0], "a"] = other.loc[other.index[0], "a"] + 1.0
    assert _fit(frame, target).fingerprint_ != _fit(other, target).fingerprint_


def test_fingerprint_changes_with_seed(frame, target):
    # The fingerprint incorporates the seed (roadmap D12), so distinct seeds yield
    # distinct fingerprints even when the fitted state happens to coincide.
    a = _fit(frame, target, numerical_method="ple", target_aware=True, placement_strategy="cart", random_state=0)
    b = _fit(frame, target, numerical_method="ple", target_aware=True, placement_strategy="cart", random_state=1)
    assert a.fingerprint_ != b.fingerprint_


def test_fingerprint_stable_across_processes(frame, target):
    script = textwrap.dedent(
        """
        import numpy as np, pandas as pd
        from pretab import Preprocessor
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"a": rng.random(50), "b": rng.random(50) * 5.0,
                              "c": rng.choice(["x", "y", "z"], 50)})
        y = np.random.default_rng(1).random(50)
        p = Preprocessor(numerical_method="rbf", target_aware=False,
                         placement_strategy="quantile").fit(frame, y)
        print(p.fingerprint_)
        """
    )
    def _run():
        result = subprocess.run(  # noqa: S603 - fixed interpreter + inline script, no untrusted input
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    first = _run()
    second = _run()
    assert first == second
    assert first == _fit(frame, target).fingerprint_


def test_reproducibility_report_contents(frame, target):
    p = _fit(frame, target)
    report = p.reproducibility_report()
    assert report["fingerprint"] == p.fingerprint_
    assert report["schema_version"] == 1
    assert set(report["library_versions"]) == {"numpy", "scipy", "scikit_learn"}
    assert report["n_features_in"] == frame.shape[1]
    assert report["n_output_features"] == len(p.get_feature_names_out())
    assert report["output_format"] == "dense"
    assert "a" in report["representations"]
