"""Tests for the fingerprint and reproducibility report (P9.2).

Covers determinism within and across processes, sensitivity to configuration /
data / seed changes, round-trip stability, and the ``reproducibility_report``
contents.
"""

import subprocess
import sys
import textwrap
from contextlib import nullcontext
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from pretab import Preprocessor
from pretab.compose.registry import (
    CATEGORICAL_ALIASES,
    CATEGORICAL_METHODS,
    NUMERICAL_ALIASES,
    NUMERICAL_METHODS,
    TRANSFORMER_REGISTRY,
)


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


@pytest.mark.parametrize("per_feature", [False, True], ids=["global", "override"])
@pytest.mark.parametrize(
    ("feature", "method", "canonical"),
    [
        pytest.param(feature, method, canonical, id=f"{feature}-{method}")
        for feature, methods, aliases in (
            ("a", NUMERICAL_METHODS, NUMERICAL_ALIASES),
            ("c", CATEGORICAL_METHODS, CATEGORICAL_ALIASES),
        )
        for method, canonical in [
            *((name, name) for name in sorted(methods)),
            *((alias, name) for alias, name in sorted(aliases.items()) if name in methods),
        ]
        if TRANSFORMER_REGISTRY[canonical].optional_dependency is None
    ],
)
def test_legacy_strings_keep_fitted_artifact_through_report_and_clone(
    frame, target, feature, method, canonical, per_feature
):
    frame = pd.concat([frame.assign(c=pd.Categorical(frame["c"]).codes)] * 20, ignore_index=True)
    target = np.tile(target, 20)
    supervised = TRANSFORMER_REGISTRY[canonical].requires_target
    parameters: dict[str, Any] = {
        "numerical_method": "none",
        "categorical_method": "none",
        "target_aware": supervised,
        "placement_strategy": "cart" if supervised else "uniform",
        "random_state": 17,
        "scaling": "none",
    }
    if per_feature:
        parameters["feature_preprocessing"] = {feature: method}
    else:
        parameters["numerical_method" if feature == "a" else "categorical_method"] = method
    with (
        pytest.warns(DeprecationWarning, match="OneHotFromOrdinalTransformer is deprecated")
        if canonical == "onehot_from_ordinal"
        else nullcontext()
    ):
        processor = Preprocessor(**parameters).fit(frame, target)
    original = np.asarray(processor.transform(frame, return_array=True))
    names = processor.get_feature_names_out()
    widths = dict(processor.output_dims_)
    fingerprint = processor.fingerprint_
    requested = processor.get_params()

    processor.get_resolved_config()
    fresh = clone(processor)
    assert isinstance(fresh, Preprocessor)
    with (
        pytest.warns(DeprecationWarning, match="OneHotFromOrdinalTransformer is deprecated")
        if canonical == "onehot_from_ordinal"
        else nullcontext()
    ):
        fresh.fit(frame, target)
    for candidate in (processor, fresh):
        actual = np.asarray(candidate.transform(frame, return_array=True))
        assert actual.dtype == original.dtype
        assert actual.shape == original.shape
        assert actual.tobytes() == original.tobytes()
        np.testing.assert_array_equal(candidate.get_feature_names_out(), names)
        assert candidate.output_dims_ == widths
        assert candidate.total_output_dim_ == original.shape[1]
        assert candidate.fingerprint_ == fingerprint
        assert candidate.get_params() == requested


def test_fingerprint_survives_round_trip(frame, target):
    p = _fit(frame, target)
    restored = Preprocessor.from_spec(p.to_spec())
    assert restored.fingerprint_ == p.fingerprint_


def test_fingerprint_stable_after_inference_and_lifecycle_changes(frame, target):
    p = _fit(frame, target)
    fingerprint = p.fingerprint_
    for batch in (frame, frame.iloc[:3]):
        p.transform(batch)
        assert p.fingerprint_ == fingerprint
    p.mark_stale("new training data available")
    p.freeze()
    assert p.fingerprint_ == fingerprint
    restored = Preprocessor.from_spec(p.to_spec())
    assert restored.fingerprint_ == fingerprint
    assert restored.is_frozen()
    assert restored.stale_reason_ == "new training data available"


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
