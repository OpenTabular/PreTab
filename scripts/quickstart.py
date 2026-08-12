"""End-to-end sanity check for PreTab, doubling as a CI smoke test and a
five-minute artifact for reviewers. Run it with::

    python scripts/quickstart.py

Each check exercises a distinct part of the public API against a fixed,
synthetic dataset and prints a single-line result. The script exits with a
non-zero status if any check fails or raises.
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

from pretab import CrossFittedTransformer, LeakageWarning, Preprocessor, list_representations
from pretab.transformers import NaturalCubicSplineTransformer, PLETransformer

SEED = 0
N_ROWS = 400


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def make_dataset(n=N_ROWS, seed=SEED):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "tenure": rng.uniform(0.0, 20.0, size=n),
            "income": rng.normal(55_000, 12_000, size=n),
            "usage": rng.exponential(scale=3.0, size=n),
            "plan": rng.choice(["basic", "standard", "premium"], size=n),
            "region": rng.choice(["north", "south", "east", "west"], size=n),
        }
    )
    y = 0.4 * np.sin(X["tenure"] / 3) + X["income"] / 1e5 - 0.2 * X["usage"] + rng.normal(0, 0.1, size=n)
    return X, y.to_numpy()


def check_mixed_preprocessing(X, y):
    config = {
        "tenure": "naturalspline",
        "income": "rbf",
        "usage": "ple",
        "plan": "one-hot",
        "region": "int",
    }
    pre = Preprocessor(feature_preprocessing=config, task="regression", random_state=SEED)
    array = pre.fit_transform(X, y, return_array=True)
    if not isinstance(array, np.ndarray):
        raise TypeError("fit_transform(return_array=True) did not return an ndarray")
    expect(array.shape[0] == len(X), "row count changed during preprocessing")
    expect(np.isfinite(array).all(), "preprocessed output contains non-finite values")
    return f"{array.shape[0]} rows -> {array.shape[1]} columns"


def check_feature_lineage(X, y):
    pre = Preprocessor(
        feature_preprocessing={"tenure": "naturalspline", "income": "rbf", "usage": "ple"},
        categorical_method="one-hot",
        task="regression",
        random_state=SEED,
    ).fit(X, y)
    lineage = pre.get_feature_lineage()
    expect(len(lineage) == pre.total_output_dim_, "lineage does not cover every output column")
    sources = {record.source_features[0] for record in lineage}
    expect(sources == set(X.columns), "lineage is missing a source feature")
    return f"{len(lineage)}/{pre.total_output_dim_} columns traced to a source feature"


def check_leakage_safe_cross_fitting():
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-3.0, 3.0, size=(200, 1))
    y = rng.normal(size=200)

    with warnings.catch_warnings(record=True) as direct:
        warnings.simplefilter("always")
        PLETransformer(output_dim=10, random_state=SEED).fit(x, y)
    expect(
        any(issubclass(w.category, LeakageWarning) for w in direct),
        "fitting a target-aware transformer outside a pipeline should warn",
    )

    with warnings.catch_warnings(record=True) as piped:
        warnings.simplefilter("always")
        Pipeline([("ple", PLETransformer(output_dim=10, random_state=SEED))]).fit(x, y)
    expect(
        not any(issubclass(w.category, LeakageWarning) for w in piped),
        "fitting inside a pipeline should not warn",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=LeakageWarning)
        naive = PLETransformer(output_dim=10, random_state=SEED).fit(x, y).transform(x)
    cross = CrossFittedTransformer(PLETransformer(output_dim=10, random_state=SEED), n_folds=5, random_state=SEED)
    out_of_fold = cross.fit_transform(x, y)

    changed = int((~np.all(naive == out_of_fold, axis=1)).sum())
    expect(changed > len(x) // 2, "cross-fitting did not change enough rows to look out-of-fold")
    return f"warns outside a pipeline, silent inside one, {changed}/{len(x)} rows re-encoded out-of-fold"


def check_sklearn_pipeline(X, y):
    x = X[["tenure"]].to_numpy()
    pipeline = Pipeline(
        [
            ("spline", NaturalCubicSplineTransformer(output_dim=8)),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit(x, y)
    predictions = pipeline.predict(x)
    expect(predictions.shape == y.shape, "prediction shape does not match the target")
    expect(np.isfinite(predictions).all(), "predictions contain non-finite values")
    score = r2_score(y, predictions)
    return f"Ridge on 8 spline basis columns, R2 = {score:.3f}"


def check_serialization_roundtrip(X, y):
    pre = Preprocessor(
        feature_preprocessing={"tenure": "naturalspline", "usage": "ple"},
        categorical_method="int",
        task="regression",
        random_state=SEED,
    ).fit(X, y)
    reloaded = Preprocessor.from_spec(pre.to_spec())

    original = pre.transform(X, return_array=True)
    restored = reloaded.transform(X, return_array=True)
    if not (isinstance(original, np.ndarray) and isinstance(restored, np.ndarray)):
        raise TypeError("transform(return_array=True) did not return an ndarray")
    np.testing.assert_array_equal(original, restored)
    expect(pre.fingerprint_ == reloaded.fingerprint_, "fingerprint changed across a spec round trip")
    return f"fingerprint {pre.fingerprint_[:12]} reproduced bit-for-bit after a spec round trip"


def check_representation_discovery():
    supervised_numerical = list_representations(feature_kind="numerical", supervised=True)
    all_methods = list_representations()
    expect("ple" in supervised_numerical, "the registry lost a documented method")
    expect("one-hot" not in supervised_numerical, "a categorical-only method leaked into a numerical filter")
    return f"{len(all_methods)} registered methods, {len(supervised_numerical)} target-aware numerical"


CHECKS = [
    ("mixed-type preprocessing", check_mixed_preprocessing, True),
    ("feature lineage", check_feature_lineage, True),
    ("leakage-safe cross-fitting", check_leakage_safe_cross_fitting, False),
    ("sklearn pipeline compatibility", check_sklearn_pipeline, True),
    ("portable serialization", check_serialization_roundtrip, True),
    ("representation discovery", check_representation_discovery, False),
]


def main():
    import pretab

    print(f"PreTab quickstart (pretab {pretab.__version__})")
    print("-" * 64)

    X, y = make_dataset()
    start = time.perf_counter()
    failed = []

    for index, (label, check, needs_data) in enumerate(CHECKS, start=1):
        prefix = f"[{index}/{len(CHECKS)}] {label}"
        try:
            detail = check(X, y) if needs_data else check()
            print(f"{prefix:<45} ok   {detail}")
        except Exception as exc:  # a failing check should not stop the rest from running
            failed.append(label)
            print(f"{prefix:<45} FAIL {exc}")

    elapsed = time.perf_counter() - start
    print("-" * 64)
    if failed:
        print(f"{len(failed)}/{len(CHECKS)} checks failed: {', '.join(failed)}")
        return 1

    print(f"all {len(CHECKS)} checks passed in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
