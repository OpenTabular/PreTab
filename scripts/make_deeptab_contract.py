"""Generate synthetic DeepTab contract fixtures: python -m scripts.make_deeptab_contract."""

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

import pretab

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests/integration/_fixtures/deeptab_contract/v1"
TOLERANCES = {"atol": 1e-9, "rtol": 1e-7}


@dataclass(frozen=True)
class ContractCase:
    fit_data: pd.DataFrame
    probe_data: pd.DataFrame
    config: dict[str, Any]
    target: np.ndarray | None = None


def contract_cases() -> dict[str, ContractCase]:
    base = {"numerical_method": "minmax", "categorical_method": "one-hot", "output_structure": "matrix"}
    reading = np.arange(32, dtype=np.float64)
    target = reading / 100 + (reading >= 8) + 3 * (reading >= 16) + 2 * (reading >= 24)
    return {
        "dense_numeric_boolean": ContractCase(
            pd.DataFrame({"level": [0.0, 2.0, 4.0, 6.0], "flag": [True, False, True, False]}),
            pd.DataFrame({"level": [1.0, 5.0], "flag": [False, True]}),
            {**base, "output_format": "dense"},
        ),
        "sparse_unknown_missing": ContractCase(
            pd.DataFrame({"level": [0.0, 2.0, 4.0, 6.0], "city": pd.Series(["a", "b", None, "a"], dtype=object)}),
            pd.DataFrame({"level": [1.0, np.nan, 5.0], "city": pd.Series(["new", None, "a"], dtype=object)}),
            {**base, "output_format": "sparse", "missing_policy": "separate_state"},
        ),
        "dense_blocks_adaptive": ContractCase(
            pd.DataFrame({"reading": reading, "city": pd.Series(["a", "b"] * 16, dtype=object)}),
            pd.DataFrame({"reading": [2.0, 15.0, 29.0], "city": pd.Series(["a", "b", "a"], dtype=object)}),
            {
                "output_structure": "blocks",
                "output_format": "dense",
                "categorical_method": "one-hot",
                "feature_preprocessing": {"reading": {"method": "rbf", "output_dim": 12, "adaptive": True}},
                "min_output_dim": 3,
                "max_output_dim": 4,
                "target_aware": True,
                "placement_strategy": "cart",
                "random_state": 17,
            },
            target,
        ),
    }


def _encode_config(config: dict[str, Any]) -> dict[str, Any]:
    encoded = config.copy()
    if encoded.get("feature_preprocessing") is not None:
        encoded["feature_preprocessing"] = [
            {"feature": feature, "settings": settings} for feature, settings in encoded["feature_preprocessing"].items()
        ]
    return encoded


def _source_digest() -> str:
    root = Path(pretab.__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted(root.rglob("*.py")):
        digest.update(source.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(source.read_bytes() + b"\0")
    return digest.hexdigest()


def _optional_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def fixture_environment() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "pretab": pretab.__version__,
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scipy": version("scipy"),
        "scikit_learn": version("scikit-learn"),
        "polars": _optional_version("polars"),
        "pyarrow": _optional_version("pyarrow"),
        "platform": platform.platform(),
        "pretab_source_sha256": _source_digest(),
    }


def capture_fixture(case_id: str, case: ContractCase, environment: dict[str, str | None]) -> dict[str, Any]:
    estimator = pretab.Preprocessor(**case.config).fit(case.fit_data, case.target)
    blocks = estimator.transform(case.probe_data, return_array=False)
    if not isinstance(blocks, dict):
        raise TypeError(f"{case_id}: expected a mapping of transformed blocks")
    samples = np.concatenate(
        [block.toarray() if sparse.issparse(block) else np.asarray(block) for block in blocks.values()], axis=1
    )
    feature_info = estimator.get_feature_info(verbose=False)
    return {
        "schema_version": 1,
        "case_id": case_id,
        "input_schema": {
            "container": "pandas",
            "columns": [{"name": label, "dtype": str(dtype)} for label, dtype in case.fit_data.dtypes.items()],
        },
        "constructor_config": _encode_config(case.config),
        "resolved_config": _encode_config(estimator.get_resolved_config()),
        "output_feature_names": estimator.get_feature_names_out().tolist(),
        "block_widths": [{"name": name, "width": block.shape[1]} for name, block in blocks.items()],
        "feature_info_dimensions": {
            kind: {name: info["dimension"] for name, info in entries.items()}
            for kind, entries in zip(("numerical", "categorical", "embedding"), feature_info, strict=True)
        },
        "fingerprint": estimator.fingerprint_,
        "transformed_rows": [{"row_index": index, "values": row} for index, row in enumerate(samples.tolist())],
        "tolerances": TOLERANCES.copy(),
        "library_versions": environment.copy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIXTURE_DIR)
    output_dir = parser.parse_args().output_dir
    environment = fixture_environment()
    output_dir.mkdir(parents=True, exist_ok=True)
    for case_id, case in contract_cases().items():
        fixture = capture_fixture(case_id, case, environment)
        path = output_dir / f"{case_id}.json"
        path.write_text(json.dumps(fixture, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
