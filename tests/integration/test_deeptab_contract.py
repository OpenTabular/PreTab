"""Validate the synthetic fixtures consumed by the DeepTab compatibility harness."""

import json

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from scipy import sparse

from pretab import Preprocessor
from scripts.make_deeptab_contract import FIXTURE_DIR, capture_fixture, contract_cases, fixture_environment

CASES = contract_cases()


@pytest.fixture(scope="module")
def schema_validator():
    schema = json.loads((FIXTURE_DIR.parent / "schema-v1.json").read_bytes())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_fixture_catalog_matches_case_recipes():
    assert {path.stem for path in FIXTURE_DIR.glob("*.json")} == set(CASES)


@pytest.mark.parametrize("case_id", CASES)
def test_fitted_deeptab_contract(case_id, schema_validator):
    expected = json.loads((FIXTURE_DIR / f"{case_id}.json").read_bytes())
    json.dumps(expected, allow_nan=False)
    schema_validator.validate(expected)

    actual = capture_fixture(case_id, CASES[case_id], fixture_environment())
    actual = json.loads(json.dumps(actual, allow_nan=False))
    for field in expected.keys() - {"fingerprint", "transformed_rows", "library_versions"}:
        assert actual[field] == expected[field], field

    if actual["library_versions"] == expected["library_versions"]:
        assert actual["fingerprint"] == expected["fingerprint"]

    width = len(expected["output_feature_names"])
    assert sum(block["width"] for block in expected["block_widths"]) == width
    for expected_row, actual_row in zip(expected["transformed_rows"], actual["transformed_rows"], strict=True):
        assert actual_row["row_index"] == expected_row["row_index"]
        assert len(expected_row["values"]) == len(actual_row["values"]) == width
        np.testing.assert_allclose(
            actual_row["values"], expected_row["values"], equal_nan=False, **expected["tolerances"]
        )


@pytest.mark.parametrize("case_id", CASES)
def test_deeptab_case_preserves_output_mode_and_column_order(case_id):
    case = CASES[case_id]
    estimator = Preprocessor(**case.config).fit(case.fit_data, case.target)
    result = estimator.transform(case.probe_data)
    blocks = estimator.transform(case.probe_data, return_array=False)
    matrix = estimator.transform(case.probe_data, return_array=True)

    def dense(values):
        return values.toarray() if sparse.issparse(values) else np.asarray(values)

    assert isinstance(blocks, dict)
    np.testing.assert_array_equal(np.concatenate([dense(block) for block in blocks.values()], axis=1), dense(matrix))
    assert dense(matrix).shape[1] == len(estimator.get_feature_names_out())

    if case_id == "dense_numeric_boolean":
        assert isinstance(result, np.ndarray)
        assert estimator.get_feature_names_out().tolist() == ["num_level", "cat_flag_False", "cat_flag_True"]
        np.testing.assert_allclose(result, [[1 / 6, 1, 0], [5 / 6, 0, 1]], equal_nan=False)
    elif case_id == "sparse_unknown_missing":
        assert sparse.issparse(result)
        np.testing.assert_array_equal(dense(blocks["num_level"])[:, -1], [0, 1, 0])
        city = dense(blocks["cat_city"])
        assert not city[0].any()
        np.testing.assert_array_equal(city[:, -1], [0, 1, 0])
        assert city[2, :-1].sum() == 1
    else:
        assert isinstance(result, dict)
        assert list(result) == ["num_reading", "cat_city"]
        assert all(isinstance(block, np.ndarray) for block in result.values())
        numerical, categorical, embeddings = estimator.get_feature_info(verbose=False)
        assert 3 <= result["num_reading"].shape[1] == numerical["reading"]["dimension"] <= 4
        assert result["cat_city"].shape[1] == categorical["city"]["dimension"] == 2
        assert not embeddings
