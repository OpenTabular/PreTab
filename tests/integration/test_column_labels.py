"""Column labels the ColumnTransformer cannot take verbatim.

Regression guards for issue #55: integer labels were read as *positions* by
scikit-learn's ColumnTransformer, silently swapping columns.
"""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from pretab import Preprocessor
from pretab.exceptions import PretabDataError


@pytest.fixture
def int_labelled():
    # Labels are a permutation of the positions: label 1 sits at position 0.
    return pd.DataFrame({1: [10.0, 20, 30, 40, 50, 60], 0: [1.0, 2, 3, 4, 5, 6]})


def _passthrough(**kwargs):
    return Preprocessor(numerical_method="none", scaling="none", output_structure="blocks", **kwargs)


def _blocks(out):
    return cast("dict[str, np.ndarray]", out)


def test_integer_labels_route_each_column_to_its_own_block(int_labelled):
    out = _blocks(_passthrough().fit_transform(int_labelled))
    np.testing.assert_array_equal(out["num_1"].ravel(), int_labelled[1].to_numpy())
    np.testing.assert_array_equal(out["num_0"].ravel(), int_labelled[0].to_numpy())
    # The ColumnTransformer works on a relabelled copy; the caller's frame keeps its labels.
    assert list(int_labelled.columns) == [1, 0]


def test_feature_preprocessing_with_integer_keys_targets_the_labelled_column(int_labelled):
    pre = Preprocessor(feature_preprocessing={1: "minmax", 0: "none"}, scaling="none", output_structure="blocks")
    out = _blocks(pre.fit_transform(int_labelled))
    np.testing.assert_allclose(out["num_1"].ravel(), np.linspace(0.0, 1.0, 6))
    np.testing.assert_array_equal(out["num_0"].ravel(), int_labelled[0].to_numpy())


def test_non_contiguous_integer_labels_fit_and_transform():
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 3))).drop(columns=[0])
    pre = Preprocessor(numerical_method="minmax").fit(frame)
    assert pre.get_feature_names_out().tolist() == ["num_1", "num_2"]
    out = np.asarray(pre.transform(frame))
    assert out.shape == (20, 2)
    np.testing.assert_allclose(out[:, 0], (frame[1] - frame[1].min()) / np.ptp(frame[1]))


def test_integer_labels_of_different_column_types_reach_their_own_pipeline():
    # Routed by position, the string column landed in the numerical pipeline and fit
    # failed with "Cannot use median strategy with non-numeric data".
    frame = pd.DataFrame({1: list("abcabc"), 0: [1.0, 2, 3, 4, 5, 6]})
    pre = Preprocessor(numerical_method="minmax", scaling="none", categorical_method="int").fit(frame)
    assert pre.get_feature_names_out().tolist() == ["num_0", "cat_1"]
    out = np.asarray(pre.transform(frame), dtype=float)
    np.testing.assert_allclose(out[:, 0], np.linspace(0.0, 1.0, 6))
    np.testing.assert_array_equal(out[:, 1], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    default = Preprocessor().fit(frame, np.arange(6.0))
    assert (default.numerical_features_, default.categorical_features_) == ([0], [1])
    np.testing.assert_array_equal(np.asarray(default.transform(frame))[:, -1], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    ("dtype", "missing_policy", "expected_codes"),
    [
        ("bool", "impute", [2.0, 1.0, 2.0, 1.0, 2.0, 2.0]),
        ("boolean", "impute", [2.0, 1.0, 2.0, 1.0, 2.0, 2.0]),
        ("boolean", "propagate", [2.0, 1.0, 0.0, 1.0, 2.0, 2.0]),
    ],
)
def test_boolean_columns_with_integer_labels_keep_routing_and_encoding(dtype, missing_policy, expected_codes):
    frame = pd.DataFrame({1: pd.array([True, False, True, False, True, True], dtype=dtype), 0: [1.0, 2, 3, 4, 5, 6]})
    if dtype == "boolean":
        frame.loc[2, 1] = pd.NA
    original = frame.copy(deep=True)
    pre = _passthrough(categorical_method="int", missing_policy=missing_policy).fit(frame)
    assert pre.numerical_features_ == [0]
    assert pre.categorical_features_ == [1]
    assert pre.output_dims_ == {0: 1, 1: 1}
    for data in (frame, frame[[0, 1]], frame.to_numpy(na_value=np.nan)):
        out = _blocks(pre.transform(data))
        np.testing.assert_array_equal(out["num_0"].ravel(), frame[0].to_numpy())
        np.testing.assert_array_equal(out["cat_1"].ravel(), expected_codes)
    pd.testing.assert_frame_equal(frame, original)


def test_integer_labels_are_matched_by_label_at_transform(int_labelled):
    pre = _passthrough(output_format="dense").fit(int_labelled)
    reordered = int_labelled[[0, 1]]
    np.testing.assert_array_equal(
        _blocks(pre.transform(reordered))["num_1"], _blocks(pre.transform(int_labelled))["num_1"]
    )


def test_array_input_after_an_integer_labelled_fit_is_matched_by_position(int_labelled):
    pre = _passthrough().fit(int_labelled)
    out = _blocks(pre.transform(int_labelled.to_numpy()))
    np.testing.assert_array_equal(out["num_1"].ravel(), int_labelled[1].to_numpy())
    np.testing.assert_array_equal(out["num_0"].ravel(), int_labelled[0].to_numpy())


def test_integer_label_metadata_uses_the_input_labels(int_labelled):
    pre = Preprocessor(numerical_method="minmax").fit(int_labelled)
    assert pre.get_feature_names_out().tolist() == ["num_1", "num_0"]
    assert pre.output_dims_ == {1: 1, 0: 1}
    numerical, _, _ = pre.get_feature_info(verbose=False)
    assert list(numerical) == [1, 0]
    lineage = [(record.output_feature, record.source_features) for record in pre.get_feature_lineage()]
    assert lineage == [("num_1", ("1",)), ("num_0", ("0",))]


def test_labels_colliding_as_strings_are_rejected():
    frame = pd.DataFrame({1: [1.0, 2.0, 3.0], "1": [4.0, 5.0, 6.0]})
    with pytest.raises(PretabDataError, match="unique when converted to strings"):
        Preprocessor(numerical_method="minmax").fit(frame)


def test_get_feature_names_out_accepts_the_integer_labels_seen_at_fit():
    """The ColumnTransformer is fitted on string labels, so ``input_features`` given as
    the original integer labels must still be accepted (it already was before)."""
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(40, 2)))
    pre = Preprocessor(numerical_method="minmax").fit(X)
    np.testing.assert_array_equal(pre.get_feature_names_out([0, 1]), pre.get_feature_names_out())
