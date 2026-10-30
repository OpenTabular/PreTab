"""Column labels the ColumnTransformer cannot take verbatim.

Regression guards for issue #71: labels starting with ``_`` or containing ``__``
were embedded into estimator names that scikit-learn rejects.
Regression guards for issue #55: integer labels were read as *positions* by
scikit-learn's ColumnTransformer, silently swapping columns.
"""

from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

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


@pytest.mark.parametrize("label", ["_score", "income__log", "__index_level_0__", "trailing__"])
def test_numerical_labels_with_underscores_fit(label):
    values = np.random.default_rng(0).normal(size=50)
    frame = pd.DataFrame({label: values})
    pre = Preprocessor(numerical_method="minmax").fit(frame)
    assert pre.get_feature_names_out().tolist() == [f"num_{label}"]
    blocks = _blocks(pre.transform(frame, return_array=False))
    assert list(blocks) == [f"num_{label}"]
    np.testing.assert_allclose(blocks[f"num_{label}"].ravel(), (values - values.min()) / np.ptp(values))
    assert pre.output_dims_ == {label: 1}


def test_categorical_and_expanded_labels_with_underscores_fit():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"_city": rng.choice(["a", "b"], 50), "x__y": rng.normal(size=50)})
    pre = Preprocessor(categorical_method="one-hot", output_dim=3).fit(frame, rng.normal(size=50))
    names = pre.get_feature_names_out().tolist()
    assert names == ["num_x__y_ple0", "num_x__y_ple1", "num_x__y_ple2", "cat__city_a", "cat__city_b"]
    assert [record.output_feature for record in pre.get_feature_lineage()] == names
    assert list(_blocks(pre.transform(frame, return_array=False))) == ["num_x__y", "cat__city"]
    assert pre.output_dims_ == {"x__y": 3, "_city": 2}


def test_positional_step_names_do_not_leak_into_names_or_block_keys():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"_city": rng.choice(["a", "b"], 50), "x": rng.normal(size=50)})
    pre = Preprocessor(numerical_method="minmax", categorical_method="one-hot", missing_policy="separate_state")
    pre.fit(frame)
    # "_city" runs under an internal positional step name; every public name uses the label.
    assert [name for name, _, _ in pre.column_transformer_.transformers_] == ["num_x", "cat_col1"]
    names = [
        "num_x__representation__x",
        "num_x__missing__x__missing",
        "cat__city__representation___city_a",
        "cat__city__representation___city_b",
        "cat__city__missing___city__missing",
    ]
    assert pre.get_feature_names_out().tolist() == names
    assert [record.output_feature for record in pre.get_feature_lineage()] == names
    assert list(_blocks(pre.transform(frame, return_array=False))) == ["num_x", "cat__city"]


def test_underscore_labels_do_not_collide_with_other_steps():
    rng = np.random.default_rng(0)
    # "_col0" would map to the fallback name of the first step without collision handling.
    frame = pd.DataFrame({"_a": rng.normal(size=30), "col0": rng.normal(size=30)})
    pre = Preprocessor(numerical_method="minmax").fit(frame)
    assert pre.get_feature_names_out().tolist() == ["num__a", "num_col0"]


def test_preprocessor_chains_after_a_pandas_column_transformer():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"age": rng.normal(40, 10, 50), "income": rng.normal(50, 10, 50)})
    upstream = ColumnTransformer([("imp", SimpleImputer(), ["age", "income"])]).set_output(transform="pandas")
    pipe = make_pipeline(upstream, Preprocessor(numerical_method="minmax")).fit(X, rng.normal(size=50))
    assert cast(Preprocessor, pipe[-1]).get_feature_names_out().tolist() == ["num_imp__age", "num_imp__income"]
    assert np.asarray(pipe.transform(X)).shape == (50, 2)


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


def test_underscore_and_integer_labels_preserve_routing_and_public_metadata():
    frame = pd.DataFrame(
        {
            "_score": [10.0, 20, 30, 40, 50, 60],
            1: [1.0, 2, 3, 4, 5, 6],
            "col0": [6.0, 5, 4, 3, 2, 1],
            0: pd.array([True, False, None, False, True, True], dtype="boolean"),
        }
    )
    original = frame.copy(deep=True)
    pre = _passthrough(categorical_method="int").fit(frame)
    labels = ["_score", 1, "col0", 0]
    names = ["num__score", "num_1", "num_col0", "cat_0"]
    assert pre.column_transformer_.transformers_[0][0] == "num_col0_1"
    assert pre.get_feature_names_out().tolist() == names
    assert pre.output_dims_ == dict.fromkeys(labels, 1)
    numerical, categorical, _ = pre.get_feature_info(verbose=False)
    assert list(numerical) == labels[:-1]
    assert list(categorical) == [0]
    assert [(record.output_feature, record.source_features) for record in pre.get_feature_lineage()] == [
        (name, (str(label),)) for name, label in zip(names, labels, strict=True)
    ]
    for data in (frame, frame[[0, "col0", 1, "_score"]], frame.to_numpy(na_value=np.nan)):
        out = _blocks(pre.transform(data))
        assert list(out) == names
        for label in labels[:-1]:
            np.testing.assert_array_equal(out[f"num_{label}"].ravel(), frame[label].to_numpy())
        np.testing.assert_array_equal(out["cat_0"].ravel(), [2.0, 1.0, 2.0, 1.0, 2.0, 2.0])
    pd.testing.assert_frame_equal(frame, original)
