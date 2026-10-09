"""Column labels the ColumnTransformer cannot take verbatim.

Regression guards for issue #55 (integer labels were read as *positions* by
scikit-learn's ColumnTransformer, silently swapping columns) and issue #71 (labels
starting with ``_`` or containing ``__`` were embedded into estimator names that
scikit-learn rejects).
"""

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
    return Preprocessor(numerical_method="none", scaling=None, output_structure="blocks", **kwargs)


def test_integer_labels_route_each_column_to_its_own_block(int_labelled):
    out = _passthrough().fit_transform(int_labelled)
    np.testing.assert_array_equal(out["num_1"].ravel(), int_labelled[1].to_numpy())
    np.testing.assert_array_equal(out["num_0"].ravel(), int_labelled[0].to_numpy())


def test_feature_preprocessing_with_integer_keys_targets_the_labelled_column(int_labelled):
    pre = Preprocessor(feature_preprocessing={1: "minmax", 0: "none"}, scaling=None, output_structure="blocks")
    out = pre.fit_transform(int_labelled)
    np.testing.assert_allclose(out["num_1"].ravel(), np.linspace(0.0, 1.0, 6))
    np.testing.assert_array_equal(out["num_0"].ravel(), int_labelled[0].to_numpy())


def test_non_contiguous_integer_labels_fit_and_transform():
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 3))).drop(columns=[0])
    pre = Preprocessor(numerical_method="minmax").fit(frame)
    assert pre.get_feature_names_out().tolist() == ["num_1", "num_2"]
    assert pre.transform(frame).shape == (20, 2)


def test_integer_labels_are_matched_by_label_at_transform(int_labelled):
    pre = _passthrough(output_format="dense").fit(int_labelled)
    reordered = int_labelled[[0, 1]]
    np.testing.assert_array_equal(pre.transform(reordered)["num_1"], pre.transform(int_labelled)["num_1"])


def test_integer_label_metadata_uses_the_input_labels(int_labelled):
    pre = Preprocessor(numerical_method="minmax").fit(int_labelled)
    assert pre.output_dims_ == {1: 1, 0: 1}
    numerical, _, _ = pre.get_feature_info(verbose=False)
    assert list(numerical) == [1, 0]
    assert [record.source_features for record in pre.get_feature_lineage()] == [("1",), ("0",)]


def test_labels_colliding_as_strings_are_rejected():
    frame = pd.DataFrame({1: [1.0, 2.0, 3.0], "1": [4.0, 5.0, 6.0]})
    with pytest.raises(PretabDataError, match="unique when converted to strings"):
        Preprocessor(numerical_method="minmax").fit(frame)


def test_fit_does_not_modify_the_callers_column_labels(int_labelled):
    Preprocessor(numerical_method="minmax").fit(int_labelled)
    assert list(int_labelled.columns) == [1, 0]


@pytest.mark.parametrize("label", ["_score", "income__log", "__index_level_0__", "trailing__"])
def test_numerical_labels_with_underscores_fit(label):
    frame = pd.DataFrame({label: np.random.default_rng(0).normal(size=50)})
    pre = Preprocessor(numerical_method="minmax").fit(frame)
    assert pre.get_feature_names_out().tolist() == [f"num_{label}"]
    assert list(pre.transform(frame, return_array=False)) == [f"num_{label}"]
    assert pre.output_dims_ == {label: 1}


def test_categorical_and_expanded_labels_with_underscores_fit():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"_city": rng.choice(["a", "b"], 50), "x__y": rng.normal(size=50)})
    pre = Preprocessor(categorical_method="one-hot", output_dim=3).fit(frame, rng.normal(size=50))
    names = pre.get_feature_names_out().tolist()
    assert names == ["num_x__y_ple0", "num_x__y_ple1", "num_x__y_ple2", "cat__city_a", "cat__city_b"]
    assert [record.output_feature for record in pre.get_feature_lineage()] == names
    assert pre.output_dims_ == {"x__y": 3, "_city": 2}


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
    assert pipe[-1].get_feature_names_out().tolist() == ["num_imp__age", "num_imp__income"]
    assert pipe.transform(X).shape == (50, 2)
