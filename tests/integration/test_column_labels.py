"""Column labels the ColumnTransformer cannot take verbatim.

Regression guards for issue #71: labels starting with ``_`` or containing ``__``
were embedded into estimator names that scikit-learn rejects.
"""

from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from pretab import Preprocessor


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
