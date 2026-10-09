"""Standalone numerical transformers follow scikit-learn's feature-name contract.

Fitted on a DataFrame they record ``feature_names_in_``, name their outputs after
the columns, reject renamed or reordered columns at ``transform`` and validate
``get_feature_names_out(input_features)``. Before, column names were dropped
(outputs were named ``x0_...``) and a reordered frame was silently accepted.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import (
    check_dataframe_column_names_consistency,
    check_transformer_get_feature_names_out_pandas,
)

from pretab.transformers import (
    BSplineTransformer,
    CubicRegressionSplineTransformer,
    FourierFeatureTransformer,
    NumericBinningTransformer,
    PLETransformer,
    RBFExpansionTransformer,
    ReLUExpansionTransformer,
    TensorProductSplineTransformer,
)

pytestmark = pytest.mark.filterwarnings("ignore::pretab.exceptions.LeakageWarning")

TRANSFORMERS = [
    lambda: RBFExpansionTransformer(output_dim=3),
    lambda: ReLUExpansionTransformer(output_dim=3),
    lambda: PLETransformer(output_dim=3, random_state=0),
    lambda: NumericBinningTransformer(output_dim=3, encode="onehot"),
    lambda: BSplineTransformer(output_dim=5),
    lambda: CubicRegressionSplineTransformer(),
    lambda: FourierFeatureTransformer(),
    lambda: TensorProductSplineTransformer(),
]
IDS = ["rbf", "relu", "ple", "binning", "bspline", "cubicspline", "fourier", "tensorspline"]


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"age": rng.normal(40, 10, 100), "income": rng.uniform(1, 5, 100)})
    return X, X["age"].to_numpy() / 10 + rng.normal(size=100)


@pytest.mark.parametrize("make", TRANSFORMERS, ids=IDS)
def test_dataframe_fit_records_feature_names_and_uses_them(make, frame):
    X, y = frame
    transformer = make().fit(X, y)
    assert list(transformer.feature_names_in_) == ["age", "income"]
    names = list(transformer.get_feature_names_out())
    assert all(("age" in name) or ("income" in name) for name in names)
    assert not any(name.startswith("x0") for name in names)


@pytest.mark.parametrize("make", TRANSFORMERS, ids=IDS)
def test_reordered_columns_are_rejected(make, frame):
    X, y = frame
    transformer = make().fit(X, y)
    with pytest.raises(ValueError, match="feature names"):
        transformer.transform(X[["income", "age"]])


@pytest.mark.parametrize("make", TRANSFORMERS, ids=IDS)
def test_array_after_a_dataframe_fit_warns_like_scikit_learn(make, frame):
    X, y = frame
    transformer = make().fit(X, y)
    with pytest.warns(UserWarning, match="does not have valid feature names"):
        transformer.transform(X.to_numpy())


@pytest.mark.parametrize("make", TRANSFORMERS, ids=IDS)
def test_input_features_are_validated(make, frame):
    X, y = frame
    transformer = make().fit(X, y)
    with pytest.raises(ValueError):
        transformer.get_feature_names_out(["age"])
    with pytest.raises(ValueError, match="not equal to feature_names_in_"):
        transformer.get_feature_names_out(["a", "b"])


@pytest.mark.parametrize("make", TRANSFORMERS, ids=IDS)
def test_array_fit_keeps_generic_names_and_drops_stale_feature_names(make, frame):
    X, y = frame
    transformer = make().fit(X, y).fit(X.to_numpy(), y)
    assert not hasattr(transformer, "feature_names_in_")
    assert "x0" in str(transformer.get_feature_names_out()[0])


@pytest.mark.parametrize(
    "make",
    [
        lambda: RBFExpansionTransformer(output_dim=3),
        lambda: PLETransformer(output_dim=3, random_state=0),
        lambda: NumericBinningTransformer(output_dim=3),
        lambda: BSplineTransformer(output_dim=5),
    ],
    ids=["rbf", "ple", "binning", "bspline"],
)
def test_scikit_learn_feature_name_checks_pass(make):
    estimator = make()
    name = type(estimator).__name__
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_dataframe_column_names_consistency(name, estimator)
        check_transformer_get_feature_names_out_pandas(name, estimator)
