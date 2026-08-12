"""``random_state`` + missing-value host control on the Preprocessor.

Verifies that the reproducibility seed and the imputation knobs
(``numerical_imputation`` / ``categorical_imputation`` / ``add_missing_indicator``)
are exposed on the :class:`Preprocessor`, drive the per-column pipelines, keep
prior behavior when unset, and make stochastic fits reproducible -- so a
standalone user or an embedding host (DeepTab) can pin a global seed and choose a
missing-value policy.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from pretab.preprocessor import Preprocessor
from pretab.transformers import RBFExpansionTransformer


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        {
            "a": rng.randn(200),
            "b": rng.rand(200) * 5,
        }
    )
    y = pd.Series(X["a"] * 2 + rng.randn(200) * 0.1)
    return X, y


def _numerical_transformer(pre, feature):
    """Return the fitted numerical method transformer for ``feature``."""
    name = f"num_{feature}"
    for tname, transformer, _cols in pre.column_transformer_.transformers_:
        if tname == name:
            return transformer.steps[-1][1]
    raise AssertionError(f"no numerical transformer named {name}")


# --- exposure & round-trip ------------------------------------------------- #


def test_new_params_defaults_and_get_params():
    pre = Preprocessor()
    assert pre.random_state is None
    assert pre.numerical_imputation == "median"
    assert pre.categorical_imputation == "most_frequent"
    assert pre.add_missing_indicator is False
    params = pre.get_params()
    assert params["random_state"] is None
    assert params["numerical_imputation"] == "median"
    assert params["categorical_imputation"] == "most_frequent"
    assert params["add_missing_indicator"] is False


def test_clone_preserves_new_params():
    pre = Preprocessor(random_state=99, numerical_imputation="mean", add_missing_indicator=True)
    cloned = clone(pre)
    assert isinstance(cloned, Preprocessor)
    assert cloned.random_state == 99
    assert cloned.numerical_imputation == "mean"
    assert cloned.add_missing_indicator is True


# --- random_state forwarding ----------------------------------------------- #


@pytest.mark.parametrize("method", ["ple", "rbf"])
def test_random_state_forwarded_when_set(data, method):
    X, y = data
    pre = Preprocessor(numerical_method=method, random_state=123).fit(X, y)
    assert _numerical_transformer(pre, "a").random_state == 123


def test_unset_random_state_preserves_component_defaults(data):
    X, y = data
    # PLE keeps its own default seed (51) when the Preprocessor seed is unset.
    ple = _numerical_transformer(Preprocessor(numerical_method="ple").fit(X, y), "a")
    assert ple.random_state == 51
    # Feature maps stay unseeded (None) when the Preprocessor seed is unset.
    rbf = _numerical_transformer(Preprocessor(numerical_method="rbf").fit(X, y), "a")
    assert rbf.random_state is None


@pytest.mark.parametrize("method", ["ple", "rbf", "quantile"])
def test_fixed_random_state_makes_fit_reproducible(data, method):
    X, y = data
    o1 = Preprocessor(numerical_method=method, random_state=7).fit(X, y).transform(X, return_array=True)
    o2 = Preprocessor(numerical_method=method, random_state=7).fit(X, y).transform(X, return_array=True)
    assert isinstance(o1, np.ndarray)
    assert isinstance(o2, np.ndarray)
    np.testing.assert_array_equal(o1, o2)


# --- missing-value / imputation policy ------------------------------------- #


def test_numerical_imputation_median_fills_nan(data):
    X, y = data
    X = X.copy()
    X.iloc[0, 0] = np.nan
    # Default "median" imputes before PLE, so NaN is filled and the fit succeeds.
    pre = Preprocessor(numerical_method="ple").fit(X, y)
    out = pre.transform(X, return_array=True)
    assert isinstance(out, np.ndarray)
    assert np.isfinite(out).all()


def test_numerical_imputation_none_lets_nan_reach_transformer(data):
    X, y = data
    X = X.copy()
    X.iloc[0, 0] = np.nan
    # Disabling imputation lets NaN reach PLE, which requires finite input.
    pre = Preprocessor(numerical_method="ple", numerical_imputation=None)
    with pytest.raises(ValueError):
        pre.fit(X, y)


def test_add_missing_indicator_appends_columns(data):
    X, y = data
    X = X.copy()
    X.iloc[0, 0] = np.nan
    base = Preprocessor(numerical_method="standardization").fit(X, y).transform(X, return_array=True)
    with_ind = (
        Preprocessor(numerical_method="standardization", add_missing_indicator=True)
        .fit(X, y)
        .transform(X, return_array=True)
    )
    assert isinstance(base, np.ndarray)
    assert isinstance(with_ind, np.ndarray)
    assert with_ind.shape[1] > base.shape[1]


# --- transformer / helper level seeding ------------------------------------ #


def test_rbf_transformer_seeded_centers_reproducible(data):
    X, y = data
    r1 = RBFExpansionTransformer(output_dim=5, target_aware=True, random_state=3).fit(X.values, y.values)
    r2 = RBFExpansionTransformer(output_dim=5, target_aware=True, random_state=3).fit(X.values, y.values)
    for a, b in zip(r1.centers_, r2.centers_, strict=True):
        np.testing.assert_array_equal(a, b)
