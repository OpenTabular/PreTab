"""Phase 18: ``random_state`` + ``handle_missing`` host control on the Preprocessor.

Verifies that both knobs are exposed on the :class:`Preprocessor`, propagate to
the underlying numerical methods, keep prior behavior when unset, and make
stochastic fits reproducible -- so a standalone user or an embedding host
(DeepTab) can pin a global seed and choose a missing-value policy.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from pretab.core.exceptions import PretabDataError
from pretab.pipeline import get_numerical_transformer_steps
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
    assert pre.handle_missing == "median"
    params = pre.get_params()
    assert params["random_state"] is None
    assert params["handle_missing"] == "median"


def test_clone_preserves_new_params():
    pre = Preprocessor(random_state=99, handle_missing="error")
    cloned = clone(pre)
    assert isinstance(cloned, Preprocessor)
    assert cloned.random_state == 99
    assert cloned.handle_missing == "error"


# --- random_state forwarding ----------------------------------------------- #

@pytest.mark.parametrize("method", ["ple", "rbf"])
def test_random_state_forwarded_when_set(data, method):
    X, y = data
    pre = Preprocessor(numerical_method=method, random_state=123).fit(X, y)
    assert _numerical_transformer(pre, "a").random_state == 123


def test_unset_random_state_preserves_component_defaults(data):
    X, y = data
    # PLE keeps its own default seed (51) when the Preprocessor seed is unset.
    ple = _numerical_transformer(
        Preprocessor(numerical_method="ple").fit(X, y), "a"
    )
    assert ple.random_state == 51
    # Feature maps stay unseeded (None) when the Preprocessor seed is unset.
    rbf = _numerical_transformer(
        Preprocessor(numerical_method="rbf").fit(X, y), "a"
    )
    assert rbf.random_state is None


@pytest.mark.parametrize("method", ["ple", "rbf", "quantile"])
def test_fixed_random_state_makes_fit_reproducible(data, method):
    X, y = data
    o1 = Preprocessor(numerical_method=method, random_state=7).fit(X, y).transform(X, return_array=True)
    o2 = Preprocessor(numerical_method=method, random_state=7).fit(X, y).transform(X, return_array=True)
    assert isinstance(o1, np.ndarray)
    assert isinstance(o2, np.ndarray)
    np.testing.assert_array_equal(o1, o2)


# --- handle_missing policy ------------------------------------------------- #

def test_handle_missing_forwarded_to_ple(data):
    X, y = data
    pre = Preprocessor(numerical_method="ple", handle_missing="error").fit(X, y)
    assert _numerical_transformer(pre, "a").handle_missing == "error"


def test_handle_missing_median_imputes_nan(data):
    X, y = data
    X = X.copy()
    X.iloc[0, 0] = np.nan
    # Default "median" keeps the mean imputer, so NaN is filled before PLE.
    pre = Preprocessor(numerical_method="ple", handle_missing="median").fit(X, y)
    out = pre.transform(X, return_array=True)
    assert isinstance(out, np.ndarray)
    assert np.isfinite(out).all()


def test_handle_missing_error_rejects_nan(data):
    X, y = data
    X = X.copy()
    X.iloc[0, 0] = np.nan
    # "error" drops the imputer, so NaN reaches PLE which raises.
    pre = Preprocessor(numerical_method="ple", handle_missing="error")
    with pytest.raises(ValueError):
        pre.fit(X, y)


# The "error" policy must hold for *every* numerical method, not just PLE.
#
# ``handle_missing`` only ever dropped the imputer and was forwarded to PLE
# alone, so the guarantee depended on whether the chosen transformer happened to
# notice NaN. The scikit-learn scalers ignore missing values by design and the
# PreTab families declare ``allow_nan``, so everything except PLE silently
# emitted a NaN-contaminated matrix -- and the unsupervised feature maps were
# worse still, because ``np.percentile`` over a NaN column makes every center NaN.
@pytest.mark.parametrize(
    "method",
    ["minmax", "standardization", "robust", "quantile", "rbf", "relu", "tanh",
     "cubicspline", "pspline", "tprs", "none", "ple"],
)
def test_handle_missing_error_rejects_nan_for_every_method(data, method):
    X, y = data
    X = X.copy()
    X.iloc[0, 0] = np.nan

    with pytest.raises(ValueError):
        Preprocessor(numerical_method=method, handle_missing="error").fit(X, y)


@pytest.mark.parametrize("method", ["minmax", "rbf", "cubicspline"])
def test_handle_missing_median_still_imputes_for_every_method(data, method):
    X, y = data
    X = X.copy()
    X.iloc[0, 0] = np.nan

    out = Preprocessor(numerical_method=method, handle_missing="median").fit(X, y).transform(
        X, return_array=True
    )
    assert isinstance(out, np.ndarray)
    assert np.isfinite(out).all()


def test_handle_missing_error_raises_a_pretab_error_naming_the_option(data):
    X, y = data
    X = X.copy()
    X.iloc[0, 0] = np.nan

    with pytest.raises(PretabDataError, match="handle_missing='error'"):
        Preprocessor(numerical_method="minmax", handle_missing="error").fit(X, y)


def test_nan_check_step_only_present_when_erroring():
    erroring = [name for name, _ in get_numerical_transformer_steps("minmax", add_imputer=False)]
    imputing = [name for name, _ in get_numerical_transformer_steps("minmax", add_imputer=True)]

    assert "nan_check" in erroring and "imputer" not in erroring
    assert "imputer" in imputing and "nan_check" not in imputing


def test_handle_missing_error_still_transforms_clean_data(data):
    X, y = data
    out = Preprocessor(numerical_method="rbf", handle_missing="error").fit(X, y).transform(
        X, return_array=True
    )
    assert isinstance(out, np.ndarray)
    assert np.isfinite(out).all()


# --- transformer / helper level seeding ------------------------------------ #

def test_rbf_transformer_seeded_centers_reproducible(data):
    X, y = data
    r1 = RBFExpansionTransformer(output_dim=5, target_aware=True, random_state=3).fit(X.values, y.values)
    r2 = RBFExpansionTransformer(output_dim=5, target_aware=True, random_state=3).fit(X.values, y.values)
    for a, b in zip(r1.centers_, r2.centers_):
        np.testing.assert_array_equal(a, b)
