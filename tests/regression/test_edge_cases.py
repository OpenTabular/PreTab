"""P8.6 edge-case regression suite.

End-to-end :class:`~pretab.Preprocessor` guards for the recurring production edge
cases: constant features, ``custombin`` discretization alongside string
categoricals, duplicate support points, missing values, and unseen categories.
These pin the *observable* Preprocessor behaviour (shape, finiteness,
determinism, and typed errors) so later refactors cannot silently change
edge-case handling.
"""

import numpy as np
import pandas as pd
import pytest

from pretab import Preprocessor
from pretab.exceptions import PretabDataError


def _finite(array) -> bool:
    return bool(np.isfinite(array).all())


# --- constant features ---------------------------------------------------------


def test_constant_numeric_graceful_method_is_finite():
    X = pd.DataFrame({"const": np.full(50, 3.14), "vary": np.linspace(0.0, 1.0, 50)})
    out = Preprocessor(numerical_method="minmax").fit_transform(X, return_array=True)
    assert out.shape == (50, 2)
    assert _finite(out)


def test_constant_numeric_with_error_policy_raises():
    X = pd.DataFrame({"const": np.full(50, 3.14), "vary": np.linspace(0.0, 1.0, 50)})
    with pytest.raises(PretabDataError):
        Preprocessor(numerical_method="minmax", policy={"constant": "error"}).fit(X)


# --- custombin + string categoricals ------------------------------------------


def test_custombin_is_deterministic_and_integer_coded():
    rng = np.random.RandomState(11)
    X = pd.DataFrame({"num": rng.rand(120), "cat": rng.choice(["red", "green", "blue"], size=120)})
    kwargs = {
        "numerical_method": "custombin",
        "categorical_method": "one-hot",
        "output_dim": 5,
        "target_aware": False,
        "placement_strategy": "quantile",
    }
    out1 = Preprocessor(**kwargs).fit_transform(X, return_array=True)
    out2 = Preprocessor(**kwargs).fit_transform(X, return_array=True)
    np.testing.assert_array_equal(out1, out2)
    assert _finite(out1)
    # The custombin block is integer-valued bin codes in [0, output_dim).
    bin_col = out1[:, 0]
    assert np.all(bin_col == np.floor(bin_col))
    assert bin_col.min() >= 0
    assert bin_col.max() < 5


# --- duplicate support points --------------------------------------------------


def test_duplicate_support_points_are_handled():
    # 90% of the mass sits on a single value, forcing duplicate knot candidates.
    X = pd.DataFrame({"x": np.concatenate([np.full(90, 0.5), np.linspace(0.0, 1.0, 30)])})
    p = Preprocessor(
        numerical_method="bspline",
        output_dim=8,
        target_aware=False,
        placement_strategy="quantile",
    ).fit(X)
    out = p.transform(X, return_array=True)
    assert _finite(out)
    assert out.shape[0] == len(X)


# --- missing values ------------------------------------------------------------


def test_missing_values_imputed_by_default():
    X = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]})
    out = Preprocessor(numerical_method="minmax").fit_transform(X, return_array=True)
    assert not np.isnan(out).any()


def test_missing_values_separate_state_marks_rows():
    X = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]})
    p = Preprocessor(numerical_method="minmax", missing_policy="separate_state").fit(X)
    names = list(p.get_feature_names_out())
    assert any(n.endswith("__missing") for n in names)
    out = p.transform(X, return_array=True)
    assert _finite(out)


# --- unseen categories ---------------------------------------------------------


def test_unseen_categories_do_not_crash():
    train = pd.DataFrame({"c": ["a", "b", "a", "b", "a", "b"]})
    unseen = pd.DataFrame({"c": ["a", "b", "c", "a", "z", "b"]})
    p = Preprocessor(categorical_method="one-hot").fit(train)
    out = p.transform(unseen, return_array=True)
    assert _finite(out)
    # handle_unknown="ignore" encodes unseen categories as an all-zero row.
    assert out[2].sum() == 0.0
    assert out[4].sum() == 0.0
