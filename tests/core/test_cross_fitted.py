"""Tests for :class:`~pretab.core.supervised.CrossFittedTransformer` (Phase 7, P7.3).

Verifies out-of-fold (leakage-free) training features, the all-data model used by
``transform``, spec bookkeeping (``cross_fitted`` / ``n_folds``), and input
validation.
"""

import warnings
from typing import Any, cast

import numpy as np
import pytest
from sklearn.model_selection import KFold

from pretab import CrossFittedTransformer, LeakageWarning
from pretab.exceptions import IncompatibleParamsError, InvalidParamError, PretabDataError
from pretab.transformers import PLETransformer


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(400, 1))
    y = (X[:, 0] > 0).astype(float) + rng.normal(scale=0.1, size=400)
    return X, y


def test_fit_transform_is_out_of_fold(data):
    """Each training row is encoded by a fold model that never saw it."""
    X, y = data
    cf = CrossFittedTransformer(PLETransformer(output_dim=8), n_folds=5, shuffle=True, random_state=0)
    Xt = cf.fit_transform(X, y)

    assert Xt.shape == (X.shape[0], 8)
    splitter = KFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, test_idx in splitter.split(X):
        fold = PLETransformer(output_dim=8).fit(X[train_idx], y[train_idx])
        expected = fold.transform(X[test_idx])
        np.testing.assert_allclose(Xt[test_idx], expected)


def test_cross_fitting_emits_no_leakage_warning(data):
    X, y = data
    cf = CrossFittedTransformer(PLETransformer(output_dim=6), n_folds=4, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LeakageWarning)
        cf.fit_transform(X, y)


def test_transform_uses_all_data_model(data):
    """``transform`` on unseen data uses ``estimator_`` fit on all training data."""
    X, y = data
    cf = CrossFittedTransformer(PLETransformer(output_dim=6), n_folds=4, random_state=0)
    cf.fit(X, y)

    reference = PLETransformer(output_dim=6).fit(X, y)
    X_new = np.linspace(-2, 2, 25).reshape(-1, 1)
    np.testing.assert_allclose(cf.transform(X_new), reference.transform(X_new))


def test_spec_records_cross_fitting(data):
    X, y = data
    cf = CrossFittedTransformer(PLETransformer(output_dim=6), n_folds=5, random_state=0)
    cf.fit(X, y)
    spec = cf.get_representation_spec(["f0"])

    assert spec.cross_fitted is True
    assert spec.n_folds == 5
    assert spec.uses_target is True
    assert spec.family == "piecewise_linear"
    assert spec == type(spec).from_dict(spec.to_dict())


def test_contract_properties(data):
    X, y = data
    cf = CrossFittedTransformer(PLETransformer(), n_folds=3)
    assert cf.requires_y is True
    assert cf.is_supervised is True
    cf.fit(X, y)
    assert cf.uses_target_ is True


def test_feature_names_delegate(data):
    X, y = data
    cf = CrossFittedTransformer(PLETransformer(output_dim=6), n_folds=3, random_state=0)
    cf.fit(X, y)
    reference = PLETransformer(output_dim=6).fit(X, y)
    np.testing.assert_array_equal(cf.get_feature_names_out(["f0"]), reference.get_feature_names_out(["f0"]))


def test_requires_y(data):
    X, _ = data
    cf = CrossFittedTransformer(PLETransformer(), n_folds=3)
    with pytest.raises(IncompatibleParamsError):
        cf.fit(X, None)


def test_invalid_n_folds(data):
    X, y = data
    with pytest.raises(InvalidParamError):
        CrossFittedTransformer(PLETransformer(), n_folds=1).fit(X, y)


def test_invalid_task_rejected(data):
    X, y = data
    with pytest.raises(InvalidParamError, match="task"):
        CrossFittedTransformer(PLETransformer(), task="classificaton").fit_transform(X, y)


# --- a wrapped Preprocessor keeps one encoding across folds (issue #79) -------------


@pytest.fixture
def array_with_rare_category():
    rng = np.random.default_rng(0)
    X = np.column_stack(
        [
            rng.integers(0, 100_000, 300),  # many distinct integers: numerical
            rng.integers(1, 9, 300),  # unique ratio just below the default cat_cutoff
            rng.choice([1, 2, 3], 300),  # category codes
        ]
    )
    X[7, 2] = 0  # a category some folds never see, sorted before the others
    y = (X[:, 0] / 50_000.0) ** 2 + (X[:, 2] == 2) + rng.normal(scale=0.1, size=300)
    return X, y


def _split_columns(cross_fitted, prefix):
    names = list(cross_fitted.get_feature_names_out())
    return [i for i, name in enumerate(names) if name.startswith(prefix)]


def test_wrapped_preprocessor_out_of_fold_codes_match_transform(array_with_rare_category):
    """Each fold used to refit the whole Preprocessor: category codes shifted when a
    category was missing from a fold, and column types were re-detected on fewer rows."""
    from pretab import Preprocessor

    X, y = array_with_rare_category
    cross_fitted = CrossFittedTransformer(Preprocessor(output_dim=4, random_state=0), n_folds=5, random_state=0)
    out = cross_fitted.fit_transform(X, y)
    full = np.asarray(cross_fitted.transform(X))

    unsupervised = _split_columns(cross_fitted, "cat_")
    assert len(unsupervised) == 2  # both integer-coded columns are categorical on the full data
    np.testing.assert_array_equal(out[:, unsupervised], full[:, unsupervised])
    np.testing.assert_array_equal(np.unique(out[:, unsupervised[1]]), [1.0, 2.0, 3.0, 4.0])


def _refit_representation_per_fold(pipeline, column, y):
    """Expected out-of-fold rows of a fitted block pipeline whose last step is its
    target-aware representation: the helper steps before it keep their all-data fit."""
    from sklearn.base import clone

    helpers, representation = pipeline[:-1], pipeline[-1]
    blocks = {}
    for train_idx, test_idx in KFold(n_splits=5, shuffle=True, random_state=0).split(column):
        fold = cast(Any, clone(representation)).fit(helpers.transform(column.iloc[train_idx]), y[train_idx])
        blocks[tuple(test_idx)] = fold.transform(helpers.transform(column.iloc[test_idx]))
    out = np.empty((len(column), next(iter(blocks.values())).shape[1]))
    for test_idx, rows in blocks.items():
        out[list(test_idx)] = rows
    return out


def _fitted_block(cross_fitted, name):
    from pretab import Preprocessor

    return cast(Preprocessor, cross_fitted.estimator_).column_transformer_.named_transformers_[name]


def test_wrapped_preprocessor_target_aware_blocks_stay_out_of_fold(array_with_rare_category):
    """Only the target-aware representation is refit, on each fold's training rows; the
    helper steps before it (imputer, scaler) keep their all-data fit."""
    import pandas as pd

    from pretab import Preprocessor

    X, y = array_with_rare_category
    cross_fitted = CrossFittedTransformer(Preprocessor(output_dim=4, random_state=0), n_folds=5, random_state=0)
    out = cross_fitted.fit_transform(X, y)
    supervised = _split_columns(cross_fitted, "num_feature_0_ple")
    assert len(supervised) == 4

    block = _fitted_block(cross_fitted, "num_feature_0")
    assert [name for name, _ in block.steps] == ["imputer", "minmax", "ple"]
    expected = _refit_representation_per_fold(block, pd.DataFrame({"feature_0": X[:, 0]}), y)
    np.testing.assert_array_equal(out[:, supervised], expected)
    assert not np.allclose(out[:, supervised], np.asarray(cross_fitted.transform(X))[:, supervised])


def test_wrapped_preprocessor_with_one_hot_keeps_a_fixed_width(array_with_rare_category):
    from pretab import Preprocessor

    X, y = array_with_rare_category
    wrapped = Preprocessor(categorical_method="one-hot", output_dim=4, random_state=0)
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=5, random_state=0)
    out = cross_fitted.fit_transform(X, y)
    one_hot = _split_columns(cross_fitted, "cat_")

    assert out.shape == (len(X), len(cross_fitted.get_feature_names_out()))
    np.testing.assert_array_equal(out[:, one_hot], np.asarray(cross_fitted.transform(X))[:, one_hot])


def test_wrapped_unsupervised_preprocessor_matches_transform(array_with_rare_category):
    from pretab import Preprocessor

    X, y = array_with_rare_category
    wrapped = Preprocessor(numerical_method="minmax", target_aware=False, placement_strategy="quantile")
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=3, random_state=0)
    np.testing.assert_array_equal(cross_fitted.fit_transform(X, y), np.asarray(cross_fitted.transform(X)))


def test_wrapped_preprocessor_keeps_the_all_data_column_types():
    """25 integer levels are categorical on 1000 rows (25/1000 < cat_cutoff) but were
    re-detected as numerical on each 800-row fold, so the out-of-fold features held
    min-max values where transform returned category codes."""
    from pretab import Preprocessor

    rng = np.random.default_rng(0)
    X = rng.integers(0, 25, (1000, 1))
    cross_fitted = CrossFittedTransformer(Preprocessor(numerical_method="minmax"), n_folds=5, random_state=0)
    out = cross_fitted.fit_transform(X, rng.normal(size=1000))

    assert cast(Preprocessor, cross_fitted.estimator_).categorical_features_ == ["feature_0"]
    np.testing.assert_array_equal(out, np.asarray(cross_fitted.transform(X)))
    np.testing.assert_array_equal(np.unique(out), np.arange(1, 26))


def _one_missing_value(X):
    X[7, 0] = np.nan  # a single missing value: the fold holding out row 7 trains on none
    return X


@pytest.mark.parametrize(
    ("X", "method"),
    [
        (_one_missing_value(np.random.default_rng(0).normal(size=(200, 2))), {"numerical_method": "minmax"}),
        (_one_missing_value(np.random.default_rng(0).normal(size=(200, 2))), {"numerical_method": "standardization"}),
        (
            _one_missing_value(np.random.default_rng(0).normal(size=(200, 2))),
            {"numerical_method": "bspline", "target_aware": False, "placement_strategy": "uniform"},
        ),
        (_one_missing_value(np.random.default_rng(0).choice(["a", "b", "c"], (200, 1)).astype(object)), {}),
        (
            _one_missing_value(np.random.default_rng(0).choice(["a", "b", "c"], (200, 1)).astype(object)),
            {"categorical_method": "one-hot"},
        ),
    ],
    ids=["minmax", "standardization", "bspline-unsupervised", "categorical-int", "categorical-one-hot"],
)
def test_wrapped_preprocessor_shares_indicator_blocks_that_do_not_use_y(X, method):
    """Blocks that do not use y keep their all-data fit, missing indicator included: each
    fold used to refit them, and a fold whose training rows hold no missing value
    dropped the indicator column."""
    from pretab import Preprocessor

    y = np.random.default_rng(1).normal(size=len(X))
    wrapped = Preprocessor(add_missing_indicator=True, output_dim=6, random_state=0, **method)
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=5, random_state=0)
    out = cross_fitted.fit_transform(X, y)
    assert out.shape[1] == len(cross_fitted.get_feature_names_out()) > X.shape[1]  # indicator columns kept
    np.testing.assert_array_equal(out, np.asarray(cross_fitted.transform(X)))


def test_wrapped_preprocessor_reads_the_target_use_of_an_indicator_block():
    """With add_missing_indicator, a representation is fitted on its column and the
    indicator; deciding whether it used y must not depend on its input names."""
    from pretab import Preprocessor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 1))
    X[::4, 0] = np.nan  # every fold's training rows hold missing values
    y = np.nan_to_num(X[:, 0]) + rng.normal(scale=0.1, size=200)
    wrapped = Preprocessor(numerical_method="rbf", add_missing_indicator=True, output_dim=4, random_state=0)
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=5, random_state=0)
    out = cross_fitted.fit_transform(X, y)

    full = np.asarray(cross_fitted.transform(X))
    assert out.shape == full.shape == (200, len(cross_fitted.get_feature_names_out()))
    assert not np.allclose(out, full)  # refit on every fold


def test_width_mismatch_names_the_general_cause():
    from sklearn.preprocessing import OneHotEncoder

    x = np.array(["a"] * 99 + ["b"]).reshape(-1, 1)  # the fold holding out "b" never sees it
    y = np.arange(100.0)
    cross_fitted = CrossFittedTransformer(
        OneHotEncoder(handle_unknown="ignore", sparse_output=False), n_folds=5, random_state=0
    )
    with pytest.raises(IncompatibleParamsError, match="same columns on every fold"):
        cross_fitted.fit_transform(x, y)


# --- a wrapped Preprocessor shares its missing indicators across folds (issue #89) --


@pytest.fixture
def array_with_one_missing_value():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    X[7, 0] = np.nan  # the fold holding out row 7 trains on no missing value
    y = np.nan_to_num(X[:, 0]) + rng.normal(scale=0.1, size=200)
    return X, y


@pytest.mark.parametrize(
    ("method", "missing"),
    [
        ("ple", {"add_missing_indicator": True}),
        ("ple", {"missing_policy": "impute_with_indicator"}),
        ("rbf", {"add_missing_indicator": True}),
        ("relu", {"add_missing_indicator": True}),
        ("sigmoid", {"add_missing_indicator": True}),
        ("tanh", {"add_missing_indicator": True}),
    ],
)
def test_wrapped_preprocessor_keeps_its_missing_indicator_columns(array_with_one_missing_value, method, missing):
    """Each fold refit the block's imputer, whose missing indicator emits no column when
    the fold's training rows contain no missing value, so the width check failed."""
    from pretab import Preprocessor

    X, y = array_with_one_missing_value

    def cross_fit(**missing_config):
        wrapped = Preprocessor(numerical_method=method, output_dim=4, random_state=0, **missing_config)
        cross_fitted = CrossFittedTransformer(wrapped, n_folds=5, random_state=0)
        return cross_fitted, cross_fitted.fit_transform(X, y)

    with_indicator, out = cross_fit(**missing)
    _, imputed_only = cross_fit(missing_policy="impute")
    names = [str(name) for name in with_indicator.get_feature_names_out()]
    indicator = [i for i, name in enumerate(names) if "missingindicator" in name]

    assert out.shape == np.asarray(with_indicator.transform(X)).shape
    assert indicator  # the indicator columns of the all-data fit are kept on every fold
    # The value's out-of-fold representation is the one cross-fitting gives without an indicator.
    np.testing.assert_array_equal(np.delete(out, indicator, axis=1), imputed_only)


def test_wrapped_preprocessor_refits_only_the_representation_after_the_imputer(array_with_one_missing_value):
    """The all-data imputer (with its indicator) and scaler encode every fold; only the
    PLE is refit, on the imputed value and the indicator of the fold's training rows."""
    import pandas as pd

    from pretab import Preprocessor

    X, y = array_with_one_missing_value
    wrapped = Preprocessor(numerical_method="ple", output_dim=4, add_missing_indicator=True, random_state=0)
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=5, random_state=0)
    out = cross_fitted.fit_transform(X, y)

    block = _fitted_block(cross_fitted, "num_feature_0")
    assert [name for name, _ in block.steps] == ["imputer", "minmax", "ple"]
    columns = _split_columns(cross_fitted, "num_feature_0")
    assert len(columns) == 8  # the imputer feeds the value and its indicator to the PLE
    expected = _refit_representation_per_fold(block, pd.DataFrame({"feature_0": X[:, 0]}), y)
    np.testing.assert_array_equal(out[:, columns], expected)


@pytest.mark.xfail(
    strict=True,
    raises=PretabDataError,
    reason="Until #62 keeps the imputer's indicator out of the representation, a spline basis "
    "is also fitted on the indicator, which is constant on a fold without missing values.",
)
def test_wrapped_preprocessor_keeps_the_indicator_next_to_a_target_aware_spline(array_with_one_missing_value):
    from pretab import Preprocessor

    X, y = array_with_one_missing_value
    wrapped = Preprocessor(numerical_method="bspline", add_missing_indicator=True, random_state=0)
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=5, random_state=0)
    assert cross_fitted.fit_transform(X, y).shape == np.asarray(cross_fitted.transform(X)).shape


def test_wrapped_preprocessor_keeps_the_separate_state_missing_branch(array_with_one_missing_value):
    """A separate-state block refits only its representation branch: the imputer imputes
    the held-out missing value with the all-data median, as transform does."""
    import pandas as pd

    from pretab import Preprocessor

    X, y = array_with_one_missing_value
    wrapped = Preprocessor(numerical_method="ple", output_dim=4, missing_policy="separate_state", random_state=0)
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=5, random_state=0)
    out = cross_fitted.fit_transform(X, y)

    representation = dict(_fitted_block(cross_fitted, "num_feature_0").transformer_list)["representation"]
    expected = _refit_representation_per_fold(representation, pd.DataFrame({"feature_0": X[:, 0]}), y)
    np.testing.assert_array_equal(out[:, _split_columns(cross_fitted, "num_feature_0__representation")], expected)
    missing = _split_columns(cross_fitted, "num_feature_0__missing")
    np.testing.assert_array_equal(out[:, missing].ravel(), np.isnan(X[:, 0]).astype(float))
