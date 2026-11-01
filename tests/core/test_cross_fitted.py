"""Tests for :class:`~pretab.core.supervised.CrossFittedTransformer` (Phase 7, P7.3).

Verifies out-of-fold (leakage-free) training features, the all-data model used by
``transform``, spec bookkeeping (``cross_fitted`` / ``n_folds``), and input
validation.
"""

import warnings
from typing import cast

import numpy as np
import pytest
from sklearn.model_selection import KFold

from pretab import CrossFittedTransformer, LeakageWarning
from pretab.exceptions import IncompatibleParamsError, InvalidParamError
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


def test_wrapped_preprocessor_target_aware_blocks_stay_out_of_fold(array_with_rare_category):
    """Only the target-aware block is refit, on each fold's training rows."""
    import pandas as pd
    from sklearn.base import clone

    from pretab import Preprocessor

    X, y = array_with_rare_category
    cross_fitted = CrossFittedTransformer(Preprocessor(output_dim=4, random_state=0), n_folds=5, random_state=0)
    out = cross_fitted.fit_transform(X, y)
    supervised = _split_columns(cross_fitted, "num_feature_0_ple")
    assert len(supervised) == 4

    wrapped = cast(Preprocessor, cross_fitted.estimator_)
    block = wrapped.column_transformer_.named_transformers_["num_feature_0"]
    column = pd.DataFrame({"feature_0": X[:, 0]})
    for train_idx, test_idx in KFold(n_splits=5, shuffle=True, random_state=0).split(X):
        fold_block = clone(block).fit(column.iloc[train_idx], y[train_idx])
        np.testing.assert_allclose(out[np.ix_(test_idx, supervised)], fold_block.transform(column.iloc[test_idx]))
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


def test_width_mismatch_message_does_not_blame_adaptive_sizing_only():
    from sklearn.preprocessing import OneHotEncoder

    x = np.array(["a"] * 99 + ["b"]).reshape(-1, 1)  # the fold holding out "b" never sees it
    y = np.arange(100.0)
    cross_fitted = CrossFittedTransformer(
        OneHotEncoder(handle_unknown="ignore", sparse_output=False), n_folds=5, random_state=0
    )
    with pytest.raises(IncompatibleParamsError, match="depends on the training rows"):
        cross_fitted.fit_transform(x, y)


# --- DataFrame input is passed through (issue #58) --------------------------------


@pytest.fixture
def mixed_frame():
    import pandas as pd

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {"num": rng.normal(size=300), "city": rng.choice(["a", "b", "c"], size=300)},
        index=rng.permutation(1000)[:300],
    )
    return X, X["num"].to_numpy() ** 2 + rng.normal(scale=0.1, size=300)


def test_wrapped_preprocessor_keeps_dataframe_column_types(mixed_frame):
    """Regression guard for issue #58: np.asarray turned the frame into an object
    array, so the wrapped Preprocessor treated every column as categorical."""
    from pretab import Preprocessor

    X, y = mixed_frame
    direct = Preprocessor(output_dim=6, random_state=0).fit(X, y)
    cross_fitted = CrossFittedTransformer(Preprocessor(output_dim=6, random_state=0), n_folds=3, random_state=0)
    out = cross_fitted.fit_transform(X, y)

    wrapped = cast(Preprocessor, cross_fitted.estimator_)
    assert wrapped.numerical_features_ == ["num"]
    assert wrapped.categorical_features_ == ["city"]
    assert out.shape == (300, direct.total_output_dim_)
    assert list(cross_fitted.get_feature_names_out()) == list(direct.get_feature_names_out())
    assert len(np.unique(out[:, 0])) > 1  # a supervised encoding, not an "unseen category" code
    assert cross_fitted.transform(X).shape == out.shape


def test_wrapped_column_transformer_selects_by_column_name(mixed_frame):
    from sklearn.compose import ColumnTransformer

    X, y = mixed_frame
    wrapped = ColumnTransformer([("ple", PLETransformer(output_dim=4), ["num"])])
    out = CrossFittedTransformer(wrapped, n_folds=3, random_state=0).fit_transform(X, y)
    assert out.shape == (300, 4)


def test_wrapped_preprocessor_accepts_feature_preprocessing_by_column_name(mixed_frame):
    from pretab import Preprocessor

    X, y = mixed_frame
    wrapped = Preprocessor(
        feature_preprocessing={"num": "ple"}, numerical_method="minmax", output_dim=4, random_state=0
    )
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=3, random_state=0)
    out = cross_fitted.fit_transform(X, y)

    assert list(cross_fitted.get_feature_names_out()) == [f"num_num_ple{i}" for i in range(4)] + ["cat_city"]
    assert out.shape == (300, 5)


def test_one_dimensional_input_is_still_a_single_column():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    out = CrossFittedTransformer(PLETransformer(output_dim=4), n_folds=3, random_state=0).fit_transform(x, x**2)
    assert out.shape == (200, 4)


@pytest.mark.parametrize("seed", range(5))
def test_cross_fitting_a_discrete_feature_keeps_a_fixed_width(seed):
    """Regression guard for issue #57: tied features gave fold-dependent widths."""
    from pretab.transformers import RBFExpansionTransformer

    rng = np.random.default_rng(seed)
    x = rng.integers(0, 6, size=200).astype(float).reshape(-1, 1)
    y = np.sin(x[:, 0]) + rng.normal(0, 0.5, size=200)
    out = CrossFittedTransformer(
        RBFExpansionTransformer(output_dim=10, target_aware=True), n_folds=5, random_state=0
    ).fit_transform(x, y)
    assert out.shape == (200, 10)


@pytest.mark.parametrize("wrapped_kind", ["preprocessor", "column_transformer", "pipeline"])
def test_dataframe_representation_spec_defaults_to_fitted_names(mixed_frame, wrapped_kind):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    from pretab import Preprocessor

    X, y = mixed_frame
    if wrapped_kind == "preprocessor":
        wrapped = Preprocessor(output_dim=4, random_state=0)
    else:
        wrapped = ColumnTransformer([("ple", PLETransformer(output_dim=4), ["num"])])
        if wrapped_kind == "pipeline":
            wrapped = Pipeline([("features", wrapped)])
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=3, random_state=0).fit(X, y)

    spec = cross_fitted.get_representation_spec()
    assert spec.input_features == ("num", "city")
    assert spec.output_features == tuple(cross_fitted.get_feature_names_out())
    assert spec.cross_fitted is True
    assert spec.n_folds == 3
    assert spec.uses_target is True
    assert spec == type(spec).from_dict(spec.to_dict())
    assert cross_fitted.get_representation_spec(["num", "city"]) == spec


def test_dataframe_fold_rows_keep_index_dtypes_and_target_alignment(mixed_frame, monkeypatch):
    import pandas as pd

    from pretab import Preprocessor

    X, y = mixed_frame
    original = X.copy(deep=True)
    fits, transforms = [], []
    original_fit = Preprocessor.fit
    original_cross_fit_fold = Preprocessor._cross_fit_fold
    original_transform = Preprocessor.transform

    def record_fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        fits.append((X.copy(deep=True), np.asarray(y).copy()))
        return original_fit(self, X, y)

    def record_transform(self, X):
        assert isinstance(X, pd.DataFrame)
        transforms.append(X.copy(deep=True))
        return original_transform(self, X)

    def record_cross_fit_fold(self, X, y):
        assert isinstance(X, pd.DataFrame)
        fits.append((X.copy(deep=True), np.asarray(y).copy()))
        return original_cross_fit_fold(self, X, y)

    monkeypatch.setattr(Preprocessor, "fit", record_fit)
    monkeypatch.setattr(Preprocessor, "_cross_fit_fold", record_cross_fit_fold)
    monkeypatch.setattr(Preprocessor, "transform", record_transform)
    cross_fitted = CrossFittedTransformer(
        Preprocessor(output_dim=4, output_format="dense", random_state=0), n_folds=3, random_state=0
    )
    cross_fitted.fit_transform(X, y)
    splits = list(KFold(n_splits=3, shuffle=True, random_state=0).split(X))

    assert len(fits) == 4
    pd.testing.assert_frame_equal(fits[0][0], X)
    np.testing.assert_array_equal(fits[0][1], y)
    for (actual_X, actual_y), (train_idx, _) in zip(fits[1:], splits, strict=True):
        pd.testing.assert_frame_equal(actual_X, X.iloc[train_idx])
        np.testing.assert_array_equal(actual_y, y[train_idx])
    assert len(transforms) == 3
    for actual_X, (_, test_idx) in zip(transforms, splits, strict=True):
        pd.testing.assert_frame_equal(actual_X, X.iloc[test_idx])

    unseen = X.iloc[:5][["city", "num"]]
    cross_fitted.transform(unseen)
    pd.testing.assert_frame_equal(transforms[-1], unseen)
    pd.testing.assert_frame_equal(X, original)


@pytest.mark.parametrize("method", ["ple", "bspline", "mspline", "ispline", "rbf"])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_cross_fitting_lightgbm_refits_only_target_blocks_and_keeps_the_full_fit(method, task):
    pytest.importorskip("lightgbm")
    import pandas as pd
    from sklearn.base import clone

    from pretab import Preprocessor

    rng = np.random.default_rng(0)
    X = pd.DataFrame({1: rng.normal(size=160), 2: rng.choice([True, False], size=160)})
    y = 2 * X[1].to_numpy() + X[2].to_numpy() + rng.normal(scale=0.1, size=len(X))
    if task == "classification":
        y = np.where(y >= np.median(y), "high", "low")
    preprocessor = Preprocessor(
        numerical_method=method,
        categorical_method="one-hot",
        placement_strategy="lightgbm",
        output_dim=6,
        output_format="dense",
        task=task,
        random_state=0,
    )
    cross_fitted = CrossFittedTransformer(preprocessor, n_folds=3, task=task, random_state=0)
    out = cross_fitted.fit_transform(X, y)
    full = cross_fitted.transform(X)
    assert out.shape == full.shape == (len(X), 8)
    assert not np.allclose(out[:, :6], full[:, :6])
    np.testing.assert_array_equal(out[:, 6:], full[:, 6:])
    reference = clone(preprocessor)
    assert isinstance(reference, Preprocessor)
    reference_out = reference.fit(X, y).transform(X)
    assert isinstance(reference_out, np.ndarray)
    np.testing.assert_allclose(full, reference_out)
    assert len(cross_fitted.get_feature_names_out()) == 8
    fitted = cross_fitted.estimator_
    assert isinstance(fitted, Preprocessor)
    assert [item.uses_target for item in fitted.get_feature_lineage()] == [True] * 6 + [False] * 2


@pytest.mark.parametrize("labels", [[1, 2], [np.int64(7), np.int64(9)], [1, "flag"], ["amount", "flag"]])
def test_cross_fitting_routes_integer_labels_and_boolean_columns_like_full_fit(labels):
    import pandas as pd

    from pretab import Preprocessor

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {labels[0]: rng.normal(size=200), labels[1]: rng.choice([True, False], size=200)},
        index=rng.permutation(1000)[:200],
    )
    original = X.copy(deep=True)
    y = np.sin(X.iloc[:, 0].to_numpy()) + X.iloc[:, 1].to_numpy() + rng.normal(scale=0.1, size=200)

    def cross_fit(frame):
        return CrossFittedTransformer(
            Preprocessor(output_dim=4, output_format="dense", random_state=0), n_folds=5, random_state=0
        ).fit_transform(frame, y)

    expected = cross_fit(X.set_axis(["amount", "flag"], axis=1))
    np.testing.assert_allclose(cross_fit(X), expected)
    pd.testing.assert_frame_equal(X, original)
