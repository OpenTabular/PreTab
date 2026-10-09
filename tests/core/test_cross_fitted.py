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


def test_spec_after_a_dataframe_fit_names_the_columns(mixed_frame):
    """The default spec passed x0, x1, ... to get_feature_names_out, which a
    transformer fitted on a DataFrame rejects as not equal to feature_names_in_."""
    X, y = mixed_frame
    cf = CrossFittedTransformer(PLETransformer(output_dim=4), n_folds=3, random_state=0).fit(X[["num"]], y)
    spec = cf.get_representation_spec()

    assert spec.input_features == ("num",)
    assert spec.output_features == tuple(cf.get_feature_names_out())
    assert spec.cross_fitted is True


def test_spec_of_a_wrapped_preprocessor_names_the_columns(mixed_frame):
    """A wrapped estimator without its own spec takes the fallback path, which
    used to fail the same way for a Preprocessor fitted on a DataFrame."""
    from pretab import Preprocessor

    X, y = mixed_frame
    cf = CrossFittedTransformer(Preprocessor(output_dim=6, random_state=0), n_folds=3, random_state=0).fit(X, y)
    spec = cf.get_representation_spec()

    assert spec.input_features == ("num", "city")
    assert spec.output_features == tuple(cf.get_feature_names_out())
    assert (spec.cross_fitted, spec.n_folds) == (True, 3)


def test_one_dimensional_input_is_still_a_single_column():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    out = CrossFittedTransformer(PLETransformer(output_dim=4), n_folds=3, random_state=0).fit_transform(x, x**2)
    assert out.shape == (200, 4)


# --- a wrapped Preprocessor keeps one encoding across folds -------------------------


@pytest.fixture
def frame_with_rare_category():
    import pandas as pd

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "x": rng.normal(size=300),
            "rooms": rng.integers(1, 9, 300),  # unique ratio near the default cat_cutoff
            "city": rng.choice(["Berlin", "Paris", "Rome"], 300),
        }
    )
    X.loc[7, "city"] = "Amsterdam"  # a category some folds never see
    y = X["x"].to_numpy() ** 2 + (X["city"] == "Paris") + rng.normal(scale=0.1, size=300)
    return X, y


def _split_columns(cross_fitted, prefix):
    names = list(cross_fitted.get_feature_names_out())
    return [i for i, name in enumerate(names) if name.startswith(prefix)]


def test_wrapped_preprocessor_out_of_fold_codes_match_transform(frame_with_rare_category):
    """Each fold used to refit the whole Preprocessor: category codes shifted when a
    category was missing from a fold, and column types were re-detected on fewer rows."""
    from pretab import Preprocessor

    X, y = frame_with_rare_category
    cross_fitted = CrossFittedTransformer(Preprocessor(output_dim=4, random_state=0), n_folds=5, random_state=0)
    out = cross_fitted.fit_transform(X, y)
    full = np.asarray(cross_fitted.transform(X))

    unsupervised = _split_columns(cross_fitted, "cat_")
    assert unsupervised  # rooms and city are categorical on the full data
    np.testing.assert_array_equal(out[:, unsupervised], full[:, unsupervised])


def test_wrapped_preprocessor_target_aware_blocks_stay_out_of_fold(frame_with_rare_category):
    from pretab import Preprocessor

    X, y = frame_with_rare_category
    cross_fitted = CrossFittedTransformer(Preprocessor(output_dim=4, random_state=0), n_folds=5, random_state=0)
    out = cross_fitted.fit_transform(X, y)
    supervised = _split_columns(cross_fitted, "num_x_ple")
    assert not np.allclose(out[:, supervised], np.asarray(cross_fitted.transform(X))[:, supervised])


def test_wrapped_preprocessor_with_one_hot_keeps_a_fixed_width(frame_with_rare_category):
    from pretab import Preprocessor

    X, y = frame_with_rare_category
    wrapped = Preprocessor(categorical_method="one-hot", output_dim=4, random_state=0)
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=5, random_state=0)
    out = cross_fitted.fit_transform(X, y)
    assert out.shape == (len(X), len(cross_fitted.get_feature_names_out()))


def test_wrapped_unsupervised_preprocessor_matches_transform(frame_with_rare_category):
    from pretab import Preprocessor

    X, y = frame_with_rare_category
    wrapped = Preprocessor(numerical_method="minmax", target_aware=False, placement_strategy="quantile")
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=3, random_state=0)
    np.testing.assert_array_equal(cross_fitted.fit_transform(X, y), np.asarray(cross_fitted.transform(X)))


def test_cross_fitting_rejects_blocks(frame_with_rare_category):
    from pretab import Preprocessor

    X, y = frame_with_rare_category
    blocks = Preprocessor(output_structure="blocks", random_state=0)
    with pytest.raises(IncompatibleParamsError, match="dict of blocks"):
        CrossFittedTransformer(blocks, n_folds=3, random_state=0).fit_transform(X, y)


# --- a wrapped Preprocessor shares its missing indicators across folds --------------


@pytest.fixture
def frame_with_one_missing_value():
    import pandas as pd

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    X.loc[7, "a"] = np.nan  # the fold holding out row 7 trains on no missing value
    y = X["a"].fillna(0.0).to_numpy() + rng.normal(scale=0.1, size=200)
    return X, y


@pytest.mark.parametrize("missing", [{"add_missing_indicator": True}, {"missing_policy": "impute_with_indicator"}])
def test_wrapped_preprocessor_keeps_its_missing_indicator_columns(frame_with_one_missing_value, missing):
    """Each fold refit the block's MissingIndicator, which emits no column when the
    fold's training rows contain no missing value, so the width check failed."""
    from pretab import Preprocessor

    X, y = frame_with_one_missing_value
    wrapped = Preprocessor(numerical_method="ple", output_dim=4, random_state=0, **missing)
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=5, random_state=0)
    out = cross_fitted.fit_transform(X, y)

    assert out.shape == np.asarray(cross_fitted.transform(X)).shape
    indicator = _split_columns(cross_fitted, "num_a__missingindicator")
    assert len(indicator) == 1
    np.testing.assert_array_equal(out[:, indicator[0]], X["a"].isna().to_numpy(dtype=float))


def test_wrapped_preprocessor_refits_only_the_representation_next_to_an_indicator(frame_with_one_missing_value):
    from pretab import Preprocessor

    X, y = frame_with_one_missing_value

    def cross_fit(missing_policy):
        wrapped = Preprocessor(numerical_method="ple", output_dim=4, missing_policy=missing_policy, random_state=0)
        cross_fitted = CrossFittedTransformer(wrapped, n_folds=5, random_state=0)
        return cross_fitted, cross_fitted.fit_transform(X, y)

    with_indicator, out = cross_fit("impute_with_indicator")
    _, imputed_only = cross_fit("impute")
    names = list(with_indicator.get_feature_names_out())
    representation = [i for i, name in enumerate(names) if "missingindicator" not in name]
    np.testing.assert_allclose(out[:, representation], imputed_only)


def test_width_mismatch_names_the_general_cause():
    from sklearn.preprocessing import OneHotEncoder

    x = np.array(["a"] * 99 + ["b"]).reshape(-1, 1)  # the fold holding out "b" never sees it
    y = np.arange(100.0)
    cross_fitted = CrossFittedTransformer(OneHotEncoder(handle_unknown="ignore"), n_folds=5, random_state=0)
    with pytest.raises(IncompatibleParamsError, match="same columns on every fold"):
        cross_fitted.fit_transform(x, y)


# --- fit_transform returns the kind of output transform returns ---------------------


def _cross_fit(wrapped, X, y):
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=3, random_state=0)
    return cross_fitted, cross_fitted.fit_transform(X, y)


def test_fit_transform_keeps_strings_left_unchanged(frame_with_rare_category):
    """The out-of-fold rows were written into a float buffer, so categorical_method=None
    (strings passed through unchanged) raised in fit_transform but not in transform."""
    from pretab import Preprocessor

    X, y = frame_with_rare_category
    cross_fitted, out = _cross_fit(Preprocessor(categorical_method=None, output_dim=4, random_state=0), X, y)
    _, encoded = _cross_fit(Preprocessor(output_dim=4, random_state=0), X, y)

    assert out.dtype == cross_fitted.transform(X).dtype == object
    np.testing.assert_array_equal(out[:, _split_columns(cross_fitted, "cat_city")[0]], X["city"].to_numpy())
    supervised = _split_columns(cross_fitted, "num_x_ple")
    np.testing.assert_allclose(out[:, supervised].astype(float), encoded[:, supervised])


def test_fit_transform_keeps_sparse_output(frame_with_rare_category):
    """Sparse fold output was densified, while transform returned a CSR matrix."""
    from scipy import sparse as sp

    from pretab import Preprocessor

    X, y = frame_with_rare_category
    params = {"categorical_method": "one-hot", "output_dim": 4, "random_state": 0}
    cross_fitted, out = _cross_fit(Preprocessor(output_format="sparse", **params), X, y)
    _, dense = _cross_fit(Preprocessor(**params), X, y)

    assert sp.issparse(out)
    assert out.format == cross_fitted.transform(X).format == "csr"
    np.testing.assert_allclose(out.toarray(), dense)


def test_fit_transform_keeps_the_configured_dtype(frame_with_rare_category):
    from pretab import Preprocessor

    X, y = frame_with_rare_category
    cross_fitted, out = _cross_fit(Preprocessor(dtype=np.float32, output_dim=4, random_state=0), X, y)
    _, native = _cross_fit(Preprocessor(output_dim=4, random_state=0), X, y)

    assert out.dtype == cross_fitted.transform(X).dtype == np.float32
    np.testing.assert_allclose(out, native, rtol=1e-6)


def test_fit_transform_returns_the_dataframe_transform_returns(mixed_frame):
    import pandas as pd

    from pretab import Preprocessor

    X, y = mixed_frame
    cross_fitted, out = _cross_fit(Preprocessor(output_dim=4, random_state=0).set_output(transform="pandas"), X, y)
    full = cross_fitted.transform(X)
    _, array = _cross_fit(Preprocessor(output_dim=4, random_state=0), X, y)

    assert isinstance(out, pd.DataFrame)
    pd.testing.assert_index_equal(out.columns, full.columns)
    pd.testing.assert_index_equal(out.index, full.index)  # the Preprocessor numbers its rows
    np.testing.assert_allclose(out.to_numpy(), array)


def test_fit_transform_keeps_the_index_of_scikit_learn_dataframe_output(mixed_frame):
    import pandas as pd
    from sklearn.compose import ColumnTransformer

    X, y = mixed_frame

    def ple_columns(output):
        return ColumnTransformer([("ple", PLETransformer(output_dim=4), ["num"])]).set_output(transform=output)

    _, out = _cross_fit(ple_columns("pandas"), X, y)
    _, array = _cross_fit(ple_columns("default"), X, y)

    assert isinstance(out, pd.DataFrame)
    pd.testing.assert_index_equal(out.index, X.index)
    np.testing.assert_allclose(out.to_numpy(), array)


def test_fit_transform_returns_polars_like_transform(mixed_frame):
    pl = pytest.importorskip("polars")
    from pretab import Preprocessor

    X, y = mixed_frame
    cross_fitted, out = _cross_fit(Preprocessor(output_dim=4, random_state=0).set_output(transform="polars"), X, y)
    _, array = _cross_fit(Preprocessor(output_dim=4, random_state=0), X, y)

    assert isinstance(out, pl.DataFrame)
    assert out.columns == cross_fitted.transform(X).columns
    np.testing.assert_allclose(out.to_numpy(), array)


def test_fit_transform_rejects_folds_with_other_dataframe_columns():
    """Frames are stacked by column label: a fold that keeps other columns at the
    same width (here another top category) must raise, not widen the result with
    NaN-padded columns."""
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    # Unshuffled folds: fold 0 holds out rows 0-29 (all "a"), so its training rows keep
    # "b" as the frequent category while the all-data fit keeps "a" -- same width, other labels.
    X = pd.DataFrame({"num": np.linspace(-1.0, 1.0, 90), "cat": ["a"] * 30 + ["b"] * 30 + ["c"] * 20 + ["a"] * 10})
    y = X["num"].to_numpy() ** 2
    encoder = OneHotEncoder(max_categories=2, handle_unknown="infrequent_if_exist", sparse_output=False)
    transformer = ColumnTransformer(
        [("ple", PLETransformer(output_dim=3), ["num"]), ("oh", encoder, ["cat"])]
    ).set_output(transform="pandas")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LeakageWarning)
        with pytest.raises(IncompatibleParamsError, match=r"same output columns across folds.*cat_a"):
            CrossFittedTransformer(transformer, n_folds=3, shuffle=False).fit_transform(X, y)
