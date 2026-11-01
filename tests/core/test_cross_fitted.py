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
    original_transform = Preprocessor.transform

    def record_fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        fits.append((X.copy(deep=True), np.asarray(y).copy()))
        return original_fit(self, X, y)

    def record_transform(self, X):
        assert isinstance(X, pd.DataFrame)
        transforms.append(X.copy(deep=True))
        return original_transform(self, X)

    monkeypatch.setattr(Preprocessor, "fit", record_fit)
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
