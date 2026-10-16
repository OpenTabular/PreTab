"""Tests for :class:`~pretab.core.supervised.CrossFittedTransformer` (Phase 7, P7.3).

Verifies out-of-fold (leakage-free) training features, the all-data model used by
``transform``, spec bookkeeping (``cross_fitted`` / ``n_folds``), and input
validation.
"""

import warnings

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


# --- fit_transform returns the kind of output transform returns (issue #90) ---------


@pytest.fixture
def integer_array():
    """A numerical and a categorical integer column whose types and categories every fold sees."""
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.integers(0, 100_000, 300), rng.choice([1, 2, 3], 300)])
    y = (X[:, 0] / 50_000.0) ** 2 + (X[:, 1] == 2) + rng.normal(scale=0.1, size=300)
    return X, y


@pytest.fixture
def object_array():
    """A float and a string column in one object array."""
    rng = np.random.default_rng(0)
    X = np.empty((300, 2), dtype=object)
    X[:, 0] = rng.normal(size=300)
    X[:, 1] = rng.choice(["Berlin", "Paris", "Rome"], 300)
    return X, X[:, 0].astype(float) ** 2 + rng.normal(scale=0.1, size=300)


def _cross_fit(wrapped, X, y):
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=3, random_state=0)
    return cross_fitted, cross_fitted.fit_transform(X, y)


def test_fit_transform_keeps_strings_left_unchanged(object_array):
    """The out-of-fold rows were written into a float buffer, so categorical_method=None
    (strings passed through unchanged) raised in fit_transform but not in transform."""
    from pretab import Preprocessor

    X, y = object_array
    cross_fitted, out = _cross_fit(Preprocessor(categorical_method=None, random_state=0), X, y)

    assert out.dtype == cross_fitted.transform(X).dtype == object
    np.testing.assert_array_equal(out, X)


def test_fit_transform_stacks_out_of_fold_values_with_strings(object_array):
    from sklearn.compose import ColumnTransformer

    X, y = object_array
    wrapped = ColumnTransformer([("ple", PLETransformer(output_dim=4), [0]), ("keep", "passthrough", [1])])
    _, out = _cross_fit(wrapped, X, y)
    _, numeric = _cross_fit(PLETransformer(output_dim=4), X[:, [0]].astype(float), y)

    assert out.dtype == object
    np.testing.assert_allclose(out[:, :4].astype(float), numeric)
    np.testing.assert_array_equal(out[:, 4], X[:, 1])


def test_fit_transform_keeps_sparse_output(integer_array):
    """fit_transform could not stack sparse fold output, while transform returned CSR."""
    from scipy import sparse as sp

    from pretab import Preprocessor

    X, y = integer_array
    params = {"categorical_method": "one-hot", "output_dim": 4, "random_state": 0}
    cross_fitted, out = _cross_fit(Preprocessor(output_format="sparse", **params), X, y)
    _, dense = _cross_fit(Preprocessor(**params), X, y)

    assert sp.issparse(out)
    assert out.format == cross_fitted.transform(X).format == "csr"
    np.testing.assert_allclose(out.toarray(), dense)


def test_fit_transform_keeps_the_configured_dtype(integer_array):
    from pretab import Preprocessor

    X, y = integer_array
    cross_fitted, out = _cross_fit(Preprocessor(dtype=np.float32, output_dim=4, random_state=0), X, y)
    _, native = _cross_fit(Preprocessor(output_dim=4, random_state=0), X, y)

    assert out.dtype == cross_fitted.transform(X).dtype == np.float32
    np.testing.assert_allclose(out, native, rtol=1e-6)


def test_fit_transform_returns_the_dataframe_transform_returns(integer_array):
    import pandas as pd

    from pretab import Preprocessor

    X, y = integer_array
    cross_fitted, out = _cross_fit(Preprocessor(output_dim=4, random_state=0).set_output(transform="pandas"), X, y)
    full = cross_fitted.transform(X)
    _, array = _cross_fit(Preprocessor(output_dim=4, random_state=0), X, y)

    assert isinstance(out, pd.DataFrame)
    pd.testing.assert_index_equal(out.columns, full.columns)
    pd.testing.assert_index_equal(out.index, full.index)  # the Preprocessor numbers its rows
    np.testing.assert_allclose(out.to_numpy(), array)


def test_fit_transform_returns_polars_like_transform(integer_array):
    pl = pytest.importorskip("polars")
    from pretab import Preprocessor

    X, y = integer_array
    cross_fitted, out = _cross_fit(Preprocessor(output_dim=4, random_state=0).set_output(transform="polars"), X, y)
    _, array = _cross_fit(Preprocessor(output_dim=4, random_state=0), X, y)

    assert isinstance(out, pl.DataFrame)
    assert out.columns == cross_fitted.transform(X).columns
    np.testing.assert_allclose(out.to_numpy(), array)


def test_stacked_dataframe_folds_keep_the_input_index_they_carry():
    """scikit-learn's set_output copies the input index onto each fold's frame."""
    import pandas as pd

    from pretab.core.supervised import _stack_folds

    X = pd.DataFrame({"a": [10.0, 11.0, 12.0, 13.0]}, index=pd.Index([7, 3, 9, 1]))
    test_indices = [np.array([1, 3]), np.array([0, 2])]
    blocks = [X.iloc[rows] * 2 for rows in test_indices]

    pd.testing.assert_frame_equal(_stack_folds(blocks, test_indices, X), X * 2)
    renumbered = _stack_folds([block.reset_index(drop=True) for block in blocks], test_indices, X)
    pd.testing.assert_frame_equal(renumbered, (X * 2).reset_index(drop=True))


def test_fit_transform_rejects_folds_with_other_dataframe_columns():
    """Frames are stacked by column label: a fold that keeps other columns at the
    same width (here another top category) must raise, not widen the result with
    NaN-padded columns."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    # Unshuffled folds: fold 0 holds out rows 0-29 (all "a"), so its training rows keep
    # "b" as the frequent category while the all-data fit keeps "a" -- same width, other labels.
    X = np.empty((90, 2), dtype=object)
    X[:, 0] = np.linspace(-1.0, 1.0, 90)
    X[:, 1] = ["a"] * 30 + ["b"] * 30 + ["c"] * 20 + ["a"] * 10
    y = X[:, 0].astype(float) ** 2
    encoder = OneHotEncoder(max_categories=2, handle_unknown="infrequent_if_exist", sparse_output=False)
    transformer = ColumnTransformer([("ple", PLETransformer(output_dim=3), [0]), ("oh", encoder, [1])]).set_output(
        transform="pandas"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LeakageWarning)
        with pytest.raises(IncompatibleParamsError, match=r"same output columns across folds.*x1_a"):
            CrossFittedTransformer(transformer, n_folds=3, shuffle=False).fit_transform(X, y)


def test_cross_fitting_rejects_blocks(integer_array):
    from pretab import Preprocessor

    X, y = integer_array
    blocks = Preprocessor(output_structure="blocks", random_state=0)
    with pytest.raises(IncompatibleParamsError, match="dict of blocks"):
        CrossFittedTransformer(blocks, n_folds=3, random_state=0).fit_transform(X, y)
