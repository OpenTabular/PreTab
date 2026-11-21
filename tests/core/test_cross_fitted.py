"""Tests for :class:`~pretab.core.supervised.CrossFittedTransformer` (Phase 7, P7.3).

Verifies out-of-fold (leakage-free) training features, the all-data model used by
``transform``, spec bookkeeping (``cross_fitted`` / ``n_folds``), and input
validation.
"""

import warnings
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold

from pretab import CrossFittedTransformer, LeakageWarning, Preprocessor
from pretab.core._typing import TransformerLike
from pretab.exceptions import ConfigWarning, IncompatibleParamsError, InvalidParamError
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


@pytest.mark.parametrize("operation", ["fit", "fit_transform"])
def test_invalid_task_validation_rejected(data, operation):
    X, y = data
    transformer = CrossFittedTransformer(PLETransformer(), task_validation="WARN")
    assert transformer.get_params()["task_validation"] == "WARN"
    with pytest.raises(InvalidParamError, match=r"CrossFittedTransformer\.task_validation"):
        getattr(transformer, operation)(X, y)


def test_cross_fit_task_validation_survives_clone_and_fit(data):
    X, y = data
    transformer = CrossFittedTransformer(PLETransformer(), 3, "regression", False, 0)
    assert transformer.get_params(deep=False)["task_validation"] == "off"
    transformer.set_params(task_validation="raise")
    copied = clone(transformer)
    assert isinstance(copied, CrossFittedTransformer)
    copied.fit(X, y)
    assert copied.get_params(deep=False) == transformer.get_params(deep=False) | {"transformer": copied.transformer}


def test_nested_cross_fit_task_diagnostic_emits_once_per_operation(data):
    X, _ = data
    target = np.where(X[:, 0] < 0, "low", "high")
    inner = CrossFittedTransformer(PLETransformer(task="classification"), n_folds=3, task_validation="warn")
    outer = CrossFittedTransformer(inner, n_folds=3, task_validation="warn")

    for _ in range(2):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", ConfigWarning)
            outer.fit_transform(X, target)
        diagnostics = [item for item in recorded if item.category is ConfigWarning]
        assert len(diagnostics) == 1
        assert "cross-fitting uses task='regression'" in str(diagnostics[0].message)


def test_cross_fit_preflights_inner_policy_before_cloning(data, monkeypatch):
    X, target = data
    inner = Preprocessor(numerical_method="rbf", task="classification", task_validation="raise")
    outer = CrossFittedTransformer(inner)

    def reject_clone(*args, **kwargs):
        pytest.fail("cloned before checking the inner task policy")

    monkeypatch.setattr("pretab.core.supervised.clone", reject_clone)
    with pytest.raises(InvalidParamError, match=r"(?s)Preprocessor\.task.*target-aware placement.*continuous"):
        outer.fit(X, target)


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


@pytest.mark.parametrize("output_format", ["sparse", "auto"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sparse_mixed_frame_keeps_dtype_schema_and_missing_indicators(output_format, dtype):
    import pandas as pd
    from scipy import sparse as sp

    from pretab import Preprocessor

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "value": rng.normal(size=200),
            "city": np.repeat([f"city_{i}" for i in range(100)], 2),
            "flag": rng.choice([True, False], size=200),
        },
        index=pd.Index(rng.permutation(200), name="row"),
    )
    X.iloc[7, 0] = np.nan
    original = X.copy(deep=True)
    y = rng.normal(size=len(X))
    params = {
        "categorical_method": "one-hot",
        "output_dim": 6,
        "add_missing_indicator": True,
        "dtype": dtype,
        "random_state": 0,
    }
    wrapped, out = _cross_fit(Preprocessor(output_format=output_format, **params), X, y)
    _, expected = _cross_fit(Preprocessor(output_format="dense", **params), X, y)
    full = wrapped.transform(X)
    assert sp.issparse(out) and sp.issparse(full)
    assert out.format == full.format == "csr"
    assert out.dtype == full.dtype == dtype
    assert out.shape == expected.shape == full.shape == (len(X), 109)
    np.testing.assert_allclose(out.toarray(), expected)
    np.testing.assert_array_equal(out[:, 6:].toarray(), full[:, 6:].toarray())
    np.testing.assert_array_equal(out[:, 6].toarray().ravel(), X["value"].isna().astype(float))
    assert not np.allclose(out[:, :6].toarray(), full[:, :6].toarray())
    pd.testing.assert_frame_equal(X, original)


@pytest.mark.parametrize("sparse_format", ["csr", "csc", "coo", "bsr", "dia", "dok", "lil"])
@pytest.mark.parametrize("sparse_array", [False, True])
def test_stacking_sparse_folds_preserves_format_dtype_and_values_without_densifying(
    sparse_format, sparse_array, monkeypatch
):
    from scipy import sparse as sp

    from pretab.core.supervised import _stack_folds

    values = np.array([[1, 0], [0, 2], [3, 0], [0, 4]], dtype=np.float32)
    rows = [np.array([3, 1]), np.array([2, 0])]
    constructor = sp.csr_array if sparse_array else sp.csr_matrix
    blocks = [constructor(values[index]).asformat(sparse_format) for index in rows]

    def reject_densification(*args, **kwargs):
        pytest.fail("Sparse fold output must not be densified")

    with monkeypatch.context() as guard:
        guard.setattr(type(blocks[0]), "toarray", reject_densification)
        guard.setattr(sp.csr_matrix, "toarray", reject_densification)
        guard.setattr(sp.csr_array, "toarray", reject_densification)
        out = _stack_folds(blocks, rows, values)
    assert sp.issparse(out)
    assert out.format == sparse_format
    assert type(out) is type(blocks[0])
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out.toarray(), values)


@pytest.mark.parametrize("index_kind", ["shuffled", "duplicate", "multi"])
def test_pandas_output_preserves_the_input_index_through_the_public_wrapper(index_kind):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    from pretab import Preprocessor

    rng = np.random.default_rng(0)
    index = pd.Index(rng.permutation(120), name="row")
    if index_kind == "duplicate":
        index = pd.Index(np.arange(120) % 7, name="row")
    elif index_kind == "multi":
        index = pd.MultiIndex.from_arrays([np.arange(120) % 7, rng.permutation(120)], names=["group", "row"])
    X = pd.DataFrame({"value": rng.normal(size=120), "other": rng.normal(size=120)}, index=index)
    y = rng.normal(size=len(X))
    for transformer in [StandardScaler(), Preprocessor(numerical_method="minmax")]:
        wrapped, out = _cross_fit(transformer.set_output(transform="pandas"), X, y)
        full = wrapped.transform(X)
        assert isinstance(out, pd.DataFrame) and isinstance(full, pd.DataFrame)
        pd.testing.assert_index_equal(out.index, X.index)
        pd.testing.assert_index_equal(out.columns, full.columns)
        assert out.dtypes.equals(full.dtypes)
        if isinstance(transformer, Preprocessor):
            pd.testing.assert_frame_equal(out, full)


def test_pandas_fold_stacking_preserves_nullable_and_categorical_dtypes():
    import pandas as pd

    from pretab.core.supervised import _stack_folds

    X = pd.DataFrame(
        {
            "count": pd.array([1, None, 3, 4], dtype="Int64"),
            "flag": pd.array([True, False, None, True], dtype="boolean"),
            "text": pd.array(["a", None, "c", "d"], dtype="string"),
            "category": pd.Categorical(["a", "b", "a", "b"], ordered=True),
        },
        index=pd.Index([3, 3, 1, 7], name="row"),
    )
    rows = [np.array([3, 1]), np.array([0, 2])]
    pd.testing.assert_frame_equal(_stack_folds([X.iloc[index] for index in rows], rows, X), X)


@pytest.mark.parametrize("return_lists", [False, True])
def test_dense_fold_stacking_preserves_values_and_common_dtype(return_lists):
    from pretab.core.supervised import _stack_folds

    rows = [np.array([3, 1]), np.array([2, 0])]
    blocks = [np.array([[3], [1]], dtype=np.int16), np.array([[2.5], [0.5]], dtype=np.float32)]
    if return_lists:
        blocks = [block.tolist() for block in blocks]
    out = _stack_folds(blocks, rows, np.zeros((4, 1)))
    np.testing.assert_array_equal(out.ravel(), [0.5, 1, 2.5, 3])
    if not return_lists:
        assert out.dtype == np.float32


class _InvalidFoldOutput(TransformerMixin, BaseEstimator):
    def __init__(self, kind):
        self.kind = kind

    def fit(self, X, y=None):
        self.n_rows_ = len(X)
        return self

    def transform(self, X):
        import pandas as pd

        n_rows = len(X)
        if self.kind == "scalar":
            return 1.0
        if self.kind == "1d":
            return np.zeros(n_rows)
        if self.kind == "3d":
            return np.zeros((n_rows, 1, 1))
        if self.kind == "rows":
            return np.zeros((n_rows - 1, 1))
        if self.kind == "labels":
            return pd.DataFrame({self.get_feature_names_out()[0] if self.n_rows_ == 30 else 1: np.zeros(n_rows)})
        return pd.DataFrame(np.zeros((n_rows, 2)), columns=pd.Index(["b", "a"]))

    def get_feature_names_out(self, input_features=None):
        return np.array(["a", "b"] if self.kind == "order" else ["1"], dtype=object)


@pytest.mark.parametrize("kind", ["scalar", "1d", "3d", "rows"])
def test_cross_fitting_rejects_malformed_fold_output_with_a_clear_error(kind):
    with pytest.raises(IncompatibleParamsError, match="output shape"):
        CrossFittedTransformer(_InvalidFoldOutput(kind), n_folds=3).fit_transform(
            np.zeros((30, 1)), np.arange(30, dtype=float)
        )


@pytest.mark.parametrize("kind", ["labels", "order"])
def test_cross_fitting_rejects_changed_dataframe_labels_or_order(kind):
    with pytest.raises(IncompatibleParamsError, match="same output columns across folds"):
        CrossFittedTransformer(_InvalidFoldOutput(kind), n_folds=3).fit_transform(
            np.zeros((30, 1)), np.arange(30, dtype=float)
        )


@pytest.mark.parametrize("container", ["default", "pandas", "polars"])
def test_mixed_string_and_target_aware_output_keeps_row_order_and_missing_state(container):
    import pandas as pd

    from pretab import Preprocessor

    if container == "polars":
        pytest.importorskip("polars")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {"value": rng.normal(size=120), "city": rng.choice(["Berlin", "Paris"], 120)},
        index=pd.Index(np.arange(120) % 17, name="row"),
    )
    X.iloc[7, 0] = np.nan
    y = rng.normal(size=len(X))
    params = {
        "numerical_method": "ple",
        "categorical_method": None,
        "output_dim": 4,
        "add_missing_indicator": True,
        "output_format": "dense",
        "random_state": 0,
    }
    wrapped, out = _cross_fit(Preprocessor(**params).set_output(transform=container), X, y)
    _, reference = _cross_fit(Preprocessor(**params), X, y)
    full = wrapped.transform(X)
    if container == "pandas":
        assert isinstance(out, pd.DataFrame) and isinstance(full, pd.DataFrame)
        pd.testing.assert_index_equal(out.index, X.index)
        pd.testing.assert_index_equal(out.columns, full.columns)
        assert out.dtypes.equals(full.dtypes)
    elif container == "polars":
        pl = pytest.importorskip("polars")
        assert isinstance(out, pl.DataFrame) and isinstance(full, pl.DataFrame)
        assert out.schema == full.schema
    else:
        assert out.dtype == full.dtype == object
    values = out if isinstance(out, np.ndarray) else out.to_numpy()
    assert values.shape == (len(X), 6)
    np.testing.assert_allclose(values[:, :4].astype(float), reference[:, :4].astype(float))
    np.testing.assert_array_equal(values[:, 4].astype(float), X["value"].isna().astype(float))
    np.testing.assert_array_equal(values[:, 5], X["city"].to_numpy())


def test_polars_input_and_output_keep_nullable_columns_and_target_isolation():
    pl = pytest.importorskip("polars")
    from pretab import Preprocessor

    rng = np.random.default_rng(0)
    values = rng.normal(size=120)
    X = pl.DataFrame(
        {"value": [None if i == 7 else float(value) for i, value in enumerate(values)], "flag": [True, False] * 60}
    )
    y = rng.normal(size=len(X))
    params = {
        "output_dim": 4,
        "add_missing_indicator": True,
        "dtype": np.float32,
        "output_format": "dense",
        "random_state": 0,
    }
    wrapped, out = _cross_fit(Preprocessor(**params).set_output(transform="polars"), X, y)
    _, expected = _cross_fit(Preprocessor(**params), X, y)
    full = wrapped.transform(X)
    assert isinstance(out, pl.DataFrame) and isinstance(full, pl.DataFrame)
    assert out.schema == full.schema
    assert all(value == pl.Float32 for value in out.schema.values())
    assert out.shape == full.shape == (len(X), 6)
    np.testing.assert_allclose(out.to_numpy(), expected)
    np.testing.assert_array_equal(out.to_numpy()[:, 4], [int(i == 7) for i in range(len(X))])
    assert not np.allclose(out.to_numpy()[:, :4], full.to_numpy()[:, :4])
    assert X.null_count()["value"][0] == 1
    spec = wrapped.get_representation_spec()
    assert spec.input_features == ("value", "flag")
    assert spec.output_features == tuple(out.columns)
    assert spec.cross_fitted and spec.n_folds == 3


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


def _refit_representation_per_fold(pipeline, column, y, splitter=None):
    """Expected out-of-fold rows of a fitted block pipeline whose last step is its
    target-aware representation: the helper steps before it keep their all-data fit."""
    from sklearn.base import clone

    helpers, representation = pipeline[:-1], pipeline[-1]
    blocks = {}
    if splitter is None:
        splitter = KFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, test_idx in splitter.split(column, y):
        fold = cast(TransformerLike, clone(representation))
        fold.fit(helpers.transform(column.iloc[train_idx]), y[train_idx])
        blocks[tuple(test_idx)] = fold.transform(helpers.transform(column.iloc[test_idx]))
    out = np.empty((len(column), next(iter(blocks.values())).shape[1]))
    for test_idx, rows in blocks.items():
        out[list(test_idx)] = rows
    return out


def _fitted_block(cross_fitted, name):
    from pretab import Preprocessor

    return cast(Preprocessor, cross_fitted.estimator_).column_transformer_.named_transformers_[name]


class _JointTargetProjection(TransformerMixin, BaseEstimator):
    def __init__(self, target_aware=True):
        self.target_aware = target_aware

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        target = np.asarray(y)
        self.n_features_in_ = values.shape[1]
        self.fit_values_ = values.copy()
        self.fit_target_ = target.copy()
        self.weights_ = values.T @ target / len(values) if self.target_aware else np.arange(1, values.shape[1] + 1)
        self.target_mean_ = target.mean() if self.target_aware else 0.0
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.column_stack((values @ self.weights_, np.full(len(values), self.target_mean_)))

    def get_feature_names_out(self, input_features=None):
        return np.array(["joint_projection", "target_mean"], dtype=object)


@pytest.fixture
def grouped_cross_fit_data():
    import pandas as pd

    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "north": rng.normal(size=100),
            "east": rng.normal(size=100),
            "spare": rng.normal(size=100),
            "city": ["rare", *(["usual"] * 99)],
        }
    )
    frame.loc[7, "north"] = np.nan
    return frame, rng.normal(size=len(frame))


@pytest.mark.parametrize(
    ("target_usage", "target_aware"), [("optional", False), ("optional", True), ("required", True)]
)
@pytest.mark.parametrize("missing", ["none", "observed", "all"])
@pytest.mark.parametrize("output_format", ["dense", "sparse"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_grouped_target_aware_cross_fitting_preserves_frozen_prefix_and_missing_state(
    grouped_cross_fit_data, monkeypatch, target_usage, target_aware, missing, output_format, dtype, task
):
    import pandas as pd
    from scipy import sparse as sp

    from pretab import Preprocessor
    from pretab.compose.registry import TRANSFORMER_REGISTRY

    monkeypatch.setitem(
        TRANSFORMER_REGISTRY,
        "tprs",
        replace(
            TRANSFORMER_REGISTRY["tprs"],
            transformer_cls=_JointTargetProjection,
            target_usage=target_usage,
            allowed_args=(),
            placement_strategies=frozenset(),
        ),
    )
    fold_models = []
    original_fold = Preprocessor._cross_fit_fold

    def record_fold(self, X, y):
        fold = original_fold(self, X, y)
        fold_models.append(fold)
        return fold

    monkeypatch.setattr(Preprocessor, "_cross_fit_fold", record_fold)
    frame, target = grouped_cross_fit_data
    original_frame = frame.copy(deep=True)
    if task == "classification":
        target = (target > 0).astype(int)
    wrapped = CrossFittedTransformer(
        Preprocessor(
            numerical_method="minmax",
            categorical_method="one-hot",
            add_missing_indicator=missing == "observed",
            missing_policy="separate_state" if missing == "all" else None,
            output_format=output_format,
            dtype=dtype,
            task=task,
            feature_groups={
                "location": {
                    "columns": ["east", "north"],
                    "representation": {"method": "tprs", "target_aware": target_aware},
                }
            },
        ),
        n_folds=5,
        random_state=0,
        task=task,
    )
    out = wrapped.fit_transform(frame, target)
    full = wrapped.transform(frame)
    assert sp.issparse(out) is (output_format == "sparse")
    assert out.dtype == full.dtype == dtype
    if output_format == "sparse":
        assert out.format == full.format == "csr"
    out = out.toarray() if sp.issparse(out) else out
    full = full.toarray() if sp.issparse(full) else full
    fitted = cast(Preprocessor, wrapped.estimator_)
    span = fitted.column_transformer_.output_indices_["group_location"]
    block = _fitted_block(wrapped, "group_location")
    representation = block if missing == "none" else dict(block.transformer_list)["representation"]
    splitter_cls = StratifiedKFold if task == "classification" else KFold
    splitter = splitter_cls(n_splits=5, shuffle=True, random_state=0)
    expected = _refit_representation_per_fold(representation, frame[["east", "north"]], target, splitter).astype(dtype)
    np.testing.assert_array_equal(out[:, span.start : span.start + 2], expected)
    np.testing.assert_array_equal(out[:, : span.start], full[:, : span.start])
    np.testing.assert_array_equal(out[:, span.stop :], full[:, span.stop :])
    sources = {"none": [], "observed": ["north"], "all": ["east", "north"]}[missing]
    assert span.stop - span.start == 2 + len(sources)
    np.testing.assert_array_equal(out[:, span.start + 2 : span.stop], frame[sources].isna().astype(float))
    assert out.shape == full.shape == (len(frame), len(wrapped.get_feature_names_out()))
    if target_aware:
        assert not np.allclose(out[:, span.start : span.start + 2], full[:, span.start : span.start + 2])
    else:
        np.testing.assert_array_equal(out, full)
    assert len(fold_models) == 5
    for fold, (train_idx, _) in zip(fold_models, splitter.split(frame, target), strict=True):
        fold_block = fold.column_transformer_.named_transformers_["group_location"]
        fold_representation = fold_block if missing == "none" else dict(fold_block.transformer_list)["representation"]
        assert (fold_block is block) is not target_aware
        if target_aware:
            assert fold_representation[-1] is not representation[-1]
            for (_, helper), (_, fold_helper) in zip(
                representation.steps[:-1], fold_representation.steps[:-1], strict=True
            ):
                assert fold_helper is helper
            np.testing.assert_array_equal(
                fold_representation[-1].fit_values_,
                representation[:-1].transform(frame.iloc[train_idx][["east", "north"]]),
            )
            np.testing.assert_array_equal(fold_representation[-1].fit_target_, target[train_idx])
        if missing != "none":
            assert dict(fold_block.transformer_list)["missing"] is dict(block.transformer_list)["missing"]
        for name in ("num_spare", "cat_city"):
            assert (
                fold.column_transformer_.named_transformers_[name]
                is fitted.column_transformer_.named_transformers_[name]
            )
        assert fold.output_dims_ == fitted.output_dims_
        np.testing.assert_array_equal(fold.get_feature_names_out(), fitted.get_feature_names_out())
    records = fitted.get_feature_lineage()[span]
    assert all(record.uses_target is target_aware for record in records[:2])
    assert all(not record.uses_target for record in records[2:])
    pd.testing.assert_frame_equal(frame, original_frame)


@pytest.mark.parametrize("method", ["tensorspline", "tprs"])
@pytest.mark.parametrize("missing", ["none", "observed", "all"])
def test_builtin_groups_keep_all_data_state_while_supervised_single_is_refit(grouped_cross_fit_data, method, missing):
    from pretab import Preprocessor

    frame, target = grouped_cross_fit_data
    representation = (
        {"method": method, "output_dim": 3, "degree": 2}
        if method == "tensorspline"
        else {"method": method, "n_components": 3, "landmark_strategy": "subsample", "random_state": 0}
    )
    wrapped = CrossFittedTransformer(
        Preprocessor(
            numerical_method="ple",
            output_dim=4,
            categorical_method="one-hot",
            add_missing_indicator=missing == "observed",
            missing_policy="separate_state" if missing == "all" else None,
            feature_groups={"location": {"columns": ["east", "north"], "representation": representation}},
        ),
        n_folds=5,
        random_state=0,
    )
    out = wrapped.fit_transform(frame, target)
    fitted = cast(Preprocessor, wrapped.estimator_)
    full = wrapped.transform(frame)
    span = fitted.column_transformer_.output_indices_["group_location"]
    np.testing.assert_array_equal(out[:, span], full[:, span])
    assert not np.allclose(out[:, :4], full[:, :4])
    fingerprint = fitted.fingerprint_
    for train_idx, _ in KFold(n_splits=5, shuffle=True, random_state=0).split(frame):
        fold = fitted._cross_fit_fold(frame.iloc[train_idx], target[train_idx])
        assert fold.column_transformer_.named_transformers_["group_location"] is _fitted_block(
            wrapped, "group_location"
        )
        assert fold.column_transformer_.named_transformers_["num_spare"] is not _fitted_block(wrapped, "num_spare")
    assert fitted.fingerprint_ == fingerprint


def test_grouped_cross_fitting_retains_dictionary_error_and_matrix_remedy(grouped_cross_fit_data):
    from pretab import Preprocessor

    frame, target = grouped_cross_fit_data
    wrapped = CrossFittedTransformer(
        Preprocessor(
            numerical_method="minmax",
            output_structure="blocks",
            output_format="dense",
            feature_groups={
                "location": {
                    "columns": ["east", "north"],
                    "representation": {"method": "tensorspline", "output_dim": 3, "degree": 2},
                }
            },
        ),
        n_folds=5,
        random_state=0,
    )
    with pytest.raises(IncompatibleParamsError, match=r"dict of blocks.*output_structure='matrix'"):
        wrapped.fit_transform(frame, target)
    assert isinstance(wrapped.transform(frame), dict)


@pytest.mark.parametrize("missing", ["none", "observed", "all"])
def test_grouped_refit_preserves_prefix_and_refits_representation_suffix(grouped_cross_fit_data, monkeypatch, missing):
    from copy import copy

    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from pretab import Preprocessor
    from pretab.compose.inspection import _separate_state_branches, refit_block_representation
    from pretab.compose.registry import TRANSFORMER_REGISTRY

    monkeypatch.setitem(
        TRANSFORMER_REGISTRY,
        "tprs",
        replace(
            TRANSFORMER_REGISTRY["tprs"],
            transformer_cls=_JointTargetProjection,
            target_usage="optional",
            allowed_args=(),
            placement_strategies=frozenset(),
        ),
    )
    frame, target = grouped_cross_fit_data
    sources = ["east", "north"]
    fitted = Preprocessor(
        numerical_method="minmax",
        add_missing_indicator=missing == "observed",
        missing_policy="separate_state" if missing == "all" else None,
        feature_groups={"location": {"columns": sources, "representation": "tprs"}},
    ).fit(frame, target)
    original_block = fitted.column_transformer_.named_transformers_["group_location"]
    representation = original_block if missing == "none" else dict(original_block.transformer_list)["representation"]
    pipeline = copy(representation)
    pipeline.steps = [
        *representation.steps,
        ("suffix", StandardScaler().fit(representation.transform(frame[sources]))),
    ]
    block = pipeline
    if missing != "none":
        block = copy(original_block)
        block.transformer_list = [
            (name, pipeline if name == "representation" else branch) for name, branch in original_block.transformer_list
        ]
    original_values = block.transform(frame[sources]).copy()
    train, held_out = frame.iloc[20:][sources], frame.iloc[:20][sources]
    refitted = refit_block_representation(block, train, target[20:])
    branches = _separate_state_branches(refitted)
    if missing == "none":
        assert branches is None
        refitted_pipeline = refitted
    else:
        assert branches is not None
        refitted_pipeline, indicator = branches
        assert indicator is dict(block.transformer_list)["missing"]
    assert isinstance(refitted_pipeline, Pipeline)
    expected = clone(pipeline[-2:])
    assert isinstance(expected, Pipeline)
    expected.fit(pipeline[:-2].transform(train), target[20:])
    np.testing.assert_allclose(
        refitted_pipeline.transform(held_out), expected.transform(pipeline[:-2].transform(held_out))
    )
    for (_, helper), (_, fold_helper) in zip(pipeline.steps[:-2], refitted_pipeline.steps[:-2], strict=True):
        assert fold_helper is helper
    assert refitted_pipeline[-2] is not pipeline[-2]
    assert refitted_pipeline[-1] is not pipeline[-1]
    np.testing.assert_array_equal(block.transform(frame[sources]), original_values)
    np.testing.assert_array_equal(representation[-1].fit_target_, target)


class _ChangingJointWidth(_JointTargetProjection):
    def fit(self, X, y=None):
        super().fit(X, y)
        self.width_ = 2 if len(X) == 100 else (3 if np.asarray(X).mean() > 0 else 1)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.repeat((values @ self.weights_ + self.target_mean_)[:, None], self.width_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"joint_{index}" for index in range(self.width_)], dtype=object)


@pytest.mark.parametrize("first_group", ["positive", "negative"])
def test_grouped_cross_fitting_rejects_opposite_width_changes_with_constant_total(monkeypatch, first_group):
    import pandas as pd

    from pretab import Preprocessor
    from pretab.compose.inspection import refit_block_representation
    from pretab.compose.registry import TRANSFORMER_REGISTRY

    monkeypatch.setitem(
        TRANSFORMER_REGISTRY,
        "tprs",
        replace(
            TRANSFORMER_REGISTRY["tprs"],
            transformer_cls=_ChangingJointWidth,
            target_usage="required",
            allowed_args=(),
            placement_strategies=frozenset(),
        ),
    )
    values = np.linspace(1.0, 10.0, 100)
    frame = pd.DataFrame(
        {"positive": values, "positive_other": values + 1, "negative": -values, "negative_other": -values - 1}
    )
    target = np.random.default_rng(42).normal(size=len(frame))
    other_group = "negative" if first_group == "positive" else "positive"
    groups = {
        name: {"columns": [name, f"{name}_other"], "representation": "tprs"} for name in (first_group, other_group)
    }
    preprocessor = Preprocessor(scaling="none", feature_groups=groups).fit(frame, target)
    assert preprocessor.output_dims_ == {f"group_{first_group}": 2, f"group_{other_group}": 2}
    fold_widths = []
    for name in groups:
        columns = [name, f"{name}_other"]
        block = preprocessor.column_transformer_.named_transformers_[f"group_{name}"]
        refitted = refit_block_representation(block, frame.iloc[20:][columns], target[20:])
        fold_widths.append(len(refitted.get_feature_names_out(columns)))
    assert sorted(fold_widths) == [1, 3]
    assert sum(fold_widths) == preprocessor.total_output_dim_ == 4
    with pytest.raises(IncompatibleParamsError, match="fixed output width") as failure:
        CrossFittedTransformer(preprocessor, n_folds=5, shuffle=False).fit_transform(frame, target)
    assert f"block 'group_{first_group}'" in str(failure.value)
    assert f"expected 2, got {fold_widths[0]}" in str(failure.value)


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
        wrapped = Preprocessor(
            numerical_method=method, output_dim=4, output_format="dense", random_state=0, **missing_config
        )
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
    """The all-data imputer and scaler feed only the value to the fold's PLE;
    the raw missing indicator remains alongside the representation."""
    import pandas as pd

    from pretab import Preprocessor

    X, y = array_with_one_missing_value
    wrapped = Preprocessor(
        numerical_method="ple", output_dim=4, add_missing_indicator=True, output_format="dense", random_state=0
    )
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=5, random_state=0)
    out = cross_fitted.fit_transform(X, y)

    block = _fitted_block(cross_fitted, "num_feature_0")
    representation = dict(block.transformer_list)["representation"]
    assert [name for name, _ in representation.steps] == ["imputer", "minmax", "ple"]
    columns = _split_columns(cross_fitted, "num_feature_0")
    assert len(columns) == 5
    assert out.shape == (len(X), 9)
    expected = _refit_representation_per_fold(representation, pd.DataFrame({"feature_0": X[:, 0]}), y)
    np.testing.assert_array_equal(out[:, columns], np.column_stack((expected, np.isnan(X[:, 0]).astype(float))))


@pytest.mark.parametrize("method", ["bspline", "mspline", "ispline", "cubicspline", "naturalspline"])
def test_wrapped_preprocessor_keeps_the_indicator_next_to_a_target_aware_spline(array_with_one_missing_value, method):
    from pretab import Preprocessor

    X, y = array_with_one_missing_value
    wrapped = Preprocessor(numerical_method=method, add_missing_indicator=True, output_format="dense", random_state=0)
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=5, random_state=0)
    out = cross_fitted.fit_transform(X, y)
    assert out.shape == cross_fitted.transform(X).shape
    indicator = [i for i, name in enumerate(cross_fitted.get_feature_names_out()) if "missingindicator" in name]
    assert len(indicator) == 1
    np.testing.assert_array_equal(out[:, indicator[0]], np.isnan(X[:, 0]).astype(float))


def test_wrapped_preprocessor_keeps_the_separate_state_missing_branch(array_with_one_missing_value):
    """The representation uses the all-data imputer and scaler, while its raw missing state is shared."""
    import pandas as pd

    from pretab import Preprocessor

    X, y = array_with_one_missing_value
    wrapped = Preprocessor(
        numerical_method="ple", output_dim=4, missing_policy="separate_state", output_format="dense", random_state=0
    )
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=5, random_state=0)
    out = cross_fitted.fit_transform(X, y)

    representation = dict(_fitted_block(cross_fitted, "num_feature_0").transformer_list)["representation"]
    expected = _refit_representation_per_fold(representation, pd.DataFrame({"feature_0": X[:, 0]}), y)
    columns = _split_columns(cross_fitted, "num_feature_0")
    assert len(columns) == 5
    np.testing.assert_array_equal(out[:, columns], np.column_stack((expected, np.isnan(X[:, 0]).astype(float))))


@pytest.mark.parametrize("typed", [False, True], ids=["mapping", "value"])
@pytest.mark.parametrize("missing_policy", [None, "impute_with_indicator", "separate_state"])
def test_structured_cross_fitting_refits_only_target_representation(
    array_with_one_missing_value, monkeypatch, typed, missing_policy
):
    import pandas as pd
    from sklearn.base import clone
    from sklearn.pipeline import FeatureUnion, Pipeline

    from pretab import Preprocessor, RepresentationConfig

    values, target = array_with_one_missing_value
    frame = pd.DataFrame(values, columns=pd.Index(["supervised", "fixed"]), index=np.arange(len(values)) * 3 + 5)
    frame["city"] = np.resize(["a", "b", "c"], len(frame))
    frame.loc[frame.index[7], "city"] = "rare"
    requests = {
        "supervised": {
            "method": "rbf",
            "output_dim": 3,
            "target_aware": True,
            "placement_strategy": "cart",
            "params": {"gamma": 0.25},
        },
        "fixed": {
            "method": "rbf",
            "output_dim": 5,
            "target_aware": False,
            "placement_strategy": "uniform",
            "params": {"gamma": 0.75},
        },
        "city": {"method": "one-hot", "params": {"drop": "first", "sparse_output": False}},
    }
    wrapped = Preprocessor(
        feature_preprocessing={
            feature: RepresentationConfig.from_dict(settings) if typed else settings
            for feature, settings in requests.items()
        },
        missing_policy=missing_policy,
        random_state=0,
    )
    reference = clone(wrapped)
    assert isinstance(reference, Preprocessor)
    reference.fit(frame, target)
    folds = []
    original = Preprocessor._cross_fit_fold

    def record_fold(fitted, train_frame, train_target):
        fold = original(fitted, train_frame, train_target)
        folds.append(fold)
        return fold

    monkeypatch.setattr(Preprocessor, "_cross_fit_fold", record_fold)
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=5, random_state=0)
    out = cross_fitted.fit_transform(frame, target)
    fitted = cross_fitted.estimator_
    assert isinstance(fitted, Preprocessor)
    full = np.asarray(cross_fitted.transform(frame))
    np.testing.assert_array_equal(full, reference.transform(frame))
    assert len(folds) == 5
    indicator = int(missing_policy is not None)
    separate = int(missing_policy == "separate_state")
    assert fitted.output_dims_ == {"supervised": 3 + indicator, "fixed": 5 + separate, "city": 3 + separate}
    assert out.shape == full.shape == (len(frame), 11 + indicator + 2 * separate)
    all_blocks = fitted.column_transformer_.named_transformers_
    full_block = all_blocks["num_supervised"]
    if indicator:
        assert isinstance(full_block, FeatureUnion)
        representation = dict(full_block.transformer_list)["representation"]
    else:
        representation = full_block
    assert isinstance(representation, Pipeline)
    for fold in folds:
        fold_blocks = fold.column_transformer_.named_transformers_
        fold_block = fold_blocks["num_supervised"]
        if indicator:
            assert isinstance(fold_block, FeatureUnion)
            assert dict(fold_block.transformer_list)["missing"] is dict(full_block.transformer_list)["missing"]
            fold_representation = dict(fold_block.transformer_list)["representation"]
        else:
            fold_representation = fold_block
        assert isinstance(fold_representation, Pipeline)
        assert fold_representation.named_steps["imputer"] is representation.named_steps["imputer"]
        assert fold_representation.named_steps["minmax"] is representation.named_steps["minmax"]
        assert fold_representation.named_steps["rbf"] is not representation.named_steps["rbf"]
        assert fold_representation.named_steps["rbf"].output_dim == 3
        assert fold_representation.named_steps["rbf"].gamma == 0.25
        assert fold_blocks["num_fixed"] is all_blocks["num_fixed"]
        assert fold_blocks["cat_city"] is all_blocks["cat_city"]
        assert fold.output_dims_ == fitted.output_dims_
        np.testing.assert_array_equal(fold.get_feature_names_out(), fitted.get_feature_names_out())
    supervised = [
        record.output_index
        for record in fitted.get_feature_lineage()
        if record.source_features == ("supervised",) and record.uses_target
    ]
    assert len(supervised) == 3
    expected = _refit_representation_per_fold(representation, frame[["supervised"]], target)
    np.testing.assert_array_equal(out[:, supervised], expected)
    assert not np.allclose(out[:, supervised], full[:, supervised])
    unchanged = [index for index in range(out.shape[1]) if index not in supervised]
    np.testing.assert_array_equal(out[:, unchanged], full[:, unchanged])


@pytest.mark.parametrize("typed", [False, True], ids=["mapping", "value"])
@pytest.mark.parametrize("container", ["dense", "sparse", "auto", "pandas", "polars"])
def test_structured_cross_fitting_preserves_output_container(array_with_rare_category, typed, container):
    import pandas as pd
    from scipy import sparse

    from pretab import Preprocessor, RepresentationConfig

    polars = pytest.importorskip("polars") if container == "polars" else None
    values, target = array_with_rare_category
    frame = pd.DataFrame(
        values,
        columns=pd.Index(["signal", "count", "city"]),
        index=pd.Index(np.arange(len(values))[::-1] * 3, name="row"),
    )
    requests = {
        "signal": {"method": "rbf", "output_dim": 3, "params": {"gamma": 0.25}},
        "city": {"method": "one-hot", "params": {"drop": "first"}},
    }
    parameters = {
        "feature_preprocessing": {
            feature: RepresentationConfig.from_dict(settings) if typed else settings
            for feature, settings in requests.items()
        },
        "random_state": 0,
        "dtype": np.float32,
    }
    wrapped = Preprocessor(output_format=container if container in {"sparse", "auto"} else "dense", **parameters)
    if container in {"pandas", "polars"}:
        wrapped.set_output(transform=container)
    cross_fitted, out = _cross_fit(wrapped, frame, target)
    full = cross_fitted.transform(frame)
    assert type(out) is type(full)
    if container == "sparse":
        assert sparse.isspmatrix_csr(out)
    if sparse.issparse(out):
        assert sparse.issparse(full)
        assert out.format == full.format == "csr"
        out_values = out.toarray()
    elif isinstance(out, pd.DataFrame):
        assert isinstance(full, pd.DataFrame)
        pd.testing.assert_index_equal(out.index, frame.index)
        pd.testing.assert_index_equal(out.columns, full.columns)
        assert out.columns.tolist() == cross_fitted.get_feature_names_out().tolist()
        assert out.dtypes.equals(full.dtypes)
        out_values = out.to_numpy()
    elif container == "polars":
        assert polars is not None
        assert isinstance(out, polars.DataFrame)
        assert out.columns == full.columns == cross_fitted.get_feature_names_out().tolist()
        assert out.schema == full.schema
        out_values = out.to_numpy()
    else:
        assert isinstance(out, np.ndarray)
        out_values = out
    _, dense = _cross_fit(Preprocessor(**parameters), frame, target)
    assert out_values.shape == (len(frame), 7)
    assert out_values.dtype == np.float32
    np.testing.assert_array_equal(out_values, dense)


@pytest.mark.parametrize("typed", [False, True], ids=["mapping", "value"])
def test_structured_cross_fitting_checks_each_block_width(array_with_one_missing_value, monkeypatch, typed):
    from pretab import Preprocessor, RepresentationConfig
    from pretab.expansion.functional.rbf import RBFExpansionTransformer

    values, target = array_with_one_missing_value
    requests = {
        "feature_0": {"method": "rbf", "output_dim": 3, "params": {"gamma": 0.25}},
        "feature_1": {
            "method": "rbf",
            "output_dim": 5,
            "target_aware": False,
            "placement_strategy": "uniform",
        },
    }
    original = RBFExpansionTransformer.fit

    def change_fold_width(estimator, fold_values, fold_target=None):
        if len(fold_values) < len(values) and estimator.gamma == 0.25:
            estimator.set_params(output_dim=2)
        return original(estimator, fold_values, fold_target)

    monkeypatch.setattr(RBFExpansionTransformer, "fit", change_fold_width)
    wrapped = Preprocessor(
        feature_preprocessing={
            feature: RepresentationConfig.from_dict(settings) if typed else settings
            for feature, settings in requests.items()
        },
        random_state=0,
    )
    with pytest.raises(
        IncompatibleParamsError,
        match="fixed output width for block 'num_feature_0'; expected 3, got 2",
    ) as error:
        _cross_fit(wrapped, values, target)
    assert "same columns on every fold" in str(error.value)


@pytest.mark.parametrize("typed", [False, True], ids=["mapping", "value"])
def test_structured_cross_fitting_still_rejects_dictionary_output(integer_array, typed):
    from pretab import Preprocessor, RepresentationConfig

    values, target = integer_array
    settings = {"method": "rbf", "output_dim": 3}
    wrapped = Preprocessor(
        feature_preprocessing={"feature_0": RepresentationConfig.from_dict(settings) if typed else settings},
        output_structure="blocks",
        random_state=0,
    )
    with pytest.raises(IncompatibleParamsError, match="dict of blocks"):
        _cross_fit(wrapped, values, target)


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


@pytest.mark.parametrize("wrapped_kind", ["ple", "preprocessor", "column_transformer", "pipeline"])
def test_dataframe_representation_spec_defaults_to_fitted_names(mixed_frame, wrapped_kind):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    from pretab import Preprocessor

    X, y = mixed_frame
    if wrapped_kind == "ple":
        X = X[["num"]]
        wrapped = PLETransformer(output_dim=4)
    elif wrapped_kind == "preprocessor":
        wrapped = Preprocessor(output_dim=4, random_state=0)
    else:
        wrapped = ColumnTransformer([("ple", PLETransformer(output_dim=4), ["num"])])
        if wrapped_kind == "pipeline":
            wrapped = Pipeline([("features", wrapped)])
    cross_fitted = CrossFittedTransformer(wrapped, n_folds=3, random_state=0).fit(X, y)

    spec = cross_fitted.get_representation_spec()
    assert spec.input_features == tuple(X.columns)
    assert spec.output_features == tuple(cross_fitted.get_feature_names_out())
    assert spec.cross_fitted is True
    assert spec.n_folds == 3
    assert spec.uses_target is True
    assert spec == type(spec).from_dict(spec.to_dict())
    assert cross_fitted.get_representation_spec(list(X.columns)) == spec


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
@pytest.mark.parametrize("add_missing_indicator", [False, True])
def test_cross_fitting_lightgbm_refits_only_target_blocks_and_keeps_the_full_fit(method, task, add_missing_indicator):
    pytest.importorskip("lightgbm")
    import pandas as pd
    from sklearn.base import clone

    from pretab import Preprocessor

    rng = np.random.default_rng(0)
    X = pd.DataFrame({1: rng.normal(size=160), 2: rng.choice([True, False], size=160)})
    y = 2 * X[1].to_numpy() + X[2].to_numpy() + rng.normal(scale=0.1, size=len(X))
    if task == "classification":
        y = np.where(y >= np.median(y), "high", "low")
    if add_missing_indicator:
        X.loc[7, 1] = np.nan
    preprocessor = Preprocessor(
        numerical_method=method,
        categorical_method="one-hot",
        placement_strategy="lightgbm",
        output_dim=6,
        output_format="dense",
        task=task,
        add_missing_indicator=add_missing_indicator,
        random_state=0,
    )
    cross_fitted = CrossFittedTransformer(preprocessor, n_folds=3, task=task, random_state=0)
    out = cross_fitted.fit_transform(X, y)
    full = cross_fitted.transform(X)
    width = 8 + int(add_missing_indicator)
    assert out.shape == full.shape == (len(X), width)
    assert not np.allclose(out[:, :6], full[:, :6])
    np.testing.assert_array_equal(out[:, 6:], full[:, 6:])
    reference = clone(preprocessor)
    assert isinstance(reference, Preprocessor)
    reference_out = reference.fit(X, y).transform(X)
    assert isinstance(reference_out, np.ndarray)
    np.testing.assert_allclose(full, reference_out)
    assert len(cross_fitted.get_feature_names_out()) == width
    fitted = cross_fitted.estimator_
    assert isinstance(fitted, Preprocessor)
    assert [item.uses_target for item in fitted.get_feature_lineage()] == [True] * 6 + [False] * (width - 6)


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
