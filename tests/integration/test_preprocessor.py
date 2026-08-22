import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from pretab.exceptions import IncompatibleParamsError, PretabDataError
from pretab.preprocessor import Preprocessor  # Adjust the import as needed


@pytest.fixture
def sample_data():
    df = pd.DataFrame(
        {
            "num1": np.linspace(0, 1, 100),
            "num2": np.random.randn(100),
            "cat1": np.random.choice(["A", "B", "C"], size=100),
            "cat2": np.random.randint(0, 5, size=100),
        }
    )
    y = df["num1"] * 2 + df["num2"] + np.random.randn(100) * 0.1
    return df, y


def test_fit_transform_returns_dict(sample_data):
    X, y = sample_data
    pre = Preprocessor()
    out = pre.fit_transform(X, y)
    assert isinstance(out, dict)
    assert all(isinstance(k, str) for k in out)
    assert all(isinstance(v, np.ndarray) for v in out.values())
    assert sum(v.shape[0] for v in out.values()) == 4 * len(X)


def test_transform_array_output(sample_data):
    X, y = sample_data
    pre = Preprocessor()
    pre.fit(X, y)
    out = pre.transform(X, return_array=True)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == len(X)
    assert out.ndim == 2


def test_transform_raises_before_fit(sample_data):
    X, _ = sample_data
    pre = Preprocessor()
    with pytest.raises(NotFittedError):
        _ = pre.transform(X)


def test_embedding_integration(sample_data):
    X, y = sample_data
    embed = np.random.rand(len(X), 10)
    pre = Preprocessor()
    out = pre.fit_transform(X, y, embeddings=embed)
    assert "embedding_1" in out
    assert out["embedding_1"].shape == (len(X), 10)


def test_multiple_embeddings(sample_data):
    X, y = sample_data
    embeds = [np.random.rand(len(X), 3), np.random.rand(len(X), 7)]
    pre = Preprocessor()
    out = pre.fit_transform(X, y, embeddings=embeds)
    assert "embedding_1" in out and "embedding_2" in out
    assert out["embedding_1"].shape[1] == 3
    assert out["embedding_2"].shape[1] == 7


def test_embeddings_are_required_after_embedding_aware_fit(sample_data):
    X, y = sample_data
    embeddings = np.random.rand(len(X), 4)
    pre = Preprocessor().fit(X, y, embeddings=embeddings)

    with pytest.raises(PretabDataError, match="required during transform"):
        pre.transform(X)


def test_embeddings_are_rejected_with_array_output(sample_data):
    X, y = sample_data
    embeddings = np.random.rand(len(X), 4)
    pre = Preprocessor().fit(X, y, embeddings=embeddings)

    with pytest.raises(IncompatibleParamsError, match="only with dictionary output"):
        pre.transform(X, embeddings=embeddings, return_array=True)


def test_embeddings_are_rejected_with_dataframe_output(sample_data):
    X, y = sample_data
    embeddings = np.random.rand(len(X), 4)
    pre = Preprocessor().fit(X, y, embeddings=embeddings).set_output(transform="pandas")

    with pytest.raises(IncompatibleParamsError, match="only with dictionary output"):
        pre.transform(X, embeddings=embeddings)


def test_feature_info_returns_three_dicts(sample_data):
    X, y = sample_data
    pre = Preprocessor()
    pre.fit(X, y)
    info = pre.get_feature_info(verbose=False)
    assert isinstance(info, tuple)
    assert len(info) == 3
    assert all(isinstance(d, dict) for d in info)


def test_dict_output_shapes_add_up(sample_data):
    X, y = sample_data
    pre = Preprocessor()
    out = pre.fit_transform(X, y)
    assert isinstance(out, dict)
    shapes = [v.shape for v in out.values()]
    assert all(s[0] == len(X) for s in shapes)


def test_dict_keys_reflect_column_names(sample_data):
    X, y = sample_data
    pre = Preprocessor()
    out = pre.fit_transform(X, y)
    assert isinstance(out, dict)
    expected_prefixes = ["num_", "cat_"]
    for k in out:
        if "embedding" not in k:
            assert any(k.startswith(p) for p in expected_prefixes)


# --- sklearn estimator-contract compliance (Phase 11) ---

EXPECTED_PARAMS = {
    "numerical_method",
    "categorical_method",
    "feature_preprocessing",
    "output_dim",
    "adaptive",
    "min_output_dim",
    "max_output_dim",
    "task",
    "target_aware",
    "placement_strategy",
    "degree",
    "scaling",
    "cat_cutoff",
    "treat_all_integers_as_numerical",
    "random_state",
    "numerical_imputation",
    "categorical_imputation",
    "add_missing_indicator",
    "missing_policy",
    "policy",
    "max_output_features",
    "max_features_per_input",
    "max_dense_memory",
    "overflow_policy",
    "output_format",
    "dtype",
    "verbose",
    "preset",
}


def test_get_params_is_complete():
    assert set(Preprocessor().get_params()) == EXPECTED_PARAMS


def test_init_stores_params_verbatim():
    # __init__ must not mutate constructor args (sklearn clone invariant).
    pre = Preprocessor(numerical_method="PLE", feature_preprocessing=None, verbose=2)
    assert pre.numerical_method == "PLE"
    assert pre.feature_preprocessing is None
    assert pre.verbose == 2


def test_clone_preserves_every_param():
    pre = Preprocessor(
        numerical_method="cubicspline",
        output_dim=8,
        adaptive=True,
        feature_preprocessing={"a": "pspline"},
        verbose=1,
    )
    cloned = clone(pre)
    assert isinstance(cloned, Preprocessor)
    assert cloned.get_params() == pre.get_params()


def test_set_params_roundtrip():
    pre = Preprocessor()
    pre.set_params(output_dim=11, numerical_method="rbf")
    assert pre.get_params()["output_dim"] == 11
    assert pre.get_params()["numerical_method"] == "rbf"


def test_set_params_rejects_unknown_key():
    with pytest.raises(ValueError):
        Preprocessor().set_params(not_a_param=1)


def test_check_is_fitted_before_and_after(sample_data):
    X, y = sample_data
    pre = Preprocessor()
    with pytest.raises(NotFittedError):
        check_is_fitted(pre)
    pre.fit(X, y)
    check_is_fitted(pre)  # no raise
    assert pre.n_features_in_ == X.shape[1]


def test_get_feature_names_out_matches_array_width(sample_data):
    X, y = sample_data
    pre = Preprocessor()
    pre.fit(X, y)
    names = pre.get_feature_names_out()
    arr = pre.transform(X, return_array=True)
    assert isinstance(arr, np.ndarray)
    assert len(names) == arr.shape[1]


def test_get_feature_names_out_before_fit_raises():
    with pytest.raises(NotFittedError):
        Preprocessor().get_feature_names_out()


def test_get_feature_names_out_does_not_duplicate_feature_name(sample_data):
    """Regression guard: output names must not repeat as num_<feat>__<feat>_... ."""
    X, y = sample_data
    pre = Preprocessor()
    pre.fit(X, y)
    names = list(pre.get_feature_names_out())
    assert names
    assert all("__" not in name for name in names)
    assert "num_num1_ple0" in names
    lineage_names = [record.output_feature for record in pre.get_feature_lineage()]
    assert lineage_names == names


def test_lowercase_and_none_method_resolution(sample_data):
    X, y = sample_data
    # Mixed-case / None methods are resolved at fit time, not stored on the instance.
    pre = Preprocessor(numerical_method="PLE", categorical_method=None)  # type: ignore[arg-type]
    out = pre.fit_transform(X, y)
    assert isinstance(out, dict)
    assert pre.numerical_method == "PLE"  # unchanged on the instance
    assert pre.categorical_method is None


def test_total_output_dim_matches_array_width(sample_data):
    X, y = sample_data
    pre = Preprocessor().fit(X, y)
    arr = pre.transform(X, return_array=True)
    assert isinstance(arr, np.ndarray)
    assert pre.total_output_dim_ == arr.shape[1]


def test_output_dims_keys_are_input_features(sample_data):
    X, y = sample_data
    pre = Preprocessor().fit(X, y)
    assert set(pre.output_dims_) == set(X.columns)


def test_output_dims_sum_to_total_output_dim(sample_data):
    X, y = sample_data
    pre = Preprocessor().fit(X, y)
    assert sum(pre.output_dims_.values()) == pre.total_output_dim_


def test_output_dims_reflects_per_feature_widths():
    X = pd.DataFrame(
        {
            "a": np.linspace(0, 1, 60),
            "b": np.linspace(-1, 1, 60),
        }
    )
    y = pd.Series(np.random.randn(60))
    # 'a' uses the rbf feature map (output_dim columns); 'b' is overridden to
    # plain min-max scaling (a single column) via feature_preprocessing.
    pre = Preprocessor(
        numerical_method="rbf",
        output_dim=5,
        feature_preprocessing={"b": "minmax"},
    ).fit(X, y)
    dims = pre.output_dims_
    assert dims["a"] == 5
    assert dims["b"] == 1
    assert sum(dims.values()) == pre.total_output_dim_


def test_output_dims_nonuniform_for_one_hot_categorical():
    X = pd.DataFrame(
        {
            "num": np.linspace(0, 1, 30),
            "cat": ["A", "B", "C"] * 10,
        }
    )
    y = pd.Series(np.random.randn(30))
    pre = Preprocessor(numerical_method="minmax", categorical_method="one-hot").fit(X, y)
    dims = pre.output_dims_
    assert dims["num"] == 1
    assert dims["cat"] == 3  # one-hot of three categories
    assert sum(dims.values()) == pre.total_output_dim_


def test_output_dims_and_total_before_fit_raise():
    with pytest.raises(NotFittedError):
        _ = Preprocessor().output_dims_
    with pytest.raises(NotFittedError):
        _ = Preprocessor().total_output_dim_
