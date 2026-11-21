"""Polars DataFrames as Preprocessor input.

Regression guard: a polars frame used to be treated as a pandas frame and failed
with ``AttributeError: 'list' object has no attribute 'duplicated'``, so a
scikit-learn ``Pipeline`` with ``set_output(transform="polars")`` crashed when it
handed its polars output to the Preprocessor.
"""

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import pytest

from pretab import FeatureGroup, Preprocessor, RepresentationConfig
from pretab.exceptions import DataWarning, PretabDataError

pl = pytest.importorskip("polars")


@runtime_checkable
class _NamedFrame(Protocol):
    columns: pd.Index | list[str]


@pytest.fixture
def frames():
    """The same mixed-type data as a pandas and as a polars frame (Float64, Int64,
    String, Categorical and Boolean columns)."""
    rng = np.random.default_rng(0)
    n = 120
    pandas_frame = pd.DataFrame(
        {
            "income": rng.normal(5e4, 1e4, n),
            "visits": rng.integers(0, 1000, n),
            "rooms": rng.integers(1, 4, n),
            "plan": rng.choice(["basic", "premium"], n),
            "city": rng.choice(["a", "b", "c"], n),
            "flag": rng.choice([True, False], n),
        }
    )
    polars_frame = pl.DataFrame(
        {
            "income": pandas_frame["income"].to_numpy(),
            "visits": pandas_frame["visits"].to_numpy(),
            "rooms": pandas_frame["rooms"].to_numpy(),
            "plan": pandas_frame["plan"].tolist(),
            "city": pandas_frame["city"].tolist(),
            "flag": pandas_frame["flag"].to_numpy(),
        }
    ).with_columns(pl.col("city").cast(pl.Categorical))
    return pandas_frame, polars_frame, rng.normal(size=n)


@pytest.mark.parametrize(
    "params",
    [
        {"numerical_method": "minmax"},
        {"numerical_method": "bspline", "random_state": 0},
        {"numerical_method": "ple", "categorical_method": "one-hot", "random_state": 0},
    ],
)
def test_polars_input_matches_pandas_input(frames, params):
    pandas_frame, polars_frame, y = frames
    from_pandas = Preprocessor(**params).fit(pandas_frame, y)
    from_polars = Preprocessor(**params).fit(polars_frame, y)

    assert from_polars.numerical_features_ == ["income", "visits"]
    assert from_polars.categorical_features_ == ["rooms", "plan", "city", "flag"]
    assert from_polars.categorical_features_ == from_pandas.categorical_features_
    assert list(from_polars.feature_names_in_) == polars_frame.columns
    np.testing.assert_array_equal(from_polars.get_feature_names_out(), from_pandas.get_feature_names_out())

    expected = np.asarray(from_pandas.transform(pandas_frame))
    np.testing.assert_allclose(np.asarray(from_polars.transform(polars_frame)), expected)
    np.testing.assert_allclose(np.asarray(Preprocessor(**params).fit_transform(polars_frame, y)), expected)
    # A preprocessor fitted on pandas transforms the polars frame the same way.
    np.testing.assert_allclose(np.asarray(from_pandas.transform(polars_frame)), expected)
    # Columns are matched by name, so a reordered polars frame transforms the same.
    reordered = polars_frame.select(polars_frame.columns[::-1])
    np.testing.assert_allclose(np.asarray(from_polars.transform(reordered)), expected)


@pytest.mark.parametrize("typed", [False, True], ids=["mapping", "value"])
@pytest.mark.parametrize("output_format", ["dense", "sparse"])
def test_mixed_structured_inputs_preserve_polars_column_types(frames, typed, output_format):
    from scipy import sparse

    pandas_frame, polars_frame, _ = frames
    original = polars_frame.clone()
    settings = {
        "flag": "int",
        "city": {"method": "one-hot", "params": {"drop": "first"}},
        "visits": "rbf",
        "plan": "one-hot",
        "income": {"method": "rbf", "output_dim": 2, "params": {"gamma": 0.25}},
    }
    options = {
        "feature_preprocessing": {
            feature: RepresentationConfig.from_dict(value) if typed and isinstance(value, dict) else value
            for feature, value in settings.items()
        },
        "numerical_method": "none",
        "output_dim": 3,
        "target_aware": False,
        "placement_strategy": "uniform",
        "scaling": "none",
        "random_state": 17,
        "output_format": output_format,
    }
    from_pandas = Preprocessor(**options).fit(pandas_frame)
    from_polars = Preprocessor(**options).fit(polars_frame)
    expected = from_pandas.transform(pandas_frame)
    assert not isinstance(expected, dict)
    expected = expected.toarray() if sparse.issparse(expected) else np.asarray(expected)
    assert from_polars.numerical_features_ == ["income", "visits"]
    assert from_polars.categorical_features_ == ["rooms", "plan", "city", "flag"]
    dimensions = {"income": 2, "visits": 3, "rooms": 1, "plan": 2, "city": 2, "flag": 1}
    assert from_polars.output_dims_ == dimensions
    assert list(from_polars.output_dims_) == list(dimensions)
    assert from_polars.get_feature_info(verbose=False) == from_pandas.get_feature_info(verbose=False)
    assert from_polars.get_feature_lineage() == from_pandas.get_feature_lineage()
    np.testing.assert_array_equal(from_polars.get_feature_names_out(), from_pandas.get_feature_names_out())
    for preprocessor, incoming in (
        (from_polars, polars_frame.select(polars_frame.columns[::-1])),
        (from_polars, pandas_frame.iloc[:, ::-1]),
        (from_pandas, polars_frame.select(polars_frame.columns[::-1])),
    ):
        result = preprocessor.transform(incoming)
        assert not isinstance(result, dict)
        assert sparse.issparse(result) == (output_format == "sparse")
        np.testing.assert_array_equal(result.toarray() if sparse.issparse(result) else np.asarray(result), expected)
        blocks = preprocessor.transform(incoming, return_array=False)
        assert isinstance(blocks, dict)
        assert list(blocks) == ["num_income", "num_visits", "cat_rooms", "cat_plan", "cat_city", "cat_flag"]
        assert [np.shape(block)[1] for block in blocks.values()] == list(dimensions.values())
    assert polars_frame.equals(original)


@pytest.mark.parametrize("method", ["tensorspline", "tprs"])
@pytest.mark.parametrize("output_format", ["dense", "sparse", "auto"])
@pytest.mark.parametrize("typed", [False, True], ids=["mapping", "value"])
def test_structured_and_grouped_polars_inputs_match_pandas(frames, method, output_format, typed):
    from scipy import sparse

    pandas_frame, polars_frame, _ = frames
    pandas_frame = pandas_frame.astype({"visits": float, "flag": object})
    missing_rows = {"visits": 7, "plan": 13, "city": 19, "flag": 23}
    nullable_columns = []
    for feature, row in missing_rows.items():
        values = polars_frame[feature].to_list()
        values[row] = None
        dtype = pl.Enum(["premium", "basic", "unused"]) if feature == "plan" else polars_frame[feature].dtype
        nullable_columns.append(pl.Series(feature, values, dtype=dtype))
        pandas_frame.loc[row, feature] = np.nan
    polars_frame = polars_frame.with_columns(nullable_columns)
    original = polars_frame.clone()
    representation = (
        {"method": method, "output_dim": 3, "degree": 2}
        if method == "tensorspline"
        else {"method": method, "n_components": 3, "landmark_strategy": "subsample", "random_state": 0}
    )
    group_sources = ["visits", "income"]
    groups = (
        [FeatureGroup("location", group_sources, RepresentationConfig.from_dict(representation))]
        if typed
        else {"location": {"columns": group_sources, "representation": representation}}
    )
    overrides = {"method": "one-hot", "params": {"drop": "first"}}
    options = {
        "numerical_method": "none",
        "categorical_method": "one-hot",
        "target_aware": False,
        "placement_strategy": "uniform",
        "missing_policy": "separate_state",
        "output_format": output_format,
        "dtype": np.float32,
        "feature_groups": groups,
        "feature_preprocessing": {
            feature: RepresentationConfig.from_dict(overrides) if typed else overrides
            for feature in ("flag", "city", "plan")
        },
    }
    from_pandas = Preprocessor(**options).fit(pandas_frame)
    from_polars = Preprocessor(**options).fit(polars_frame)
    reference = from_pandas.transform(pandas_frame)
    assert not isinstance(reference, dict)
    expected = reference.toarray() if sparse.issparse(reference) else np.asarray(reference)
    assert from_polars.numerical_features_ == ["income", "visits"]
    assert from_polars.categorical_features_ == ["rooms", "plan", "city", "flag"]
    assert list(from_polars.feature_names_in_) == polars_frame.columns
    assert from_polars.output_dims_ == from_pandas.output_dims_
    assert from_polars.get_feature_lineage() == from_pandas.get_feature_lineage()
    names = list(from_polars.get_feature_names_out())
    np.testing.assert_array_equal(names, from_pandas.get_feature_names_out())
    indicators = {}
    for feature, width in {"plan": 2, "city": 3, "flag": 2}.items():
        assert from_polars.output_dims_[feature] == width
        matches = [
            index
            for index, name in enumerate(names)
            if name.startswith(f"cat_{feature}__") and name.endswith("__missing")
        ]
        assert len(matches) == 1
        indicators[feature] = matches[0]
    for preprocessor, incoming in (
        (from_polars, polars_frame.select(polars_frame.columns[::-1])),
        (from_polars, pandas_frame.iloc[:, ::-1]),
        (from_pandas, polars_frame.select(polars_frame.columns[::-1])),
    ):
        actual = preprocessor.transform(incoming)
        assert not isinstance(actual, dict)
        assert sparse.issparse(actual) == (preprocessor.output_format_ == "sparse")
        values = actual.toarray() if sparse.issparse(actual) else np.asarray(actual)
        np.testing.assert_array_equal(values, expected)
        for feature, index in indicators.items():
            np.testing.assert_array_equal(values[:, index], pandas_frame[feature].isna())
        blocks = preprocessor.transform(incoming, return_array=False)
        assert isinstance(blocks, dict)
        assert list(blocks) == ["group_location", "cat_rooms", "cat_plan", "cat_city", "cat_flag"]
        group = blocks["group_location"]
        values = group.toarray() if sparse.issparse(group) else np.asarray(group)
        np.testing.assert_array_equal(values[:, -2:], pandas_frame[group_sources].isna().astype(np.float32))
    assert polars_frame.equals(original)
    assert {feature: polars_frame[feature].null_count() for feature in missing_rows} == dict.fromkeys(missing_rows, 1)
    with pytest.raises(PretabDataError) as failure:
        from_polars.transform(polars_frame.drop("visits"))
    assert "group 'group_location'" in str(failure.value)
    assert "missing sources ('visits',)" in str(failure.value)


@pytest.mark.parametrize("container", ["default", "pandas", "polars", "blocks"])
@pytest.mark.parametrize("output_format", ["dense", "sparse"])
def test_polars_input_output_matrix(frames, container, output_format):
    from scipy import sparse

    pandas_frame, polars_frame, y = frames
    reference = Preprocessor(numerical_method="minmax", categorical_method="one-hot", output_format="dense")
    expected = reference.fit_transform(pandas_frame, y)
    assert isinstance(expected, np.ndarray)
    names = list(reference.get_feature_names_out())

    blocks = container == "blocks"
    pre = Preprocessor(
        numerical_method="minmax",
        categorical_method="one-hot",
        output_format=output_format,
        output_structure="blocks" if blocks else "matrix",
    ).set_output(transform="default" if blocks else container)
    training = pre.fit_transform(polars_frame, y)
    np.testing.assert_array_equal(pre.get_feature_names_out(), names)
    reordered = polars_frame.select(polars_frame.columns[::-1])
    for result in (training, pre.transform(reordered)):
        if blocks:
            assert isinstance(result, dict)
            assert list(result) == ["num_income", "num_visits", "cat_rooms", "cat_plan", "cat_city", "cat_flag"]
            values = []
            for block, width in zip(result.values(), reference.output_dims_.values(), strict=True):
                if output_format == "sparse":
                    assert isinstance(block, sparse.csr_matrix)
                    block = block.toarray()
                else:
                    assert isinstance(block, np.ndarray)
                assert block.shape == (polars_frame.height, width)
                values.append(block)
            actual = np.concatenate(values, axis=1)
        elif container in ("pandas", "polars"):
            assert isinstance(result, pd.DataFrame if container == "pandas" else pl.DataFrame)
            assert isinstance(result, _NamedFrame)
            assert list(result.columns) == names
            actual = np.asarray(result)
        elif output_format == "sparse":
            assert isinstance(result, sparse.csr_matrix)
            actual = result.toarray()
        else:
            assert isinstance(result, np.ndarray)
            actual = result
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("output", ["pandas", "polars"])
def test_polars_input_with_global_dataframe_output(frames, output):
    from sklearn import config_context

    pandas_frame, polars_frame, y = frames
    expected = Preprocessor(numerical_method="minmax").fit_transform(pandas_frame, y)
    with config_context(transform_output=output):
        pre = Preprocessor(numerical_method="minmax").fit(polars_frame, y)
        result = pre.transform(polars_frame)
    assert isinstance(result, pd.DataFrame if output == "pandas" else pl.DataFrame)
    assert isinstance(result, _NamedFrame)
    assert list(result.columns) == list(pre.get_feature_names_out())
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


def test_polars_input_round_trips_through_spec(frames):
    _, polars_frame, y = frames
    pre = Preprocessor(numerical_method="minmax", categorical_method="one-hot").fit(polars_frame, y)
    restored = Preprocessor.from_spec(pre.to_spec())
    np.testing.assert_array_equal(restored.get_feature_names_out(), pre.get_feature_names_out())
    np.testing.assert_array_equal(restored.transform(polars_frame), pre.transform(polars_frame))


def test_polars_nulls_are_missing_values():
    frame = pl.DataFrame(
        {
            "x": [1.0, None, 3.0, 4.0] * 10,
            "city": ["a", None, "b", "a"] * 10,
            "flag": [True, None, False, True] * 10,
        }
    )
    original = frame.clone()
    pre = Preprocessor(
        numerical_method="minmax",
        categorical_method="one-hot",
        missing_policy="impute_with_indicator",
        output_format="dense",
    )
    pre.fit(frame)
    names = list(pre.get_feature_names_out())
    # The null is imputed and flagged, not encoded as a "None" category.
    assert any("missingindicator_city" in name for name in names)
    assert not any("None" in name for name in names)
    output = pre.transform(frame)
    for column in frame.columns:
        index = next(i for i, name in enumerate(names) if f"missingindicator_{column}" in name)
        np.testing.assert_array_equal(np.asarray(output)[:, index], [0, 1, 0, 0] * 10)
    restored = Preprocessor.from_spec(pre.to_spec())
    np.testing.assert_array_equal(restored.transform(frame), output)
    assert frame.equals(original)


@pytest.mark.parametrize("dtype", ["Float64", "Utf8", "Boolean"])
def test_polars_all_null_column_is_retained_like_pandas(dtype):
    frame = pl.DataFrame({"empty": pl.Series([None] * 8, dtype=getattr(pl, dtype))})
    pandas_dtype = float if dtype == "Float64" else object
    expected_frame = pd.DataFrame({"empty": pd.Series([None] * 8, dtype=pandas_dtype)})
    options = {"numerical_method": "none", "categorical_method": "one-hot", "add_missing_indicator": True}
    with pytest.warns(DataWarning, match="no observed"):
        from_polars = Preprocessor(**options).fit(frame)
    with pytest.warns(DataWarning, match="no observed"):
        from_pandas = Preprocessor(**options).fit(expected_frame)
    np.testing.assert_array_equal(from_polars.get_feature_names_out(), from_pandas.get_feature_names_out())
    np.testing.assert_array_equal(from_polars.transform(frame), from_pandas.transform(expected_frame))


def test_representation_search_on_polars_matches_pandas(frames):
    """Each fold's Preprocessor gets a frame with the polars column types too, not
    an object array that would turn every column categorical (and score a
    different model than the final refit)."""
    from sklearn.linear_model import Ridge

    from pretab import RepresentationSearchCV

    pandas_frame, polars_frame, y = frames
    y = np.asarray(pandas_frame["income"]) / 1e4 + y

    def search(X):
        return RepresentationSearchCV(Ridge(), ["minmax", "bspline"], cv=3, random_state=0).fit(X, y)

    from_pandas, from_polars = search(pandas_frame), search(polars_frame)

    assert from_polars.cv_results_ == pytest.approx(from_pandas.cv_results_)
    assert from_polars.best_method_ == from_pandas.best_method_
    np.testing.assert_allclose(from_polars.predict(polars_frame), from_pandas.predict(pandas_frame))


@pytest.fixture
def frames_with_one_null_integer():
    """An integer column with a single null, as polars and as the pandas frame it converts to."""
    rng = np.random.default_rng(0)
    kids = rng.integers(0, 5, 300).astype(float)
    kids[7] = np.nan
    x = rng.normal(size=300)
    polars_frame = pl.DataFrame(
        {"x": x, "kids": pl.Series([None if np.isnan(v) else int(v) for v in kids], dtype=pl.Int64)}
    )
    # As a whole the column converts to float (with NaN), so it is numerical.
    pandas_frame = pd.DataFrame({"x": x, "kids": kids})
    return pandas_frame, polars_frame, x + np.nan_to_num(kids) + rng.normal(scale=0.1, size=300)


def test_search_converts_a_polars_frame_once(frames_with_one_null_integer):
    """Rows of a fold without the null would convert to an int column and be
    detected as categorical, unlike the refit on the whole frame."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    from pretab import RepresentationSearchCV

    pandas_frame, polars_frame, y = frames_with_one_null_integer

    def search(X):
        params = {"cv": KFold(5, shuffle=True, random_state=0), "random_state": 0}
        return RepresentationSearchCV(Ridge(), ["minmax"], **params).fit(X, y)

    assert search(polars_frame).cv_results_ == pytest.approx(search(pandas_frame).cv_results_)


@pytest.mark.parametrize("method", ["minmax", "ple", "bspline"])
def test_cross_fitting_polars_input_matches_pandas_input(frames, method):
    from pretab import CrossFittedTransformer

    pandas_frame, polars_frame, y = frames
    columns = ["income", "plan", "city", "flag"]
    pandas_frame = pandas_frame[columns]
    polars_frame = polars_frame.select(columns)

    def cross_fit(X):
        return CrossFittedTransformer(
            Preprocessor(
                numerical_method=method,
                categorical_method="one-hot",
                output_dim=6,
                output_format="dense",
                random_state=0,
            ),
            n_folds=3,
            random_state=0,
        ).fit_transform(X, y)

    np.testing.assert_allclose(cross_fit(polars_frame), cross_fit(pandas_frame))


def test_cross_fitting_converts_a_polars_frame_once(frames_with_one_null_integer):
    from pretab import CrossFittedTransformer

    pandas_frame, polars_frame, y = frames_with_one_null_integer

    def cross_fit(X):
        wrapped = CrossFittedTransformer(
            Preprocessor(numerical_method="minmax", output_format="dense"), n_folds=5, random_state=0
        )
        return wrapped, wrapped.fit_transform(X, y)

    from_pandas, expected = cross_fit(pandas_frame)
    from_polars, actual = cross_fit(polars_frame)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(from_polars.transform(polars_frame), from_pandas.transform(pandas_frame))
    np.testing.assert_array_equal(from_polars.get_feature_names_out(), from_pandas.get_feature_names_out())
    assert from_polars.get_representation_spec().input_features == ("x", "kids")


@pytest.mark.parametrize("method", ["ple", "bspline", "rbf"])
@pytest.mark.parametrize("policy", ["impute_with_indicator", "separate_state"])
def test_cross_fitting_keeps_single_missing_value_indicators_for_polars(method, policy):
    from pretab import CrossFittedTransformer

    rng = np.random.default_rng(0)
    X = rng.normal(size=200)
    X[7] = np.nan
    pandas_frame = pd.DataFrame({"x": X})
    polars_frame = pl.DataFrame({"x": [None if np.isnan(value) else float(value) for value in X]})
    y = rng.normal(size=len(X))

    def cross_fit(frame):
        wrapped = CrossFittedTransformer(
            Preprocessor(numerical_method=method, missing_policy=policy, output_format="dense", random_state=0),
            n_folds=5,
            random_state=0,
        )
        return wrapped, wrapped.fit_transform(frame, y)

    reference, expected = cross_fit(pandas_frame)
    wrapped, actual = cross_fit(polars_frame)
    assert actual.shape == wrapped.transform(polars_frame).shape == (len(X), 8)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(actual[:, -1], np.isnan(X).astype(float))
    np.testing.assert_array_equal(wrapped.get_feature_names_out(), reference.get_feature_names_out())
    assert polars_frame.null_count()["x"][0] == 1
