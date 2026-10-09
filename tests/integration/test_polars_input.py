"""Polars DataFrames as Preprocessor input.

Regression guard: a polars frame used to be treated as a pandas frame and failed
with ``AttributeError: 'list' object has no attribute 'duplicated'``, so a
scikit-learn ``Pipeline`` with ``set_output(transform="polars")`` crashed when it
handed its polars output to the Preprocessor.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from pretab import Preprocessor

pl = pytest.importorskip("polars")


@pytest.fixture
def frames():
    """The same mixed-type data as a pandas and as a polars frame."""
    rng = np.random.default_rng(0)
    n = 120
    pandas_frame = pd.DataFrame(
        {
            "income": rng.normal(5e4, 1e4, n),
            "visits": rng.integers(0, 1000, n),
            "rooms": rng.integers(1, 4, n),
            "member": rng.random(n) > 0.5,
            "city": rng.choice(["a", "b", "c"], n),
        }
    )
    polars_frame = pl.DataFrame(
        {
            "income": pandas_frame["income"].to_numpy(),
            "visits": pandas_frame["visits"].to_numpy(),
            "rooms": pandas_frame["rooms"].to_numpy(),
            "member": pandas_frame["member"].to_numpy(),
            "city": pandas_frame["city"].tolist(),
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
    assert from_polars.categorical_features_ == ["rooms", "member", "city"]
    assert from_polars.categorical_features_ == from_pandas.categorical_features_
    assert list(from_polars.feature_names_in_) == polars_frame.columns
    np.testing.assert_array_equal(from_polars.get_feature_names_out(), from_pandas.get_feature_names_out())

    expected = np.asarray(from_pandas.transform(pandas_frame))
    np.testing.assert_allclose(np.asarray(from_polars.transform(polars_frame)), expected)
    np.testing.assert_allclose(np.asarray(Preprocessor(**params).fit_transform(polars_frame, y)), expected)
    # Columns are matched by name, so a reordered polars frame transforms the same.
    reordered = polars_frame.select(polars_frame.columns[::-1])
    np.testing.assert_allclose(np.asarray(from_polars.transform(reordered)), expected)


def test_polars_input_with_polars_output(frames):
    _, polars_frame, y = frames
    pre = Preprocessor(numerical_method="minmax").set_output(transform="polars").fit(polars_frame, y)
    out = pre.transform(polars_frame)
    assert isinstance(out, pl.DataFrame)
    assert out.columns == list(pre.get_feature_names_out())
    assert out.shape == (polars_frame.height, pre.total_output_dim_)


def test_polars_nulls_are_missing_values():
    frame = pl.DataFrame({"x": [1.0, None, 3.0, 4.0] * 10, "city": ["a", None, "b", "a"] * 10})
    pre = Preprocessor(numerical_method="minmax", categorical_method="one-hot", missing_policy="impute_with_indicator")
    pre.fit(frame)
    names = list(pre.get_feature_names_out())
    # The null is imputed and flagged, not encoded as a "None" category.
    assert any("missingindicator_city" in name for name in names)
    assert not any("None" in name for name in names)


@pytest.mark.parametrize("container", ["pandas", "polars"])
def test_pipeline_set_output_hands_frames_to_the_preprocessor(frames, container):
    pandas_frame, _, y = frames
    numeric = pandas_frame[["income", "visits"]]
    pipe = Pipeline([("impute", SimpleImputer()), ("pretab", Preprocessor(numerical_method="minmax"))])
    pipe.set_output(transform=container)

    out = pipe.fit_transform(numeric, y)

    assert type(out).__module__.split(".")[0] == container
    assert out.shape == (len(numeric), 2)
    assert list(out.columns) == ["num_income", "num_visits"]
    np.testing.assert_allclose(np.asarray(out), np.asarray(pipe.transform(numeric)))


def test_representation_search_on_polars_matches_pandas(frames):
    """Each fold's Preprocessor gets a polars frame too, not an object array that
    would turn every column categorical (and score a different model than the
    final refit)."""
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


def test_cross_fitted_preprocessor_on_polars_matches_pandas(frames):
    """A polars frame reaches the wrapped Preprocessor as a frame, so its column
    types are detected as for the equivalent pandas frame."""
    from pretab import CrossFittedTransformer

    pandas_frame, polars_frame, y = frames

    def cross_fit(X):
        cross_fitted = CrossFittedTransformer(Preprocessor(output_dim=4, random_state=0), n_folds=3, random_state=0)
        return cross_fitted, cross_fitted.fit_transform(X, y)

    (from_pandas, expected), (from_polars, out) = cross_fit(pandas_frame), cross_fit(polars_frame)

    np.testing.assert_array_equal(from_polars.get_feature_names_out(), from_pandas.get_feature_names_out())
    np.testing.assert_allclose(np.asarray(out, dtype=float), np.asarray(expected, dtype=float))
    np.testing.assert_allclose(
        np.asarray(from_polars.transform(polars_frame), dtype=float),
        np.asarray(from_pandas.transform(pandas_frame), dtype=float),
    )


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


def test_cross_fitting_converts_a_polars_frame_once(frames_with_one_null_integer):
    """The clone-and-refit path takes every fold's rows from one conversion."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer

    from pretab import CrossFittedTransformer

    pandas_frame, polars_frame, y = frames_with_one_null_integer

    def cross_fit(X):
        pipeline = make_pipeline(
            FunctionTransformer(feature_names_out="one-to-one"), Preprocessor(numerical_method="minmax")
        )
        return np.asarray(CrossFittedTransformer(pipeline, n_folds=5, random_state=0).fit_transform(X, y), dtype=float)

    np.testing.assert_allclose(cross_fit(polars_frame), cross_fit(pandas_frame))
