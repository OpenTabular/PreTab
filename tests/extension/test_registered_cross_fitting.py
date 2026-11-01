"""Cross-fitting a Preprocessor that uses a registered, target-consuming class.

Regression guard: a wrapped Preprocessor refits only its target-aware blocks per
fold, but a registered class without a ``RepresentationSpec`` (a plain
scikit-learn transformer such as ``TargetEncoder``) was never recognised as
target-aware. Its out-of-fold rows were then encoded by the all-data fit, which
had seen their own targets.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, TargetEncoder

from pretab import CrossFittedTransformer, Preprocessor, register_representation
from pretab.compose.inspection import block_uses_target, refit_block_representation
from pretab.exceptions import IncompatibleParamsError


class _MeanEncoder(TransformerMixin, BaseEstimator):
    """Plain, optionally supervised encoder: category -> mean target when target-aware."""

    def __init__(self, target_aware=True):
        self.target_aware = target_aware

    def fit(self, X, y=None):
        x = np.asarray(X, dtype=object)[:, 0]
        if self.target_aware:
            y = np.asarray(y, dtype=float)
            self.mapping_ = {c: y[x == c].mean() for c in np.unique(x)}
        else:
            self.mapping_ = {c: float(i) for i, c in enumerate(np.unique(x))}
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        return np.array([[self.mapping_.get(c, 0.0)] for c in np.asarray(X, dtype=object)[:, 0]])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else ["x0"], dtype=object)


class _OffByDefaultMeanEncoder(_MeanEncoder):
    def __init__(self, target_aware=False):
        super().__init__(target_aware=target_aware)


class _ChangingWidthEncoder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        x = np.asarray(X, dtype=float)
        self.n_features_in_ = x.shape[1]
        self.width_ = 2 if len(x) == 100 else (3 if x.mean() > 0 else 1)
        self.target_mean_ = np.asarray(y, dtype=float).mean()
        return self

    def transform(self, X):
        return np.repeat(np.asarray(X, dtype=float) + self.target_mean_, self.width_, axis=1)

    def get_feature_names_out(self, input_features=None):
        name = "x0" if input_features is None else input_features[0]
        return np.array([f"{name}_{i}" for i in range(self.width_)], dtype=object)


@pytest.fixture
def noise_target_data():
    """An id column with four rows per id and a target of pure noise."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=400), "id": np.repeat([f"id{i}" for i in range(100)], 4)})
    return X, rng.normal(size=400)


@pytest.mark.parametrize(
    ("cls", "supervision"),
    [(TargetEncoder, "supervised"), (_MeanEncoder, "optional")],
)
def test_registered_target_encoder_is_refit_per_fold(noise_target_data, cls, supervision):
    X, y = noise_target_data
    register_representation("registered_encoder", cls, feature_kind="categorical", supervision=supervision)
    pre = Preprocessor(numerical_method="minmax", categorical_method="registered_encoder", random_state=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_of_fold = np.asarray(CrossFittedTransformer(pre, n_folds=5, random_state=0).fit_transform(X, y))
        lineage = {item.output_feature: item.uses_target for item in pre.fit(X, y).get_feature_lineage()}

    # An encoding fit without each row's own target cannot predict pure noise; the
    # all-data fit correlated at about 0.5.
    assert abs(np.corrcoef(out_of_fold[:, -1], y)[0, 1]) < 0.2
    assert lineage["cat_id"] is True


def test_registered_target_encoder_keeps_its_missing_indicator():
    """The registered encoder is refit on the imputed value; its raw indicator stays alongside it."""
    rng = np.random.default_rng(0)
    X = rng.choice(["a", "b", "c"], (200, 1)).astype(object)
    X[7, 0] = np.nan
    y = rng.normal(size=200)
    register_representation("registered_encoder", TargetEncoder, feature_kind="categorical", supervision="supervised")
    pre = Preprocessor(
        categorical_method="registered_encoder", add_missing_indicator=True, output_format="dense", random_state=0
    )

    cross_fitted = CrossFittedTransformer(pre, n_folds=5, random_state=0)
    out = cross_fitted.fit_transform(X, y)
    full = cross_fitted.transform(X)

    assert list(cross_fitted.get_feature_names_out()) == ["cat_feature_0", "cat_feature_0__missingindicator_feature_0"]
    assert out.shape == full.shape == (200, 2)
    assert not np.allclose(out, full)  # refit on every fold
    np.testing.assert_array_equal(out[:, 1], pd.isna(X[:, 0]).astype(float))


@pytest.mark.parametrize("target_aware", [False, True])
@pytest.mark.parametrize("boolean_input", [False, True])
def test_optional_class_follows_the_preprocessor_target_aware_across_folds(
    noise_target_data, target_aware, boolean_input
):
    X, y = noise_target_data
    if boolean_input:
        X = X.assign(id=np.arange(len(X)) % 2 == 0)
    register_representation(
        "off_by_default_encoder", _OffByDefaultMeanEncoder, feature_kind="categorical", supervision="optional"
    )
    cross_fitted = CrossFittedTransformer(
        Preprocessor(
            numerical_method="minmax",
            categorical_method="off_by_default_encoder",
            target_aware=target_aware,
            placement_strategy="cart" if target_aware else "quantile",
            output_format="dense",
            random_state=0,
        ),
        n_folds=5,
        random_state=0,
    )
    out = cross_fitted.fit_transform(X, y)
    full = cross_fitted.transform(X)
    fitted = cross_fitted.estimator_
    assert isinstance(fitted, Preprocessor)
    encoder = fitted.column_transformer_.named_transformers_["cat_id"].steps[-1][1]
    assert encoder.target_aware is target_aware
    lineage = {item.output_feature: item.uses_target for item in fitted.get_feature_lineage()}
    assert lineage["cat_id"] is target_aware
    if not target_aware:
        np.testing.assert_array_equal(out, full)
    else:
        expected = np.empty(len(X))
        for train_idx, test_idx in KFold(n_splits=5, shuffle=True, random_state=0).split(X):
            encoder = _OffByDefaultMeanEncoder(target_aware=True).fit(X.iloc[train_idx][["id"]], y[train_idx])
            expected[test_idx] = encoder.transform(X.iloc[test_idx][["id"]]).ravel()
        np.testing.assert_allclose(out[:, -1], expected)
        assert abs(np.corrcoef(out[:, -1], y)[0, 1]) < 0.2
        assert not np.allclose(out[:, -1], full[:, -1])


def test_registered_target_using_override_of_a_builtin_is_refit_per_fold(noise_target_data):
    _, y = noise_target_data
    X = pd.DataFrame({"x": np.repeat(np.arange(100, dtype=float), 4)})
    register_representation("minmax", _MeanEncoder, feature_kind="numerical", supervision="optional", override=True)
    cross_fitted = CrossFittedTransformer(
        Preprocessor(numerical_method="minmax", output_format="dense"), n_folds=5, random_state=0
    )
    out = cross_fitted.fit_transform(X, y)
    assert abs(np.corrcoef(out[:, 0], y)[0, 1]) < 0.2
    fitted = cross_fitted.estimator_
    assert isinstance(fitted, Preprocessor)
    assert fitted.get_feature_lineage()[0].uses_target is True


def test_registered_method_named_like_a_helper_does_not_make_the_helper_target_aware(noise_target_data):
    X, y = noise_target_data
    X = X.copy()
    X.loc[7, "id"] = "rare"
    register_representation("to_float", _MeanEncoder, feature_kind="categorical", supervision="supervised")
    cross_fitted = CrossFittedTransformer(
        Preprocessor(numerical_method="minmax", categorical_method="one-hot", output_format="dense"),
        n_folds=5,
        random_state=0,
    )
    np.testing.assert_array_equal(cross_fitted.fit_transform(X, y), cross_fitted.transform(X))
    fitted = cross_fitted.estimator_
    assert isinstance(fitted, Preprocessor)
    assert not any(item.uses_target for item in fitted.get_feature_lineage())


def test_opposite_block_width_changes_are_rejected_even_when_total_width_is_fixed():
    register_representation("changing_width", _ChangingWidthEncoder, feature_kind="numerical", supervision="supervised")
    X = pd.DataFrame({"a": np.linspace(1, 2, 100), "b": np.linspace(-2, -1, 100)})
    y = np.random.default_rng(0).normal(size=100)
    cross_fitted = CrossFittedTransformer(
        Preprocessor(numerical_method="changing_width", scaling="none", output_format="dense"),
        n_folds=5,
        random_state=0,
    )
    # Full-data widths are (2, 2); fold widths (3, 1) keep the same total but move the feature boundary.
    with pytest.raises(IncompatibleParamsError, match=r"block 'num_a'.*expected 2, got 3"):
        cross_fitted.fit_transform(X, y)


@pytest.mark.parametrize("leading_helpers", [False, True])
@pytest.mark.parametrize("trailing_scaler", ["none", "suffix", "scaler"])
@pytest.mark.parametrize("missing_union", [False, True])
def test_refit_representation_shares_only_the_prefix_and_missing_branch(
    leading_helpers, trailing_scaler, missing_union
):
    register_representation("pipeline_mean", _MeanEncoder, feature_kind="numerical", supervision="optional")
    X = pd.DataFrame({"x": np.repeat(np.arange(20, dtype=float), 10)})
    y = np.random.default_rng(0).normal(size=len(X))
    steps = []
    if leading_helpers:
        X.loc[7, "x"] = np.nan
        steps.extend([("imputer", SimpleImputer()), ("scale", MinMaxScaler())])
    prefix_length = len(steps)
    steps.append(("pipeline_mean", _MeanEncoder()))
    if trailing_scaler != "none":
        steps.append((trailing_scaler, MinMaxScaler()))
    pipeline = Pipeline(steps)
    fitted = (
        FeatureUnion([("representation", pipeline), ("missing", MissingIndicator(features="all"))])
        if missing_union
        else pipeline
    ).fit(X, y)
    assert block_uses_target(fitted)
    original = fitted.transform(X).copy()
    original_input = X.copy(deep=True)
    train, held_out = X.iloc[40:], X.iloc[:40]
    refitted = refit_block_representation(fitted, train, y[40:])
    assert refitted is not fitted
    if missing_union:
        assert isinstance(fitted, FeatureUnion)
        assert isinstance(refitted, FeatureUnion)
        old_branches = dict(fitted.transformer_list)
        new_branches = dict(refitted.transformer_list)
        assert new_branches["missing"] is old_branches["missing"]
        new_pipeline = new_branches["representation"]
        np.testing.assert_array_equal(refitted.transform(held_out)[:, -1], pd.isna(held_out["x"]).astype(float))
    else:
        new_pipeline = refitted
    assert isinstance(new_pipeline, Pipeline)
    for index, (_, step) in enumerate(pipeline.steps):
        assert (new_pipeline.steps[index][1] is step) is (index < prefix_length)

    prefix = pipeline[:prefix_length] if prefix_length else None
    prepared_train = prefix.transform(train) if prefix is not None else train
    prepared_test = prefix.transform(held_out) if prefix is not None else held_out
    expected = clone(pipeline[prefix_length:])
    assert isinstance(expected, Pipeline)
    expected.fit(prepared_train, y[40:])
    np.testing.assert_allclose(new_pipeline.transform(held_out), expected.transform(prepared_test))
    np.testing.assert_array_equal(fitted.transform(X), original)
    pd.testing.assert_frame_equal(X, original_input)
    pd.testing.assert_frame_equal(train, original_input.iloc[40:])


def test_refit_a_bare_encoder_does_not_change_its_all_data_fit(noise_target_data):
    X, y = noise_target_data
    values = X[["id"]]
    fitted = _MeanEncoder().fit(values, y)
    original = fitted.transform(values).copy()
    refitted = refit_block_representation(fitted, values.iloc[80:], y[80:])
    assert isinstance(refitted, _MeanEncoder)
    assert refitted is not fitted
    expected = _MeanEncoder().fit(values.iloc[80:], y[80:]).transform(values.iloc[:80])
    np.testing.assert_array_equal(refitted.transform(values.iloc[:80]), expected)
    np.testing.assert_array_equal(fitted.transform(values), original)


@pytest.mark.parametrize("target_kind", ["continuous", "binary", "multiclass"])
def test_named_boolean_target_encoder_keeps_its_indicator_and_excludes_held_out_targets(target_kind):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"flag": rng.choice([True, False], size=200).astype(object)}, index=np.arange(200) + 1000)
    X.iloc[7, 0] = np.nan
    original = X.copy(deep=True)
    if target_kind == "continuous":
        y = rng.normal(size=len(X))
        task = "regression"
        splitter = KFold(5, shuffle=True, random_state=0)
    else:
        classes = np.array(["A", "B"] if target_kind == "binary" else ["A", "B", "C"])
        y = classes[np.arange(len(X)) % len(classes)]
        task = "classification"
        splitter = StratifiedKFold(5, shuffle=True, random_state=0)
    register_representation("registered_encoder", TargetEncoder, feature_kind="categorical", supervision="supervised")
    cross_fitted = CrossFittedTransformer(
        Preprocessor(
            categorical_method="registered_encoder",
            add_missing_indicator=True,
            output_format="dense",
            task=task,
            random_state=0,
        ),
        n_folds=5,
        task=task,
        random_state=0,
    )
    out = cross_fitted.fit_transform(X, y)
    width = 3 if target_kind == "multiclass" else 1
    assert out.shape == cross_fitted.transform(X).shape == (len(X), width + 1)
    fitted = cross_fitted.estimator_
    assert isinstance(fitted, Preprocessor)
    branches = dict(fitted.column_transformer_.named_transformers_["cat_flag"].transformer_list)
    pipeline = branches["representation"]
    prepared = pipeline[:-1].transform(X)
    expected = np.empty((len(X), width))
    for train_idx, test_idx in splitter.split(X, y):
        encoder = clone(pipeline[-1])
        assert isinstance(encoder, TargetEncoder)
        encoder.fit(prepared[train_idx], y[train_idx])
        expected[test_idx] = encoder.transform(prepared[test_idx])
    np.testing.assert_allclose(out[:, :width], expected)
    np.testing.assert_array_equal(out[:, -1], pd.isna(X["flag"]).astype(float))
    assert [item.uses_target for item in fitted.get_feature_lineage()] == [True] * width + [False]
    pd.testing.assert_frame_equal(X, original)
