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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder

from pretab import CrossFittedTransformer, Preprocessor, register_representation
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
