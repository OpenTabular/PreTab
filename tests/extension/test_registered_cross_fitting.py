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
from sklearn.preprocessing import TargetEncoder

from pretab import CrossFittedTransformer, Preprocessor, register_representation


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
