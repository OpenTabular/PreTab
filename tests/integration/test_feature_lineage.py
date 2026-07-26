import warnings

import numpy as np
import pandas as pd
import pytest

from pretab import Preprocessor
from pretab.core.representation import FeatureLineage


@pytest.fixture
def mixed_frame():
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {
            "age": rng.uniform(18, 80, n),
            "income": rng.uniform(1000, 9000, n),
            "score": rng.uniform(0, 1, n),
            "hour": rng.integers(0, 24, n).astype(float),
            "city": rng.choice(["ny", "sf", "la"], n),
            "tier": rng.choice(["a", "b"], n),
        }
    )
    y = (df["income"] / 1000 + rng.normal(0, 1, n)).to_numpy()
    return df, y


def _fit(df, y, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pre = Preprocessor(**kwargs)
        pre.fit(df, y)
    return pre


def test_lineage_covers_all_output_columns_default(mixed_frame):
    df, y = mixed_frame
    pre = _fit(df, y)
    names = list(pre.get_feature_names_out())
    lineage = pre.get_feature_lineage()
    assert len(lineage) == len(names)
    assert [record.output_feature for record in lineage] == names
    assert [record.output_index for record in lineage] == list(range(len(names)))


def test_lineage_records_are_complete(mixed_frame):
    df, y = mixed_frame
    pre = _fit(
        df,
        y,
        feature_preprocessing={
            "age": "bspline",
            "income": "standardization",
            "score": "ple",
            "hour": "rbf",
            "city": "one-hot",
            "tier": "int",
        },
    )
    lineage = pre.get_feature_lineage()
    for record in lineage:
        assert isinstance(record, FeatureLineage)
        assert record.source_features
        assert all(isinstance(source, str) for source in record.source_features)
        assert record.family
        assert record.component
        assert record.component_index >= 0


def test_lineage_marks_supervised_representation(mixed_frame):
    df, y = mixed_frame
    pre = _fit(df, y, feature_preprocessing={"score": "ple"})
    ple_records = [record for record in pre.get_feature_lineage() if record.family == "piecewise_linear"]
    assert ple_records
    assert all(record.uses_target for record in ple_records)


def test_lineage_families_reflect_methods(mixed_frame):
    df, y = mixed_frame
    pre = _fit(
        df,
        y,
        feature_preprocessing={
            "age": "bspline",
            "income": "standardization",
            "score": "ple",
            "hour": "rbf",
            "city": "one-hot",
            "tier": "int",
        },
    )
    families_by_source = {}
    for record in pre.get_feature_lineage():
        families_by_source.setdefault(record.source_features, set()).update([record.family])
    assert families_by_source[("age",)] == {"bspline"}
    assert families_by_source[("income",)] == {"standardization"}
    assert families_by_source[("score",)] == {"piecewise_linear"}
    assert families_by_source[("hour",)] == {"rbf"}
    assert families_by_source[("city",)] == {"onehot"}
    assert families_by_source[("tier",)] == {"ordinal"}


def test_lineage_round_trips_through_dict(mixed_frame):
    df, y = mixed_frame
    pre = _fit(df, y)
    for record in pre.get_feature_lineage():
        data = record.to_dict()
        rebuilt = FeatureLineage(
            output_feature=data["output_feature"],
            output_index=data["output_index"],
            source_features=tuple(data["source_features"]),
            family=data["family"],
            component=data["component"],
            component_index=data["component_index"],
            uses_target=data["uses_target"],
            is_interaction=data["is_interaction"],
        )
        assert rebuilt == record


def test_lineage_source_features_are_single_input_per_block(mixed_frame):
    df, y = mixed_frame
    pre = _fit(df, y)
    for record in pre.get_feature_lineage():
        # The preprocessor expands each column independently, so every output
        # column traces back to exactly one source feature.
        assert len(record.source_features) == 1
        assert not record.is_interaction
