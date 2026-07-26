"""Standalone time-series transformers.

These transformers are **not** part of the :class:`~pretab.preprocessor.Preprocessor`
pipeline. ``LagFeatureTransformer`` and ``RollingStatsTransformer`` intentionally
change the row count (they drop the initial, incomplete windows) and assume the
rows are ordered in time, so they cannot be used inside the
:class:`~sklearn.compose.ColumnTransformer` the preprocessor builds.
``CyclicalTimeTransformer`` preserves the row count but requires a per-feature
``period`` argument, so it is also applied directly rather than routed through the
pipeline. Use them standalone on ordered arrays.
"""

from .cyclic import CyclicalTimeTransformer
from .lag import LagFeatureTransformer
from .rolling_stats import RollingStatsTransformer

__all__ = [
    "CyclicalTimeTransformer",
    "LagFeatureTransformer",
    "RollingStatsTransformer",
]
