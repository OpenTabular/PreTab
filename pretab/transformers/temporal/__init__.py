"""Standalone time-series transformers.

These transformers are **not** part of the :class:`~pretab.preprocessor.Preprocessor`
pipeline. ``LagFeatureTransformer`` and ``RollingStatsTransformer`` intentionally
change the row count (they drop the initial, incomplete windows) and assume the
rows are ordered in time, so they cannot be used inside the
:class:`~sklearn.compose.ColumnTransformer` the preprocessor builds. Use them
standalone on ordered arrays.
"""

from .lag import LagFeatureTransformer
from .rolling_stats import RollingStatsTransformer

__all__ = [
    "LagFeatureTransformer",
    "RollingStatsTransformer",
]
