"""Shared type aliases for PreTab's public and internal signatures.

Centralizing these keeps transformer, placement and compose signatures consistent
and gives a single place to evolve the accepted input/target types.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd

# Accepted feature-matrix inputs.
ArrayLike = np.ndarray | pd.DataFrame | pd.Series | list

# Accepted supervision targets (``None`` for unsupervised transforms).
TargetLike = np.ndarray | pd.Series | list | None

# Canonical placement-strategy vocabulary (see :mod:`pretab.core.parameters`).
PlacementStrategyName = Literal["uniform", "quantile", "cart", "lightgbm"]

# Supervised-task discriminator used by the supervised placement selectors.
Task = Literal["regression", "classification"]


class TransformerLike(Protocol):
    """Minimal duck-typed transformer interface used by internal wrappers."""

    def fit(self, X: Any, y: Any = ...) -> Any: ...
    def transform(self, X: Any) -> Any: ...
    def get_feature_names_out(self, input_features: Any = ...) -> Any: ...


class PredictorLike(Protocol):
    """Minimal duck-typed supervised-estimator interface (fit + predict)."""

    def fit(self, X: Any, y: Any = ...) -> Any: ...
    def predict(self, X: Any) -> Any: ...


__all__ = [
    "ArrayLike",
    "PlacementStrategyName",
    "PredictorLike",
    "TargetLike",
    "Task",
    "TransformerLike",
]
