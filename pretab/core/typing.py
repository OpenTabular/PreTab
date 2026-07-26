"""Shared type aliases for PreTab's public and internal signatures.

Centralizing these keeps transformer, placement and compose signatures consistent
and gives a single place to evolve the accepted input/target types.
"""

from __future__ import annotations

from typing import Literal

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

__all__ = [
    "ArrayLike",
    "PlacementStrategyName",
    "TargetLike",
    "Task",
]
