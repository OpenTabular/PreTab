"""Placement subsystem: where basis units go and how many there are.

A single home for location + resolution logic shared by splines, feature maps,
PLE and periodic encoders. Splits the two concerns cleanly: *where* (the
:class:`~pretab.placement.base.BasePlacementStrategy` families) and *how many*
(the :class:`~pretab.placement.resolution.BaseResolutionPolicy` policies). The
family adapters translate a strategy's generic locations into knots / thresholds
/ centers, and :func:`~pretab.placement.factory.create_placement_strategy` builds
a strategy from the public ``target_aware`` / ``placement_strategy`` vocabulary.
"""

from .adapters import (
    PeriodicPlacementAdapter,
    PLEPlacementAdapter,
    RBFPlacementAdapter,
    SplinePlacementAdapter,
)
from .base import BasePlacementStrategy, PlacementResult
from .factory import create_placement_strategy
from .resolution import (
    BaseResolutionPolicy,
    FixedResolution,
)
from .supervised import CARTPlacement, LightGBMPlacement
from .unsupervised import QuantilePlacement, UniformPlacement

__all__ = [
    "BasePlacementStrategy",
    "BaseResolutionPolicy",
    "CARTPlacement",
    "FixedResolution",
    "LightGBMPlacement",
    "PLEPlacementAdapter",
    "PeriodicPlacementAdapter",
    "PlacementResult",
    "QuantilePlacement",
    "RBFPlacementAdapter",
    "SplinePlacementAdapter",
    "UniformPlacement",
    "create_placement_strategy",
]
