"""Shared location-resolution utility for target-aware basis placement.

Feature maps (centers), PLE (thresholds) and the splines (knots) all take a raw
set of candidate locations for one feature and *resolve* it to a count inside a
``[min_count, max_count]`` window: too many locations are trimmed, too few are
supplemented. The splines drive this through ``_adjust_internal_knots``; the
feature-map and PLE families now place locations directly with the shared
count-based selectors.

:func:`resolve_locations` centralizes the skeleton -- sort, optional dedupe,
trim-if-over, supplement-if-under -- while each family keeps its own
*unit conversion* (bins vs knots) and its own *candidate generation* for the
supplement step (passed in as a callback). This is a behavior-preserving
extraction: the trim uses :func:`pretab.core.knots.select_knots` (even spacing)
by default, matching the previous per-family logic exactly.
"""

from collections.abc import Callable

import numpy as np

from .knots import select_knots

__all__ = ["resolve_locations", "trim_to_count"]


def trim_to_count(
    locations: np.ndarray, count: int, importance: np.ndarray | None = None
) -> np.ndarray:
    """Reduce ``locations`` to at most ``count`` entries.

    When ``importance`` is ``None`` the array is down-sampled by even spacing
    (:func:`select_knots`), reproducing the historical center/threshold/knot
    trimming. When ``importance`` is provided, the ``count`` most important
    locations are kept and returned in ascending location order.
    """
    locations = np.asarray(locations)
    if len(locations) <= count:
        return locations
    if importance is None:
        return select_knots(locations, count)
    importance = np.asarray(importance)
    # Keep the `count` highest-importance locations, then restore location order.
    top = np.argsort(importance, kind="stable")[::-1][:count]
    return locations[np.sort(top)]


def resolve_locations(
    locations: np.ndarray,
    *,
    min_count: int,
    max_count: int,
    supplement: Callable[[np.ndarray, int], np.ndarray] | None = None,
    importance: np.ndarray | None = None,
    dedupe: bool = True,
) -> np.ndarray:
    """Resolve candidate ``locations`` into the ``[min_count, max_count]`` window.

    Parameters
    ----------
    locations : ndarray
        Candidate locations for a single feature (e.g. tree split points).
    min_count, max_count : int
        Inclusive bounds on the number of resolved locations.
    supplement : callable, optional
        ``supplement(current, target) -> ndarray`` used to back-fill when fewer
        than ``min_count`` locations remain. Each family supplies its own
        candidate generation here; if ``None``, no supplementation is performed.
    importance : ndarray, optional
        Per-location importance used to keep the top ``max_count`` when trimming.
        When ``None`` (the default), trimming is by even spacing.
    dedupe : bool, default=True
        Whether to drop duplicate locations before resolving. Feature maps and
        splines dedupe; PLE thresholds are already unique and preserve their raw
        ordering, so it passes ``dedupe=False``.

    Returns
    -------
    ndarray
        Sorted locations whose count lies in ``[min_count, max_count]`` (subject
        to the number of distinct candidates the supplement can provide).
    """
    locs = np.sort(np.asarray(locations, dtype=float))
    if dedupe:
        locs = np.unique(locs)
    if len(locs) > max_count:
        locs = trim_to_count(locs, max_count, importance)
    if len(locs) < min_count and supplement is not None:
        locs = supplement(locs, min_count)
    return locs
